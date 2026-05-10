from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import openai

from app.agent.schemas import (
    EditingStyle,
    CaptionStyle,
    EditingPlan,
    AgentConfig,
    AgentResult,
    TranscriptSegment,
    HookDetection,
    EditDecision,
    EffectType,
)
from app.agent.analyzer import ContentAnalyzer
from app.agent.silence import SilenceDetector
from app.agent.effects import EffectsEngine
from app.agent.captions import DynamicCaptionEngine
from app.agent.pacing import PacingOptimizer
from app.agent.styles import StyleManager
from app.agent.plugins import PluginManager

logger = logging.getLogger(__name__)


class EditingAgent:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        plugins_dir: str | None = None,
        custom_styles_dir: str | None = None,
    ):
        self.api_key = api_key
        self.analyzer = ContentAnalyzer(api_key=api_key, model=model)
        self.silence_detector = SilenceDetector()
        self.effects_engine = EffectsEngine()
        self.caption_engine = DynamicCaptionEngine()
        self.pacing_optimizer = PacingOptimizer()
        self.style_manager = StyleManager(custom_styles_dir)
        self.plugin_manager = PluginManager()

        if plugins_dir:
            self.plugin_manager.load_from_directory(plugins_dir)

    async def edit(
        self,
        video_path: str,
        transcript_segments: list[dict[str, Any]],
        full_transcript: str,
        config: AgentConfig,
        output_dir: str,
        job_id: uuid.UUID | None = None,
    ) -> AgentResult:
        start_time = time.time()

        if config.style != EditingStyle.CUSTOM:
            config = self.style_manager.get_config_for_style(config.style, config)

        if config.plugins:
            for plugin_path in config.plugins:
                self.plugin_manager.load_from_path(plugin_path)

        self.plugin_manager.init_all(config)

        os.makedirs(output_dir, exist_ok=True)
        uid = job_id or uuid.uuid4()
        duration_before = self._get_duration(video_path)

        logger.info(f"[Agent] Starting edit pipeline for {video_path}")
        logger.info(f"[Agent] Style: {config.style.value}, Duration: {duration_before:.1f}s")

        # Step 1: Analyze content
        logger.info("[Agent] Step 1/6: Analyzing transcript...")
        segments, hooks, keywords = await self.analyzer.analyze(
            transcript_segments, full_transcript, config
        )

        # Step 2: Detect and remove silence
        silence_segments = []
        working_video = video_path
        if config.enable_silence_removal:
            logger.info("[Agent] Step 2/6: Detecting silence...")
            silence_segments = self.silence_detector.detect_from_video(
                video_path, config
            )
            if silence_segments:
                silenced_path = os.path.join(output_dir, f"{uid}_silenced.mp4")
                working_video = self.silence_detector.remove_silences(
                    video_path, silence_segments,
                    config.silence_padding_ms, silenced_path,
                )
                logger.info(f"[Agent] Removed {len(silence_segments)} silence segments")
        else:
            logger.info("[Agent] Step 2/6: Silence removal disabled, skipping")

        # Step 3: Build editing plan
        logger.info("[Agent] Step 3/6: Building editing plan...")
        plan = EditingPlan(
            style=config.style,
            total_duration=self._get_duration(working_video),
            hooks=hooks,
            keyword_highlights=keywords,
            caption_style=config.caption_style,
            segments=segments,
            silence_segments=silence_segments,
        )

        # Step 4: Optimize pacing
        if config.enable_pacing_optimization:
            logger.info("[Agent] Step 4/6: Optimizing pacing...")
            plan = self.pacing_optimizer.optimize(plan, config)
            retention_score = self.pacing_optimizer.compute_retention_score(plan)
            logger.info(f"[Agent] Retention score: {retention_score:.2f}")
        else:
            logger.info("[Agent] Step 4/6: Pacing optimization disabled, skipping")

        # Step 5: Generate and apply effects
        logger.info("[Agent] Step 5/6: Applying effects...")
        if config.enable_zoom_effects or config.enable_punch_in:
            auto_effects = self.effects_engine.generate_effects_for_beats(plan, config)
            plan.effects.extend(auto_effects)

            for hook in hooks:
                if hook.strength > 0.6:
                    plan.effects.append(
                        self.effects_engine.generate_zoom_for_hook(
                            hook.timestamp, hook.duration, hook.strength
                        )
                    )

        plan = self.plugin_manager.run_plan_hooks(plan)
        plan = self.plugin_manager.run_pre_render_hooks(plan)

        effects_path = working_video
        if plan.effects:
            effects_path = os.path.join(output_dir, f"{uid}_effects.mp4")
            effects_path = self.effects_engine.apply_effects(
                working_video, plan, config, effects_path,
            )

        # Step 6: Add captions
        logger.info("[Agent] Step 6/6: Rendering captions...")
        final_path = os.path.join(output_dir, f"{uid}_final.mp4")
        if plan.segments:
            final_path = self.caption_engine.render(
                effects_path, plan, config, final_path,
            )
        elif effects_path != video_path:
            import shutil
            shutil.copy2(effects_path, final_path)

        final_path = self.plugin_manager.run_post_render_hooks(final_path)

        duration_after = self._get_duration(final_path)
        processing_time = time.time() - start_time

        self.plugin_manager.cleanup_all()

        result = AgentResult(
            job_id=uid,
            output_path=final_path,
            style=config.style,
            plan=plan,
            effects_applied=len(plan.effects),
            silences_removed=len(silence_segments),
            duration_before=duration_before,
            duration_after=duration_after,
            processing_time=processing_time,
        )

        logger.info(
            f"[Agent] Complete: {duration_before:.1f}s -> {duration_after:.1f}s, "
            f"{len(plan.effects)} effects, {len(silence_segments)} silences removed, "
            f"took {processing_time:.1f}s"
        )

        return result

    async def analyze_only(
        self,
        transcript_segments: list[dict[str, Any]],
        full_transcript: str,
        config: AgentConfig,
    ) -> EditingPlan:
        segments, hooks, keywords = await self.analyzer.analyze(
            transcript_segments, full_transcript, config
        )

        plan = EditingPlan(
            style=config.style,
            total_duration=segments[-1].end if segments else 0,
            hooks=hooks,
            keyword_highlights=keywords,
            caption_style=config.caption_style,
            segments=segments,
        )

        if config.enable_pacing_optimization:
            plan = self.pacing_optimizer.optimize(plan, config)

        auto_effects = self.effects_engine.generate_effects_for_beats(plan, config)
        plan.effects.extend(auto_effects)

        return plan

    def _get_duration(self, video_path: str) -> float:
        import subprocess
        import json

        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                video_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
        except Exception:
            return 0.0

    def get_available_styles(self) -> list[dict[str, str]]:
        return self.style_manager.list_styles()

    def get_available_plugins(self) -> list[str]:
        return self.plugin_manager.loaded_plugins
