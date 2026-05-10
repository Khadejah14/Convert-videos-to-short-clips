from __future__ import annotations

import logging
import subprocess
import json
import os
import tempfile
from pathlib import Path

from app.agent.schemas import EditDecision, EffectType, EditingPlan, AgentConfig

logger = logging.getLogger(__name__)


class EffectsEngine:
    def __init__(self):
        self._effect_handlers = {
            EffectType.ZOOM_IN: self._apply_zoom,
            EffectType.ZOOM_OUT: self._apply_zoom,
            EffectType.PUNCH_IN: self._apply_punch_in,
            EffectType.SHAKE: self._apply_shake,
            EffectType.FLASH: self._apply_flash,
            EffectType.BLUR_TRANSITION: self._apply_blur_transition,
            EffectType.CROSS_DISSOLVE: self._apply_cross_dissolve,
        }

    def apply_effects(
        self,
        video_path: str,
        plan: EditingPlan,
        config: AgentConfig,
        output_path: str,
    ) -> str:
        if not plan.effects:
            return video_path

        effects = sorted(plan.effects, key=lambda e: e.timestamp)

        filter_chains = []
        for i, effect in enumerate(effects):
            handler = self._effect_handlers.get(effect.effect)
            if handler:
                f = handler(effect, config)
                if f:
                    filter_chains.append(f)

        if not filter_chains:
            return video_path

        combined = ",".join(filter_chains)

        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", combined,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=600)
            logger.info(f"Applied {len(filter_chains)} effects")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Effects render failed: {e.stderr[:500]}")
            return self._apply_effects_individually(video_path, effects, config, output_path)

    def _apply_effects_individually(
        self,
        video_path: str,
        effects: list[EditDecision],
        config: AgentConfig,
        output_path: str,
    ) -> str:
        current = video_path
        for i, effect in enumerate(effects):
            handler = self._effect_handlers.get(effect.effect)
            if not handler:
                continue
            f = handler(effect, config)
            if not f:
                continue

            if i < len(effects) - 1:
                tmp = tempfile.NamedTemporaryFile(suffix=f"_fx{i}.mp4", delete=False)
                tmp.close()
                out = tmp.name
            else:
                out = output_path

            cmd = [
                "ffmpeg", "-y", "-i", current,
                "-vf", f,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "copy", out,
            ]
            try:
                subprocess.run(cmd, capture_output=True, check=True, timeout=300)
                if current != video_path:
                    try:
                        os.unlink(current)
                    except OSError:
                        pass
                current = out
            except subprocess.CalledProcessError as e:
                logger.warning(f"Effect {effect.effect} failed: {e}")

        return current

    def _apply_zoom(self, decision: EditDecision, config: AgentConfig) -> str | None:
        intensity = decision.intensity * config.zoom_intensity
        if decision.effect == EffectType.ZOOM_IN:
            zoom_factor = 1.0 + (intensity * 0.3)
        else:
            zoom_factor = 1.0 - (intensity * 0.2)
            zoom_factor = max(0.8, zoom_factor)

        ts = decision.timestamp
        dur = decision.duration

        return (
            f"zoompan=z='min({zoom_factor}+on*0.001,{zoom_factor + 0.1})'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={int(dur * 30)}:s=1080x1920:fps=30"
        )

    def _apply_punch_in(self, decision: EditDecision, config: AgentConfig) -> str | None:
        scale = config.punch_in_scale
        intensity = decision.intensity
        ts = decision.timestamp
        dur = decision.duration

        zoom_start = 1.0
        zoom_peak = scale * intensity
        frames = int(dur * 30)

        return (
            f"zoompan=z='if(lte(on,{frames // 3}),"
            f"{zoom_start}+on*({zoom_peak}-{zoom_start})/{frames // 3},"
            f"if(lte(on,{frames * 2 // 3}),"
            f"{zoom_peak},"
            f"{zoom_peak}-({zoom_peak}-{zoom_start})*(on-{frames * 2 // 3})/{frames // 3}))'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s=1080x1920:fps=30"
        )

    def _apply_shake(self, decision: EditDecision, config: AgentConfig) -> str | None:
        intensity = decision.intensity * 5
        return (
            f"crop=in_w-{int(intensity * 4)}:in_h-{int(intensity * 4)}:"
            f"'(iw-{int(intensity * 4)})/2+{intensity}*sin(t*15)'"
            f":'(ih-{int(intensity * 4)})/2+{intensity}*cos(t*12)'"
        )

    def _apply_flash(self, decision: EditDecision, config: AgentConfig) -> str | None:
        ts = decision.timestamp
        dur = 0.05
        return (
            f"eq=brightness=0:enable='between(t,{ts},{ts + dur})'"
        )

    def _apply_blur_transition(self, decision: EditDecision, config: AgentConfig) -> str | None:
        ts = decision.timestamp
        dur = decision.duration
        radius = int(5 * decision.intensity)
        return (
            f"gblur=sigma='{radius}*min((t-{ts})/{dur * 0.3},({ts + dur}-t)/{dur * 0.3},1)'"
            f":enable='between(t,{ts},{ts + dur})'"
        )

    def _apply_cross_dissolve(self, decision: EditDecision, config: AgentConfig) -> str | None:
        ts = decision.timestamp
        dur = decision.duration
        return (
            f"fade=t=in:st={ts}:d={dur * 0.5}:alpha=0,"
            f"fade=t=out:st={ts + dur * 0.5}:d={dur * 0.5}:alpha=0"
        )

    def generate_zoom_for_hook(
        self,
        timestamp: float,
        duration: float,
        intensity: float,
    ) -> EditDecision:
        return EditDecision(
            timestamp=timestamp,
            duration=min(duration, 2.0),
            effect=EffectType.PUNCH_IN,
            intensity=min(1.0, intensity),
            reason="Auto-generated: hook detected at this timestamp",
        )

    def generate_effects_for_beats(
        self,
        plan: EditingPlan,
        config: AgentConfig,
    ) -> list[EditDecision]:
        effects = []
        max_per_min = config.max_effects_per_minute
        duration_min = plan.total_duration / 60.0
        max_effects = max(1, int(max_per_min * duration_min))

        for beat in plan.beats:
            if len(effects) >= max_effects:
                break

            if beat.beat_type == "hook" and config.enable_zoom_effects:
                effects.append(EditDecision(
                    timestamp=beat.start,
                    duration=min(beat.end - beat.start, 2.0),
                    effect=EffectType.PUNCH_IN,
                    intensity=0.7,
                    reason=f"Auto: {beat.beat_type} beat",
                ))
            elif beat.beat_type == "climax" and config.enable_punch_in:
                effects.append(EditDecision(
                    timestamp=beat.start,
                    duration=min(beat.end - beat.start, 1.5),
                    effect=EffectType.PUNCH_IN,
                    intensity=0.9,
                    reason="Auto: climax beat - maximum punch",
                ))
            elif beat.beat_type == "transition":
                effects.append(EditDecision(
                    timestamp=beat.start,
                    duration=0.3,
                    effect=EffectType.BLUR_TRANSITION,
                    intensity=0.5,
                    reason="Auto: transition beat",
                ))
            elif beat.energy > 0.8 and config.enable_zoom_effects:
                effects.append(EditDecision(
                    timestamp=beat.start,
                    duration=min(beat.end - beat.start, 1.5),
                    effect=EffectType.ZOOM_IN,
                    intensity=beat.energy,
                    reason="Auto: high energy segment",
                ))

        return effects[:max_effects]
