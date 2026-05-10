from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from app.agent.schemas import (
    TranscriptSegment,
    KeywordHighlight,
    CaptionStyle,
    EditingPlan,
    AgentConfig,
)

logger = logging.getLogger(__name__)

FONT_MAP = {
    "windows": {
        "bold": "C:/Windows/Fonts/arialbd.ttf",
        "regular": "C:/Windows/Fonts/arial.ttf",
    },
    "linux": {
        "bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "regular": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    },
}


def _find_font(bold: bool = True) -> str:
    import platform
    system = platform.system().lower()
    key = "bold" if bold else "regular"

    if system == "windows":
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/impact.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]

    for f in candidates:
        if os.path.exists(f):
            return f
    return "Arial"


STYLE_CONFIGS: dict[CaptionStyle, dict[str, Any]] = {
    CaptionStyle.DEFAULT: {
        "font_size_ratio": 0.05,
        "font_color": "white",
        "bg_color": "black@0.7",
        "stroke_color": "black",
        "stroke_width": 2,
        "position_y": 0.80,
        "word_highlight_color": "#FFD700",
    },
    CaptionStyle.MINIMAL: {
        "font_size_ratio": 0.04,
        "font_color": "white",
        "bg_color": None,
        "stroke_color": "black",
        "stroke_width": 1,
        "position_y": 0.85,
        "word_highlight_color": "#FFD700",
    },
    CaptionStyle.HIGHLIGHT: {
        "font_size_ratio": 0.06,
        "font_color": "white",
        "bg_color": "black@0.8",
        "stroke_color": "black",
        "stroke_width": 3,
        "position_y": 0.75,
        "word_highlight_color": "#FF4444",
    },
    CaptionStyle.KARAOKE: {
        "font_size_ratio": 0.055,
        "font_color": "#888888",
        "bg_color": "black@0.6",
        "stroke_color": "black",
        "stroke_width": 2,
        "position_y": 0.80,
        "word_highlight_color": "#FFFFFF",
        "active_color": "#00FF88",
    },
    CaptionStyle.WORD_BY_WORD: {
        "font_size_ratio": 0.07,
        "font_color": "white",
        "bg_color": "black@0.85",
        "stroke_color": "#333333",
        "stroke_width": 3,
        "position_y": 0.78,
        "word_highlight_color": "#FFD700",
    },
    CaptionStyle.NEON: {
        "font_size_ratio": 0.055,
        "font_color": "#00FFFF",
        "bg_color": None,
        "stroke_color": "#FF00FF",
        "stroke_width": 2,
        "position_y": 0.80,
        "word_highlight_color": "#FF00FF",
        "glow": True,
    },
}


class DynamicCaptionEngine:
    def __init__(self):
        self._font_cache: dict[bool, str] = {}

    def _get_font(self, bold: bool = True) -> str:
        if bold not in self._font_cache:
            self._font_cache[bold] = _find_font(bold)
        return self._font_cache[bold]

    def render(
        self,
        video_path: str,
        plan: EditingPlan,
        config: AgentConfig,
        output_path: str,
    ) -> str:
        style = STYLE_CONFIGS.get(config.caption_style, STYLE_CONFIGS[CaptionStyle.DEFAULT])

        if config.caption_style == CaptionStyle.WORD_BY_WORD:
            return self._render_word_by_word(video_path, plan, style, config, output_path)
        elif config.caption_style == CaptionStyle.KARAOKE:
            return self._render_karaoke(video_path, plan, style, config, output_path)
        else:
            return self._render_standard(video_path, plan, style, config, output_path)

    def _render_standard(
        self,
        video_path: str,
        plan: EditingPlan,
        style: dict[str, Any],
        config: AgentConfig,
        output_path: str,
    ) -> str:
        if not plan.segments:
            return video_path

        info = self._get_video_info(video_path)
        w, h = info.get("width", 1080), info.get("height", 1920)
        font_size = int(h * style["font_size_ratio"])
        font_file = self._get_font(bold=True)

        drawtext_parts = []
        for seg in plan.segments:
            text = self._escape_text(seg.text)
            if not text:
                continue

            y_expr = f"h*{style['position_y']}"

            base_opts = (
                f"fontfile='{font_file}'"
                f":fontsize={font_size}"
                f":fontcolor={style['font_color']}"
                f":borderw={style['stroke_width']}"
                f":bordercolor={style['stroke_color']}"
                f":x=(w-text_w)/2"
                f":y={y_expr}"
                f":enable='between(t,{seg.start:.3f},{seg.end:.3f})'"
            )

            if style.get("bg_color"):
                base_opts += f":box=1:boxcolor={style['bg_color']}:boxborderw=10"

            drawtext_parts.append(f"drawtext={base_opts}:text='{text}'")

        if not drawtext_parts:
            return video_path

        filter_chain = ",".join(drawtext_parts)

        highlight_parts = self._build_keyword_highlights(
            plan.keyword_highlights, w, h, font_size, font_file, style
        )
        if highlight_parts:
            filter_chain += "," + ",".join(highlight_parts)

        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", filter_chain,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=600)
            logger.info(f"Rendered {len(plan.segments)} caption segments")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Caption render failed: {e.stderr[:500]}")
            return video_path

    def _render_word_by_word(
        self,
        video_path: str,
        plan: EditingPlan,
        style: dict[str, Any],
        config: AgentConfig,
        output_path: str,
    ) -> str:
        info = self._get_video_info(video_path)
        w, h = info.get("width", 1080), info.get("height", 1920)
        font_size = int(h * style["font_size_ratio"])
        font_file = self._get_font(bold=True)

        drawtext_parts = []
        for seg in plan.segments:
            words = seg.text.split()
            if not words:
                continue

            seg_duration = seg.end - seg.start
            word_dur = seg_duration / len(words)

            for i, word in enumerate(words):
                w_start = seg.start + i * word_dur
                w_end = w_start + word_dur
                escaped = self._escape_text(word)

                is_keyword = any(
                    kw.word.lower() == word.lower().strip(".,!?")
                    for kw in plan.keyword_highlights
                    if kw.start <= w_end and kw.end >= w_start
                )

                color = style["word_highlight_color"] if is_keyword else style["font_color"]
                scale = 1.2 if is_keyword else 1.0
                fs = int(font_size * scale)

                opts = (
                    f"fontfile='{font_file}'"
                    f":fontsize={fs}"
                    f":fontcolor={color}"
                    f":borderw={style['stroke_width']}"
                    f":bordercolor={style['stroke_color']}"
                    f":x=(w-text_w)/2"
                    f":y=h*{style['position_y']}"
                    f":enable='between(t,{w_start:.3f},{w_end:.3f})'"
                )

                if style.get("bg_color") and is_keyword:
                    opts += f":box=1:boxcolor={style['bg_color']}:boxborderw=8"

                drawtext_parts.append(f"drawtext={opts}:text='{escaped}'")

        if not drawtext_parts:
            return video_path

        filter_chain = ",".join(drawtext_parts)
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", filter_chain,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy", output_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=600)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Word-by-word caption failed: {e.stderr[:500]}")
            return video_path

    def _render_karaoke(
        self,
        video_path: str,
        plan: EditingPlan,
        style: dict[str, Any],
        config: AgentConfig,
        output_path: str,
    ) -> str:
        info = self._get_video_info(video_path)
        w, h = info.get("width", 1080), info.get("height", 1920)
        font_size = int(h * style["font_size_ratio"])
        font_file = self._get_font(bold=True)

        drawtext_parts = []
        for seg in plan.segments:
            words = seg.text.split()
            if not words:
                continue

            seg_dur = seg.end - seg.start
            word_dur = seg_dur / len(words)

            inactive_text = self._escape_text(seg.text)
            base_opts = (
                f"fontfile='{font_file}'"
                f":fontsize={font_size}"
                f":fontcolor={style['font_color']}"
                f":borderw={style['stroke_width']}"
                f":bordercolor={style['stroke_color']}"
                f":x=(w-text_w)/2"
                f":y=h*{style['position_y']}"
                f":enable='between(t,{seg.start:.3f},{seg.end:.3f})'"
            )
            if style.get("bg_color"):
                base_opts += f":box=1:boxcolor={style['bg_color']}:boxborderw=10"
            drawtext_parts.append(f"drawtext={base_opts}:text='{inactive_text}'")

            for i, word in enumerate(words):
                w_time = seg.start + i * word_dur
                escaped = self._escape_text(word)
                active_opts = (
                    f"fontfile='{font_file}'"
                    f":fontsize={int(font_size * 1.1)}"
                    f":fontcolor={style.get('active_color', '#00FF88')}"
                    f":borderw={style['stroke_width'] + 1}"
                    f":bordercolor={style['stroke_color']}"
                    f":x=(w-text_w)/2"
                    f":y=h*{style['position_y']}"
                    f":enable='between(t,{w_time:.3f},{w_time + word_dur:.3f})'"
                )
                drawtext_parts.append(f"drawtext={active_opts}:text='{escaped}'")

        filter_chain = ",".join(drawtext_parts)
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", filter_chain,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy", output_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=600)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Karaoke caption failed: {e.stderr[:500]}")
            return video_path

    def _build_keyword_highlights(
        self,
        keywords: list[KeywordHighlight],
        w: int,
        h: int,
        font_size: int,
        font_file: str,
        style: dict[str, Any],
    ) -> list[str]:
        parts = []
        for kw in keywords:
            escaped = self._escape_text(kw.word)
            highlight_fs = int(font_size * kw.scale)
            color = kw.color or style.get("word_highlight_color", "#FFD700")

            opts = (
                f"fontfile='{font_file}'"
                f":fontsize={highlight_fs}"
                f":fontcolor={color}"
                f":borderw={style['stroke_width'] + 1}"
                f":bordercolor={style['stroke_color']}"
                f":x=(w-text_w)/2"
                f":y=h*{style['position_y']}"
                f":enable='between(t,{kw.start:.3f},{kw.end:.3f})'"
                f":box=1:boxcolor=black@0.6:boxborderw=6"
            )
            parts.append(f"drawtext={opts}:text='{escaped}'")

        return parts

    def _escape_text(self, text: str) -> str:
        text = text.replace("\\", "\\\\")
        text = text.replace("'", "'\\''")
        text = text.replace(":", "\\:")
        text = text.replace("%", "%%")
        text = text.replace("[", "\\[")
        text = text.replace("]", "\\]")
        text = text.replace(";", "\\;")
        return text.strip()

    def _get_video_info(self, video_path: str) -> dict[str, int]:
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-select_streams", "v:0",
                video_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    return {
                        "width": stream.get("width", 1080),
                        "height": stream.get("height", 1920),
                    }
        except Exception as e:
            logger.warning(f"Could not get video info: {e}")
        return {"width": 1080, "height": 1920}
