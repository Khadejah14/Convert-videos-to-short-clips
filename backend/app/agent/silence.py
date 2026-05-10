from __future__ import annotations

import logging
import subprocess
import json
import os
import tempfile
from pathlib import Path

from app.agent.schemas import SilenceSegment, AgentConfig

logger = logging.getLogger(__name__)


class SilenceDetector:
    def __init__(self):
        self._verify_ffmpeg()

    def _verify_ffmpeg(self):
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError("ffmpeg is required for silence detection")

    def detect(
        self,
        audio_path: str,
        config: AgentConfig,
    ) -> list[SilenceSegment]:
        threshold = config.silence_threshold_db
        min_duration = config.min_silence_duration

        cmd = [
            "ffmpeg", "-i", audio_path,
            "-af", f"silencedetect=noise={threshold}dB:d={min_duration}",
            "-f", "null", "-"
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
            return self._parse_silence_output(result.stderr, min_duration)
        except subprocess.TimeoutExpired:
            logger.warning("Silence detection timed out")
            return []

    def _parse_silence_output(
        self, stderr: str, min_duration: float
    ) -> list[SilenceSegment]:
        segments = []
        import re

        starts = re.findall(r"silence_start: ([\d.]+)", stderr)
        ends = re.findall(r"silence_end: ([\d.]+)", stderr)

        for start_str, end_str in zip(starts, ends):
            start = float(start_str)
            end = float(end_str)
            duration = end - start
            if duration >= min_duration:
                segments.append(SilenceSegment(
                    start=start,
                    end=end,
                    duration=duration,
                    keep=False,
                ))

        return segments

    def detect_from_video(self, video_path: str, config: AgentConfig) -> list[SilenceSegment]:
        audio_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_tmp.close()

        try:
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                audio_tmp.name,
            ]
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
            return self.detect(audio_tmp.name, config)
        finally:
            try:
                os.unlink(audio_tmp.name)
            except OSError:
                pass

    def remove_silences(
        self,
        video_path: str,
        silences: list[SilenceSegment],
        padding_ms: int,
        output_path: str,
    ) -> str:
        if not silences:
            return video_path

        padding_s = padding_ms / 1000.0
        keep_segments = self._compute_keep_segments(video_path, silences, padding_s)

        if not keep_segments:
            return video_path

        filter_parts = []
        for i, (start, end) in enumerate(keep_segments):
            filter_parts.append(
                f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}];"
                f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]"
            )

        v_stream = "".join(f"[v{i}]" for i in range(len(keep_segments)))
        a_stream = "".join(f"[a{i}]" for i in range(len(keep_segments)))
        filter_parts.append(
            f"{v_stream}concat=n={len(keep_segments)}:v=1:a=0[outv];"
            f"{a_stream}concat=n={len(keep_segments)}:v=0:a=1[outa]"
        )

        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=600)
            logger.info(f"Removed {len(silences)} silences, output: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Silence removal failed: {e.stderr[:500]}")
            return video_path

    def _compute_keep_segments(
        self,
        video_path: str,
        silences: list[SilenceSegment],
        padding_s: float,
    ) -> list[tuple[float, float]]:
        duration = self._get_duration(video_path)
        if duration <= 0:
            return []

        keep = []
        cursor = 0.0

        for silence in sorted(silences, key=lambda s: s.start):
            seg_start = max(cursor, silence.start - padding_s)
            seg_end = min(duration, silence.end + padding_s)

            if seg_start > cursor:
                keep.append((cursor, seg_start))
            cursor = seg_end

        if cursor < duration:
            keep.append((cursor, duration))

        return [(s, e) for s, e in keep if (e - s) > 0.1]

    def _get_duration(self, video_path: str) -> float:
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
