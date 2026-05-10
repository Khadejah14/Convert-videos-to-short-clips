from __future__ import annotations

import json
import logging
from typing import Any

import openai

from app.agent.schemas import (
    TranscriptSegment,
    HookDetection,
    KeywordHighlight,
    EditingStyle,
    EditingPlan,
    AgentConfig,
)

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM_PROMPT = """You are an expert video editor and content strategist for short-form video.
Analyze the transcript and return a structured JSON analysis.

You must detect:
1. HOOKS - Opening lines that grab attention, pattern interrupts, cliffhangers, emotional peaks
2. EMOTIONS - Emotional tone of each segment (excitement, curiosity, shock, humor, tension, empathy, anger, surprise)
3. KEYWORDS - Words that should be visually emphasized (power words, emotional words, surprising claims, numbers)
4. ENERGY - A 0.0-1.0 energy score for each segment (pacing, intensity)
5. BEATS - Narrative structure beats (setup, buildup, climax, release, hook, transition)

Return ONLY valid JSON with this structure:
{
  "segments": [
    {
      "start": 0.0,
      "end": 5.0,
      "text": "...",
      "emotion": "excitement",
      "energy": 0.8,
      "is_hook": true,
      "is_emotional_peak": false,
      "keywords": ["incredible", "never"]
    }
  ],
  "hooks": [
    {
      "timestamp": 0.0,
      "duration": 3.0,
      "type": "opening_hook",
      "strength": 0.9,
      "text": "...",
      "reason": "Strong claim that creates curiosity"
    }
  ],
  "keyword_highlights": [
    {
      "word": "incredible",
      "start": 1.2,
      "end": 1.8,
      "color": "#FFD700",
      "scale": 1.3,
      "emphasis": "bold"
    }
  ]
}"""

STYLE_MODIFIERS = {
    EditingStyle.VIRAL: "Focus on maximum engagement. Prioritize shock value, curiosity gaps, and pattern interrupts. Energy should peak early and maintain high throughout.",
    EditingStyle.CINEMATIC: "Focus on emotional storytelling. Use rising action, dramatic pauses, and emotional beats. Energy should build gradually to a climax.",
    EditingStyle.PODCAST: "Conversational tone. Identify key insights, quotable moments, and debate points. Moderate energy with spikes at revelations.",
    EditingStyle.TUTORIAL: "Clear structure. Identify steps, key tips, and common mistakes. Steady energy with emphasis on important instructions.",
    EditingStyle.DOCUMENTARY: "Authoritative tone. Identify facts, statistics, and narrative arcs. Measured pacing with dramatic emphasis on key findings.",
}


class ContentAnalyzer:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def analyze(
        self,
        transcript_segments: list[dict[str, Any]],
        full_text: str,
        config: AgentConfig,
    ) -> tuple[list[TranscriptSegment], list[HookDetection], list[KeywordHighlight]]:
        style_modifier = STYLE_MODIFIERS.get(config.style, STYLE_MODIFIERS[EditingStyle.VIRAL])

        user_prompt = self._build_prompt(transcript_segments, full_text, style_modifier)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=4096,
            )

            data = json.loads(response.choices[0].message.content)
            return self._parse_response(data)

        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return self._fallback_analysis(transcript_segments)

    def _build_prompt(
        self,
        segments: list[dict[str, Any]],
        full_text: str,
        style_modifier: str,
    ) -> str:
        segment_text = "\n".join(
            f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text']}"
            for s in segments
        )

        return f"""Analyze this video transcript for short-form editing.

STYLE GUIDANCE: {style_modifier}

TRANSCRIPT SEGMENTS:
{segment_text}

FULL TEXT:
{full_text}

Detect hooks, emotional peaks, keywords to highlight, and energy levels.
Mark the first 3 seconds as an opening_hook if it's attention-grabbing.
Mark surprising claims, emotional shifts, and pattern interrupts.
Identify power words, numbers, and emotionally charged words for visual emphasis."""

    def _parse_response(
        self, data: dict[str, Any]
    ) -> tuple[list[TranscriptSegment], list[HookDetection], list[KeywordHighlight]]:
        segments = []
        for s in data.get("segments", []):
            segments.append(TranscriptSegment(
                start=s.get("start", 0),
                end=s.get("end", 0),
                text=s.get("text", ""),
                emotion=s.get("emotion"),
                energy=s.get("energy", 0.5),
                is_hook=s.get("is_hook", False),
                is_emotional_peak=s.get("is_emotional_peak", False),
                keywords=s.get("keywords", []),
            ))

        hooks = []
        for h in data.get("hooks", []):
            hooks.append(HookDetection(
                timestamp=h.get("timestamp", 0),
                duration=h.get("duration", 1.0),
                type=h.get("type", "hook"),
                strength=min(1.0, max(0.0, h.get("strength", 0.5))),
                text=h.get("text", ""),
                reason=h.get("reason", ""),
            ))

        highlights = []
        for k in data.get("keyword_highlights", []):
            highlights.append(KeywordHighlight(
                word=k.get("word", ""),
                start=k.get("start", 0),
                end=k.get("end", 0),
                color=k.get("color", "#FFD700"),
                scale=k.get("scale", 1.2),
                emphasis=k.get("emphasis", "bold"),
            ))

        return segments, hooks, highlights

    def _fallback_analysis(
        self, raw_segments: list[dict[str, Any]]
    ) -> tuple[list[TranscriptSegment], list[HookDetection], list[KeywordHighlight]]:
        segments = []
        hooks = []
        highlights = []

        for i, s in enumerate(raw_segments):
            text = s.get("text", "")
            words = text.lower().split()
            energy = min(1.0, len(words) / 20.0)

            is_hook = i < 2 or any(
                w in text.lower()
                for w in ["wait", "stop", "did you know", "secret", "never", "truth"]
            )

            seg = TranscriptSegment(
                start=s.get("start", 0),
                end=s.get("end", 0),
                text=text,
                energy=energy,
                is_hook=is_hook,
                keywords=[w for w in words if len(w) > 6][:3],
            )
            segments.append(seg)

            if is_hook:
                hooks.append(HookDetection(
                    timestamp=seg.start,
                    duration=seg.end - seg.start,
                    type="opening_hook" if i == 0 else "pattern_interrupt",
                    strength=0.6,
                    text=text,
                    reason="Fallback detection: attention word or early position",
                ))

            for kw in seg.keywords:
                highlights.append(KeywordHighlight(
                    word=kw,
                    start=seg.start,
                    end=seg.end,
                ))

        return segments, hooks, highlights

    async def detect_emotion_spikes(
        self, segments: list[TranscriptSegment]
    ) -> list[tuple[float, float]]:
        spikes = []
        for i in range(1, len(segments)):
            if segments[i].energy - segments[i - 1].energy > 0.3:
                spikes.append((segments[i].start, segments[i].end))
        return spikes
