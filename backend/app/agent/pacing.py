from __future__ import annotations

import logging
from typing import Any

from app.agent.schemas import (
    TranscriptSegment,
    HookDetection,
    PacingBeat,
    EditingPlan,
    AgentConfig,
)

logger = logging.getLogger(__name__)

ENERGY_CURVES = {
    "hook_rising": [1.0, 0.8, 0.7, 0.85, 0.9, 1.0],
    "steady_build": [0.6, 0.65, 0.7, 0.75, 0.85, 0.95],
    "explosive_open": [1.0, 0.9, 0.6, 0.7, 0.8, 1.0],
    "storytelling": [0.5, 0.6, 0.7, 0.9, 1.0, 0.7],
    "tutorial": [0.7, 0.7, 0.7, 0.8, 0.9, 0.8],
}


class PacingOptimizer:
    def __init__(self):
        self._beat_patterns = {
            "hook_rising": [
                ("hook", 0.0, 0.10),
                ("setup", 0.10, 0.25),
                ("buildup", 0.25, 0.50),
                ("transition", 0.50, 0.55),
                ("climax", 0.55, 0.80),
                ("release", 0.80, 1.0),
            ],
            "steady_build": [
                ("setup", 0.0, 0.20),
                ("buildup", 0.20, 0.45),
                ("hook", 0.45, 0.55),
                ("buildup", 0.55, 0.75),
                ("climax", 0.75, 0.90),
                ("release", 0.90, 1.0),
            ],
            "explosive_open": [
                ("hook", 0.0, 0.15),
                ("setup", 0.15, 0.30),
                ("buildup", 0.30, 0.50),
                ("transition", 0.50, 0.55),
                ("climax", 0.55, 0.75),
                ("release", 0.75, 0.85),
                ("hook", 0.85, 1.0),
            ],
        }

    def optimize(
        self,
        plan: EditingPlan,
        config: AgentConfig,
    ) -> EditingPlan:
        curve_name = config.target_energy_curve
        curve = ENERGY_CURVES.get(curve_name, ENERGY_CURVES["hook_rising"])
        pattern = self._beat_patterns.get(curve_name, self._beat_patterns["hook_rising"])

        beats = self._generate_beats(plan.total_duration, pattern, curve, plan.segments)
        plan.beats = beats

        plan = self._adjust_energy_with_hooks(plan)
        plan = self._add_transition_beats(plan)

        return plan

    def _generate_beats(
        self,
        total_duration: float,
        pattern: list[tuple[str, float, float]],
        curve: list[float],
        segments: list[TranscriptSegment],
    ) -> list[PacingBeat]:
        beats = []

        for beat_type, start_pct, end_pct in pattern:
            start = total_duration * start_pct
            end = total_duration * end_pct

            curve_idx = min(
                len(curve) - 1,
                int(start_pct * len(curve)),
            )
            energy = curve[curve_idx]

            if segments:
                overlapping = [
                    s for s in segments
                    if s.start < end and s.end > start
                ]
                if overlapping:
                    avg_seg_energy = sum(s.energy for s in overlapping) / len(overlapping)
                    energy = (energy * 0.4) + (avg_seg_energy * 0.6)

            speed = 1.0
            if beat_type == "hook":
                speed = 0.95
            elif beat_type == "climax":
                speed = 0.9
            elif beat_type == "transition":
                speed = 1.1

            beats.append(PacingBeat(
                start=start,
                end=end,
                energy=min(1.0, max(0.0, energy)),
                beat_type=beat_type,
                suggested_speed=speed,
            ))

        return beats

    def _adjust_energy_with_hooks(self, plan: EditingPlan) -> EditingPlan:
        for hook in plan.hooks:
            for beat in plan.beats:
                if beat.start <= hook.timestamp <= beat.end:
                    boost = hook.strength * 0.3
                    beat.energy = min(1.0, beat.energy + boost)
                    if hook.strength > 0.7:
                        beat.beat_type = "hook"
                break
        return plan

    def _add_transition_beats(self, plan: EditingPlan) -> EditingPlan:
        if len(plan.beats) < 2:
            return plan

        new_beats = []
        for i in range(len(plan.beats)):
            new_beats.append(plan.beats[i])
            if i < len(plan.beats) - 1:
                gap = plan.beats[i + 1].start - plan.beats[i].end
                if gap > 1.0:
                    mid = plan.beats[i].end + gap / 2
                    new_beats.append(PacingBeat(
                        start=plan.beats[i].end,
                        end=plan.beats[i + 1].start,
                        energy=0.4,
                        beat_type="transition",
                        suggested_speed=1.1,
                    ))

        plan.beats = sorted(new_beats, key=lambda b: b.start)
        return plan

    def suggest_speed_adjustments(
        self,
        plan: EditingPlan,
        config: AgentConfig,
    ) -> list[tuple[float, float, float]]:
        adjustments = []
        for beat in plan.beats:
            if beat.beat_type == "hook" and beat.energy > 0.8:
                adjustments.append((beat.start, beat.end, 0.9))
            elif beat.beat_type == "transition":
                adjustments.append((beat.start, beat.end, 1.2))
            elif beat.energy < 0.3:
                adjustments.append((beat.start, beat.end, 1.3))
        return adjustments

    def compute_retention_score(self, plan: EditingPlan) -> float:
        if not plan.beats:
            return 0.0

        score = 0.0
        total_dur = plan.total_duration

        hook_time = sum(
            b.end - b.start for b in plan.beats if b.beat_type == "hook"
        )
        score += min(0.3, (hook_time / total_dur) * 2)

        peak_time = sum(
            b.end - b.start for b in plan.beats if b.beat_type == "climax"
        )
        score += min(0.25, (peak_time / total_dur) * 3)

        energy_variance = self._energy_variance(plan.beats)
        score += min(0.25, energy_variance * 2)

        effect_density = len(plan.effects) / max(1, total_dur / 60)
        score += min(0.2, effect_density * 0.05)

        return min(1.0, score)

    def _energy_variance(self, beats: list[PacingBeat]) -> float:
        if len(beats) < 2:
            return 0.0
        energies = [b.energy for b in beats]
        mean = sum(energies) / len(energies)
        return sum((e - mean) ** 2 for e in energies) / len(energies)
