from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from app.agent.schemas import (
    EditingStyle,
    CaptionStyle,
    AgentConfig,
    EffectType,
)

logger = logging.getLogger(__name__)

BUILTIN_STYLES: dict[str, dict[str, Any]] = {
    "viral": {
        "name": "Viral",
        "description": "Maximum engagement, fast pacing, strong hooks",
        "caption_style": "highlight",
        "zoom_intensity": 0.7,
        "punch_in_scale": 1.2,
        "enable_silence_removal": True,
        "silence_threshold_db": -30,
        "max_effects_per_minute": 8,
        "target_energy_curve": "explosive_open",
        "preferred_effects": ["punch_in", "zoom_in", "flash"],
        "keywords_emphasis": "bold",
        "keyword_colors": {
            "power": "#FF4444",
            "number": "#00FF88",
            "name": "#FFD700",
            "emotion": "#FF00FF",
        },
    },
    "cinematic": {
        "name": "Cinematic",
        "description": "Smooth, emotional, story-driven pacing",
        "caption_style": "minimal",
        "zoom_intensity": 0.3,
        "punch_in_scale": 1.08,
        "enable_silence_removal": True,
        "silence_threshold_db": -40,
        "max_effects_per_minute": 3,
        "target_energy_curve": "storytelling",
        "preferred_effects": ["zoom_in", "cross_dissolve", "blur_transition"],
        "keywords_emphasis": "italic",
        "keyword_colors": {
            "default": "#E0E0E0",
        },
    },
    "podcast": {
        "name": "Podcast",
        "description": "Conversational, insight-focused, moderate pacing",
        "caption_style": "karaoke",
        "zoom_intensity": 0.4,
        "punch_in_scale": 1.1,
        "enable_silence_removal": True,
        "silence_threshold_db": -35,
        "max_effects_per_minute": 4,
        "target_energy_curve": "steady_build",
        "preferred_effects": ["punch_in", "zoom_in"],
        "keywords_emphasis": "bold",
        "keyword_colors": {
            "insight": "#00BFFF",
            "quote": "#FFD700",
        },
    },
    "tutorial": {
        "name": "Tutorial",
        "description": "Clear, step-by-step, emphasis on key instructions",
        "caption_style": "word_by_word",
        "zoom_intensity": 0.5,
        "punch_in_scale": 1.15,
        "enable_silence_removal": True,
        "silence_threshold_db": -32,
        "max_effects_per_minute": 5,
        "target_energy_curve": "tutorial",
        "preferred_effects": ["zoom_in", "punch_in"],
        "keywords_emphasis": "bold",
        "keyword_colors": {
            "step": "#00FF88",
            "warning": "#FF4444",
            "tip": "#FFD700",
        },
    },
    "documentary": {
        "name": "Documentary",
        "description": "Authoritative, measured, fact-driven",
        "caption_style": "default",
        "zoom_intensity": 0.3,
        "punch_in_scale": 1.05,
        "enable_silence_removal": False,
        "silence_threshold_db": -45,
        "max_effects_per_minute": 2,
        "target_energy_curve": "steady_build",
        "preferred_effects": ["zoom_in", "cross_dissolve"],
        "keywords_emphasis": "bold",
        "keyword_colors": {
            "fact": "#00BFFF",
            "stat": "#00FF88",
        },
    },
}


class StyleManager:
    def __init__(self, custom_styles_dir: str | Path | None = None):
        self._styles: dict[str, dict[str, Any]] = dict(BUILTIN_STYLES)
        self._custom_dir = Path(custom_styles_dir) if custom_styles_dir else None
        if self._custom_dir and self._custom_dir.is_dir():
            self._load_custom_styles()

    def _load_custom_styles(self) -> None:
        for f in self._custom_dir.glob("*.yaml"):
            try:
                with open(f) as fh:
                    data = yaml.safe_load(fh)
                if data and "name" in data:
                    key = f.stem.lower().replace(" ", "_")
                    self._styles[key] = data
                    logger.info(f"Loaded custom style: {key}")
            except Exception as e:
                logger.warning(f"Failed to load style {f}: {e}")

        for f in self._custom_dir.glob("*.yml"):
            try:
                with open(f) as fh:
                    data = yaml.safe_load(fh)
                if data and "name" in data:
                    key = f.stem.lower().replace(" ", "_")
                    self._styles[key] = data
                    logger.info(f"Loaded custom style: {key}")
            except Exception as e:
                logger.warning(f"Failed to load style {f}: {e}")

    def get_style(self, name: str) -> dict[str, Any] | None:
        return self._styles.get(name.lower())

    def get_config_for_style(
        self,
        style: EditingStyle,
        base_config: AgentConfig,
    ) -> AgentConfig:
        style_data = self._styles.get(style.value, {})
        if not style_data:
            return base_config

        overrides: dict[str, Any] = {}

        if "caption_style" in style_data:
            try:
                overrides["caption_style"] = CaptionStyle(style_data["caption_style"])
            except ValueError:
                pass

        for field in [
            "zoom_intensity", "punch_in_scale", "enable_silence_removal",
            "silence_threshold_db", "max_effects_per_minute",
            "target_energy_curve",
        ]:
            if field in style_data:
                overrides[field] = style_data[field]

        return base_config.model_copy(update=overrides)

    def list_styles(self) -> list[dict[str, str]]:
        result = []
        for key, data in self._styles.items():
            result.append({
                "id": key,
                "name": data.get("name", key),
                "description": data.get("description", ""),
            })
        return result

    def save_custom_style(self, name: str, config: dict[str, Any]) -> None:
        if not self._custom_dir:
            return
        self._custom_dir.mkdir(parents=True, exist_ok=True)
        path = self._custom_dir / f"{name}.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        self._styles[name.lower()] = config
