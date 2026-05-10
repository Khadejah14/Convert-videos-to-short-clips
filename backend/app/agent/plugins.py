from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.agent.schemas import (
    EditDecision,
    EffectType,
    TranscriptSegment,
    KeywordHighlight,
    EditingPlan,
    AgentConfig,
)

logger = logging.getLogger(__name__)


class PluginBase(ABC):
    name: str = "base"
    version: str = "0.1.0"
    description: str = ""

    def on_init(self, config: AgentConfig) -> None:
        pass

    def on_plan_created(self, plan: EditingPlan) -> EditingPlan:
        return plan

    def on_before_render(self, plan: EditingPlan) -> EditingPlan:
        return plan

    def on_after_render(self, output_path: str) -> str:
        return output_path

    def on_cleanup(self) -> None:
        pass


class EffectPlugin(PluginBase):
    @abstractmethod
    def apply(
        self,
        video_path: str,
        decision: EditDecision,
        output_path: str,
    ) -> str:
        ...

    @abstractmethod
    def supported_effects(self) -> list[EffectType]:
        ...


class CaptionPlugin(PluginBase):
    @abstractmethod
    def render_caption(
        self,
        segment: TranscriptSegment,
        keywords: list[KeywordHighlight],
        style: dict[str, Any],
    ) -> Any:
        ...


class PluginManager:
    def __init__(self):
        self._plugins: dict[str, PluginBase] = {}
        self._effect_plugins: list[EffectPlugin] = []
        self._caption_plugins: list[CaptionPlugin] = []

    def register(self, plugin: PluginBase) -> None:
        self._plugins[plugin.name] = plugin
        if isinstance(plugin, EffectPlugin):
            self._effect_plugins.append(plugin)
        if isinstance(plugin, CaptionPlugin):
            self._caption_plugins.append(plugin)
        logger.info(f"Registered plugin: {plugin.name} v{plugin.version}")

    def load_from_path(self, module_path: str) -> None:
        try:
            mod = importlib.import_module(module_path)
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, PluginBase)
                    and attr is not PluginBase
                    and attr is not EffectPlugin
                    and attr is not CaptionPlugin
                ):
                    self.register(attr())
        except Exception as e:
            logger.error(f"Failed to load plugin from {module_path}: {e}")

    def load_from_directory(self, directory: str | Path) -> None:
        d = Path(directory)
        if not d.is_dir():
            return
        for f in d.glob("*.py"):
            if f.name.startswith("_"):
                continue
            module_path = f.stem
            self.load_from_path(f"app.agent.plugins.{module_path}")

    def get_effect_handler(self, effect_type: EffectType) -> EffectPlugin | None:
        for p in self._effect_plugins:
            if effect_type in p.supported_effects():
                return p
        return None

    def get_caption_handler(self) -> CaptionPlugin | None:
        return self._caption_plugins[0] if self._caption_plugins else None

    def init_all(self, config: AgentConfig) -> None:
        for p in self._plugins.values():
            p.on_init(config)

    def run_plan_hooks(self, plan: EditingPlan) -> EditingPlan:
        for p in self._plugins.values():
            plan = p.on_plan_created(plan)
        return plan

    def run_pre_render_hooks(self, plan: EditingPlan) -> EditingPlan:
        for p in self._plugins.values():
            plan = p.on_before_render(plan)
        return plan

    def run_post_render_hooks(self, output_path: str) -> str:
        for p in self._plugins.values():
            output_path = p.on_after_render(output_path)
        return output_path

    def cleanup_all(self) -> None:
        for p in self._plugins.values():
            p.on_cleanup()

    @property
    def loaded_plugins(self) -> list[str]:
        return list(self._plugins.keys())
