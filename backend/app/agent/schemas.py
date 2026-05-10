import uuid
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class EditingStyle(str, Enum):
    VIRAL = "viral"
    CINEMATIC = "cinematic"
    PODCAST = "podcast"
    TUTORIAL = "tutorial"
    DOCUMENTARY = "documentary"
    CUSTOM = "custom"


class EffectType(str, Enum):
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    PUNCH_IN = "punch_in"
    SHAKE = "shake"
    FLASH = "flash"
    GLITCH = "glitch"
    BLUR_TRANSITION = "blur_transition"
    CROSS_DISSOLVE = "cross_dissolve"


class CaptionStyle(str, Enum):
    DEFAULT = "default"
    MINIMAL = "minimal"
    HIGHLIGHT = "highlight"
    KARAOKE = "karaoke"
    WORD_BY_WORD = "word_by_word"
    NEON = "neon"


class EditDecision(BaseModel):
    timestamp: float
    duration: float
    effect: EffectType
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""


class KeywordHighlight(BaseModel):
    word: str
    start: float
    end: float
    color: str = "#FFD700"
    scale: float = 1.2
    emphasis: str = "bold"


class SilenceSegment(BaseModel):
    start: float
    end: float
    duration: float
    keep: bool = False


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    words: list[dict] = Field(default_factory=list)
    emotion: str | None = None
    energy: float = 0.5
    is_hook: bool = False
    is_emotional_peak: bool = False
    keywords: list[str] = Field(default_factory=list)


class HookDetection(BaseModel):
    timestamp: float
    duration: float
    type: str  # "opening_hook", "pattern_interrupt", "cliffhanger", "emotional_peak"
    strength: float = Field(ge=0.0, le=1.0)
    text: str
    reason: str


class PacingBeat(BaseModel):
    start: float
    end: float
    energy: float = Field(ge=0.0, le=1.0)
    beat_type: str  # "setup", "buildup", "climax", "release", "hook", "transition"
    suggested_speed: float = 1.0


class EditingPlan(BaseModel):
    style: EditingStyle
    total_duration: float
    beats: list[PacingBeat] = Field(default_factory=list)
    hooks: list[HookDetection] = Field(default_factory=list)
    effects: list[EditDecision] = Field(default_factory=list)
    silence_segments: list[SilenceSegment] = Field(default_factory=list)
    keyword_highlights: list[KeywordHighlight] = Field(default_factory=list)
    caption_style: CaptionStyle = CaptionStyle.DEFAULT
    segments: list[TranscriptSegment] = Field(default_factory=list)


class AgentConfig(BaseModel):
    style: EditingStyle = EditingStyle.VIRAL
    caption_style: CaptionStyle = CaptionStyle.HIGHLIGHT
    enable_silence_removal: bool = True
    silence_threshold_db: float = -35.0
    min_silence_duration: float = 0.3
    silence_padding_ms: int = 100
    enable_zoom_effects: bool = True
    zoom_intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    enable_punch_in: bool = True
    punch_in_scale: float = Field(default=1.15, ge=1.0, le=1.5)
    enable_keyword_highlight: bool = True
    enable_pacing_optimization: bool = True
    target_energy_curve: str = "hook_rising"
    max_effects_per_minute: int = 6
    custom_style_path: str | None = None
    plugins: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    job_id: uuid.UUID
    output_path: str
    style: EditingStyle
    plan: EditingPlan
    effects_applied: int
    silences_removed: int
    duration_before: float
    duration_after: float
    processing_time: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
