import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
ASSETS_DIR = BASE_DIR / "assets"


def get_caption_presets():
    presets_dir = ASSETS_DIR / "templates" / "captions"
    if not presets_dir.exists():
        return ["default", "minimal", "highlight"]
    return [f.stem for f in presets_dir.glob("*.json")]


def load_caption_style(style_name):
    presets_dir = ASSETS_DIR / "templates" / "captions"
    preset_path = presets_dir / f"{style_name}.json"
    if preset_path.exists():
        with open(preset_path, 'r') as f:
            return json.load(f)
    return None


def load_prompt(prompt_name):
    prompts_dir = ASSETS_DIR / "prompts"
    prompt_path = prompts_dir / f"{prompt_name}.txt"
    if prompt_path.exists():
        with open(prompt_path, 'r') as f:
            return f.read()
    return None


def load_crop_profile(profile_name):
    profiles_dir = ASSETS_DIR / "config" / "crop_profiles"
    profile_path = profiles_dir / f"{profile_name}.yaml"
    if profile_path.exists():
        import yaml
        with open(profile_path, 'r') as f:
            return yaml.safe_load(f)
    return None


def get_project_root():
    return BASE_DIR
