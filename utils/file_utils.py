import tempfile
import os


def create_temp_file(suffix, delete=False):
    return tempfile.NamedTemporaryFile(delete=delete, suffix=suffix)


def get_temp_path(suffix):
    return tempfile.mktemp(suffix=suffix)


def cleanup_files(*file_paths):
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except:
                pass


def format_timestamp(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def format_duration(seconds):
    return f"{seconds:.1f}s"


def generate_clip_filename(clip_num, category, suffix=""):
    safe_category = category.lower().replace(' ', '_')
    return f"clip_{clip_num}_{safe_category}{suffix}.mp4"
