import os
from moviepy.editor import VideoFileClip
from utils.file_utils import get_temp_path


def extract_audio(video_path, audio_path=None):
    video = VideoFileClip(video_path)
    if audio_path is None:
        audio_path = get_temp_path(suffix='.mp3')
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    video.close()
    return audio_path


def get_video_info(video_path):
    video = VideoFileClip(video_path)
    info = {
        'duration': video.duration,
        'size': video.size,
        'fps': video.fps if hasattr(video, 'fps') else None
    }
    video.close()
    return info
