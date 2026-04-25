import openai
from moviepy.editor import VideoFileClip
import tempfile
import os

from core.transcription import extract_audio
from core.clip_analyzer import analyze_clips_with_gpt, format_transcript_with_timestamps
from core.clip_generator import parse_clip_timestamps
from core.vision_analysis import analyze_all_clips_visual, rank_clips_with_vision
from utils.file_utils import get_temp_path, cleanup_files


STEP_MESSAGES = [
    "Step 1/6: Extracting audio...",
    "Step 2/6: Transcribing audio...",
    "Step 3/6: GPT analyzing content...",
    "Step 4/6: Extracting clips...",
]


class VideoProcessor:
    def __init__(self, api_key, clip_count=3, clip_length=30, caption_style="default", use_vision=False):
        self.api_key = api_key
        self.clip_count = clip_count
        self.clip_length = clip_length
        self.caption_style = caption_style
        self.use_vision = use_vision
        self.video_path = None
        self.audio_path = None
        self.transcript = None
        self.clips_data = []
        self.final_clips = {}
        self.vision_results = []
        self.ranked_clips = []
    
    def load_video(self, video_bytes):
        self.video_path = get_temp_path(suffix='.mp4')
        with open(self.video_path, 'wb') as f:
            f.write(video_bytes)
        return self.video_path
    
    def extract_audio(self):
        self.audio_path = extract_audio(self.video_path)
        return self.audio_path
    
    def transcribe(self):
        with open(self.audio_path, 'rb') as audio_file:
            self.transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        return self.transcript
    
    def get_transcript_text(self):
        if not self.transcript:
            return ""
        return format_transcript_with_timestamps(self.transcript.segments)
    
    def analyze_clips(self):
        transcript_text = self.get_transcript_text()
        analysis = analyze_clips_with_gpt(
            transcript_text,
            self.clip_count,
            self.clip_length,
            self.api_key
        )
        return analysis
    
    def extract_clips(self, analysis):
        self.clips_data = parse_clip_timestamps(analysis, self.clip_length, self.clip_count)
        return self.clips_data
    
    def analyze_with_vision(self):
        if not self.use_vision or not self.video_path:
            return None
        
        transcript_segments = []
        for clip in self.clips_data:
            segment = self.get_transcript_segment(clip["start"], clip["end"])
            transcript_segments.append(segment)
        
        self.vision_results = analyze_all_clips_visual(
            self.video_path,
            self.clips_data,
            transcript_segments,
            self.api_key
        )
        
        text_scores = [7.0] * len(self.clips_data)
        self.ranked_clips = rank_clips_with_vision(
            self.clips_data,
            self.vision_results,
            text_scores
        )
        
        return self.ranked_clips
    
    def get_transcript_segment(self, start_time, end_time):
        if not self.transcript or not self.transcript.segments:
            return ""
        segment_texts = []
        for seg in self.transcript.segments:
            if seg.start >= start_time - 1 and seg.end <= end_time + 1:
                segment_texts.append(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
        return "\n".join(segment_texts)
    
    def extract_clip_video(self, start_time, end_time):
        video = VideoFileClip(self.video_path)
        clip = video.subclip(start_time, end_time)
        clip_path = get_temp_path(suffix='.mp4')
        clip.write_videofile(clip_path, verbose=False, logger=None)
        clip.close()
        video.close()
        return clip_path
    
    def crop_video(self, clip_path):
        from crop_videos import SmartVerticalCropper
        cropped_path = get_temp_path(suffix='.mp4')
        cropper = SmartVerticalCropper(smoothing_factor=0.1)
        success = cropper.crop_video(clip_path, cropped_path)
        return cropped_path if success else None
    
    def add_captions(self, cropped_path):
        from captions import create_captions_video
        captioned_path = get_temp_path(suffix='.mp4')
        success = create_captions_video(
            cropped_path,
            captioned_path,
            api_key=self.api_key,
            style_preset=self.caption_style
        )
        return captioned_path if success else None
    
    def cleanup(self):
        if self.video_path:
            cleanup_files(self.video_path)
        if self.audio_path:
            cleanup_files(self.audio_path)
        for clip_data in self.final_clips.values():
            if clip_data.get('original') and clip_data['original'] != clip_data.get('final'):
                cleanup_files(clip_data['original'])
