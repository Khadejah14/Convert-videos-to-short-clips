import re
import os
from moviepy.editor import VideoFileClip
from utils.file_utils import get_temp_path, cleanup_files


CLIP_CATEGORIES = {
    1: "Hook Focus",
    2: "Emotional Peak",
    3: "Viral Moment"
}


CLIP_CATEGORIES = {
    1: "Hook Focus",
    2: "Emotional Peak",
    3: "Viral Moment"
}


def parse_clip_timestamps(analysis_text, clip_length, max_clips=3):
    min_clip = clip_length - 5
    max_clip = clip_length
    
    all_timestamps = []
    
    pattern1 = re.compile(r'(?i)start[:\s]*(\d+(?:\.\d+)?)\s*(?:s|sec)?', re.DOTALL)
    pattern2 = re.compile(r'(?i)end[:\s]*(\d+(?:\.\d+)?)\s*(?:s|sec)?', re.DOTALL)
    
    blocks = re.split(r'(?i)(?:clip|segment)\s*\d+', analysis_text)
    
    clip_num = 1
    for block in blocks:
        if not block.strip():
            continue
        
        starts = pattern1.findall(block)
        ends = pattern2.findall(block)
        
        if starts and ends:
            for start, end in zip(starts[:3], ends[:3]):
                s = float(start)
                e = float(end)
                duration = e - s
                
                if duration < 3:
                    e = s + clip_length
                    duration = clip_length
                
                if min_clip <= duration <= max_clip:
                    all_timestamps.append({
                        'number': clip_num,
                        'start': s,
                        'end': e,
                        'duration': duration
                    })
                    clip_num += 1
    
    if not all_timestamps:
        for block in blocks:
            if not block.strip():
                continue
            
            starts = pattern1.findall(block)
            ends = pattern2.findall(block)
            
            for start, end in zip(starts[:3], ends[:3]):
                s = float(start)
                e = float(end)
                duration = e - s
                
                if duration < 3:
                    e = s + clip_length
                    duration = clip_length
                
                if duration >= 3:
                    all_timestamps.append({
                        'number': clip_num,
                        'start': s,
                        'end': e,
                        'duration': duration
                    })
                    clip_num += 1
    
    all_timestamps.sort(key=lambda x: x['start'])
    
    deduplicated = _remove_overlapping_clips(all_timestamps)
    
    return deduplicated[:max_clips]


def _remove_overlapping_clips(clips):
    result = []
    for clip in clips:
        is_overlap = False
        for existing in result:
            overlap_start = max(clip['start'], existing['start'])
            overlap_end = min(clip['end'], existing['end'])
            overlap_duration = overlap_end - overlap_start
            
            if overlap_duration > 0 and (overlap_duration / clip['duration']) > 0.5:
                is_overlap = True
                break
        
        if not is_overlap:
            result.append(clip)
    
    return result


def extract_clip_from_video(video_path, start_time, end_time):
    video = VideoFileClip(video_path)
    clip = video.subclip(start_time, end_time)
    clip_path = get_temp_path(suffix='.mp4')
    clip.write_videofile(clip_path, verbose=False, logger=None)
    clip.close()
    video.close()
    return clip_path


def process_clip_to_final(clip_path, api_key, caption_style):
    cropped_path = _crop_to_vertical(clip_path)
    
    if cropped_path:
        captioned_path = _add_captions(cropped_path, api_key, caption_style)
        return cropped_path if not captioned_path else captioned_path
    
    return clip_path


def _crop_to_vertical(clip_path):
    try:
        from crop_videos import SmartVerticalCropper
        cropped_path = get_temp_path(suffix='.mp4')
        cropper = SmartVerticalCropper(smoothing_factor=0.1)
        success = cropper.crop_video(clip_path, cropped_path)
        return cropped_path if success else None
    except Exception:
        return None


def _add_captions(video_path, api_key, caption_style):
    try:
        from captions import create_captions_video
        captioned_path = get_temp_path(suffix='.mp4')
        success = create_captions_video(
            video_path,
            captioned_path,
            api_key=api_key,
            style_preset=caption_style
        )
        return captioned_path if success else None
    except Exception:
        return None


def get_category_name(clip_number):
    return CLIP_CATEGORIES.get(clip_number, f"Clip {clip_number}")


def build_clip_data(clip_num, clip_info, original_path, final_path, no_captions=False):
    return {
        'category': get_category_name(clip_num),
        'original': original_path,
        'final': final_path,
        'start': clip_info['start'],
        'end': clip_info['end'],
        'duration': clip_info['duration'],
        'no_captions': no_captions
    }


def cleanup_intermediate_files(clip_path, cropped_path):
    cleanup_files(clip_path, cropped_path)