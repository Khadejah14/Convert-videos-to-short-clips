import re


CLIP_PATTERN = r'(?i)clip\s*(\d+).*?start[:\s]*\*?(\d+(?:\.\d+)?).*?end[:\s]*\*?(\d+(?:\.\d+)?)'


def parse_clips_from_analysis(analysis, clip_length, max_clips=3):
    matches = re.findall(CLIP_PATTERN, analysis, re.DOTALL)
    
    clips_data = []
    min_clip = clip_length - 5
    max_clip = clip_length + 5
    
    for match in matches:
        clip_num = int(match[0])
        start_time = float(match[1])
        end_time = float(match[2])
        clip_duration = end_time - start_time
        
        if min_clip <= clip_duration <= max_clip:
            clips_data.append({
                'number': clip_num,
                'start': start_time,
                'end': end_time,
                'duration': clip_duration
            })
    
    clips_data.sort(key=lambda x: x['start'])
    
    deduplicated = _deduplicate_clips(clips_data)
    
    return deduplicated[:max_clips]


def _deduplicate_clips(clips):
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
