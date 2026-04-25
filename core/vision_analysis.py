import cv2
import openai
import base64
from io import BytesIO
from typing import List, Dict, Tuple


def extract_keyframes(video_path: str, start_time: float, end_time: float, num_frames: int = 3) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    duration = end_time - start_time
    
    frames = []
    for i in range(num_frames):
        timestamp = start_time + (duration * i / (num_frames - 1)) if num_frames > 1 else start_time + duration / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    cap.release()
    return frames


def encode_frame_to_base64(frame) -> str:
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


def analyze_visual_hooks(
    video_path: str,
    clip_start: float,
    clip_end: float,
    transcript_segment: str,
    api_key: str
) -> Dict:
    frames = extract_keyframes(video_path, clip_start, clip_end, num_frames=3)
    
    if not frames:
        return {"virality_score": 0, "visual_hook": "No frames extracted", "engagement_factors": []}
    
    system_prompt = """You are an Expert Visual Content Analyst specializing in viral video hooks.
Analyze the provided video frames and transcript segment to evaluate the visual hook potential.
Consider:
- Visual contrast and clarity
- Subject positioning and framing
- Expression and body language
- Background elements that enhance or distract
- Text/graphics visibility
- Eye-catching visual elements"""
    
    user_content = []
    for i, frame in enumerate(frames):
        positions = ["start", "middle", "end"]
        user_content.append({
            "type": "text",
            "text": f"Frame {i+1} ({positions[i]} of clip)"
        })
        base64_frame = encode_frame_to_base64(frame)
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_frame}"
            }
        })
    
    user_content.append({
        "type": "text",
        "text": f"""Transcript segment for context:
{transcript_segment}

Analyze the visual hook potential and respond EXACTLY in this format:
```
VISUAL_HOOK: [1-2 sentence description of the strongest visual hook element]
VIRALITY_SCORE: [0-10 score for visual engagement potential]
ENGAGEMENT_FACTORS: [List of specific visual elements that drive engagement]
```"""
    })
    
    openai.api_key = api_key
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        max_tokens=300
    )
    
    result_text = response.choices[0].message.content
    return parse_vision_response(result_text)


def parse_vision_response(response_text: str) -> Dict:
    virality_score = 5
    visual_hook = ""
    engagement_factors = []
    
    for line in response_text.split('\n'):
        line = line.strip()
        if line.startswith('VISUAL_HOOK:'):
            visual_hook = line.replace('VISUAL_HOOK:', '').strip()
        elif line.startswith('VIRALITY_SCORE:'):
            try:
                score_str = line.replace('VIRALITY_SCORE:', '').strip()
                virality_score = int(''.join(filter(str.isdigit, score_str)))
            except:
                pass
        elif line.startswith('ENGAGEMENT_FACTORS:'):
            continue
        elif line.startswith('-') or line.startswith('*'):
            engagement_factors.append(line.lstrip('-* ').strip())
    
    return {
        "virality_score": virality_score,
        "visual_hook": visual_hook,
        "engagement_factors": engagement_factors
    }


def calculate_combined_score(text_score: float, visual_score: float, text_weight: float = 0.6, visual_weight: float = 0.4) -> float:
    return (text_score * text_weight) + (visual_score * visual_weight)


def rank_clips_with_vision(
    clips_data: List[Dict],
    vision_results: List[Dict],
    text_scores: List[float]
) -> List[Dict]:
    ranked_clips = []
    
    for i, clip in enumerate(clips_data):
        visual_score = vision_results[i].get("virality_score", 5) if i < len(vision_results) else 5
        text_score = text_scores[i] if i < len(text_scores) else 7.0
        
        combined_score = calculate_combined_score(text_score, visual_score)
        
        ranked_clips.append({
            **clip,
            "text_score": text_score,
            "visual_score": visual_score,
            "combined_score": combined_score,
            "visual_hook_description": vision_results[i].get("visual_hook", "") if i < len(vision_results) else ""
        })
    
    ranked_clips.sort(key=lambda x: x["combined_score"], reverse=True)
    
    for rank, clip in enumerate(ranked_clips):
        clip["rank"] = rank + 1
    
    return ranked_clips


def analyze_all_clips_visual(
    video_path: str,
    clips_data: List[Dict],
    transcript_segments: List[str],
    api_key: str
) -> List[Dict]:
    results = []
    
    for clip in clips_data:
        result = analyze_visual_hooks(
            video_path,
            clip["start"],
            clip["end"],
            transcript_segments[clips_data.index(clip)] if clips_data.index(clip) < len(transcript_segments) else "",
            api_key
        )
        results.append(result)
    
    return results