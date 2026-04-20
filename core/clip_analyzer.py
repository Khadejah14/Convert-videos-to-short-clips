import openai


CLIP_CATEGORIES = {
    1: "Hook Focus",
    2: "Emotional Peak",
    3: "Viral Moment"
}


def analyze_clips_with_gpt(transcript_text, clip_count, clip_length, api_key=None):
    if api_key:
        openai.api_key = api_key

    min_duration = clip_length - 5
    max_duration = clip_length

    system_prompt = f"""You are an Expert Viral Content Strategist. Your goal is to identify the best segments from a video transcript for YouTube Shorts. You must identify exactly {clip_count} different clips based on these categories:

1. **HOOK FOCUS**: The strongest opening grab - first words that stop the scroll
2. **EMOTIONAL PEAK**: Laughter, surprise, awe, or a powerful insight moment  
3. **VIRAL MOMENT**: Most shareable, quotable, or memorable segment

IMPORTANT RULES:
- Each clip must be between {min_duration}-{max_duration} seconds
- Clips CANNOT overlap (use different timestamps for each)
- Each clip must be complete and make sense on its own
- Prioritize clips with clear beginnings and endings
- Look for universal human experiences that resonate broadly"""

    user_prompt = f"""Analyze this transcript and identify the best {clip_count} clips for YouTube Shorts.

OUTPUT FORMAT - EXACTLY FOLLOW THIS PATTERN:
```
CLIP 1 - HOOK FOCUS:
REASONING: [Brief explanation why this is the best hook - what makes it grab attention immediately]
START:X END:Y

CLIP 2 - EMOTIONAL PEAK:
REASONING: [Brief explanation of the emotional impact - what makes viewers feel something]
START:X END:Y

CLIP 3 - VIRAL MOMENT:
REASONING: [Brief explanation of why this is shareable - what makes people want to send it to friends]
START:X END:Y
```

If fewer than 3 good clips exist, still follow the format but note in reasoning if a clip is suboptimal.

TRANSCRIPT:
{transcript_text}"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content


def format_transcript_with_timestamps(segments):
    text = ""
    for segment in segments:
        text += f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}\n"
    return text