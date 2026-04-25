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

    system_prompt = f"""You are an Expert Viral Content Strategist. Your job is to identify exactly {clip_count} video segments that are EXACTLY {clip_length} seconds long (tolerance: {min_duration}-{max_duration} seconds).

CRITICAL REQUIREMENTS:
1. Each clip MUST be {clip_length} seconds long - this is non-negotiable
2. Clips CANNOT overlap - use completely different time ranges
3. Pick segments with clear, complete narratives
4. Start clips at natural transition points when possible

OUTPUT FORMAT - STRICTLY FOLLOW THIS PATTERN:
```
CLIP 1 - [CATEGORY]:
START: [number] END: [number]
```

Example: If clip 1 starts at 0s and should be {clip_length}s long, output START: 0 END: {clip_length}

TRANSCRIPT:
{transcript_text}"""

    user_prompt = f"""Extract exactly {clip_count} clips of {clip_length} seconds each from this transcript.

CRITICAL: Your START and END values must result in clips that are {clip_length} seconds long!
Example: If START is 0, then END must be {clip_length}

Follow the format exactly:
```
CLIP 1 - HOOK FOCUS:
START: [0] END: [{clip_length}]

CLIP 2 - EMOTIONAL PEAK:
START: [X] END: [X+{clip_length}]

CLIP 3 - VIRAL MOMENT:
START: [Y] END: [Y+{clip_length}]
```

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