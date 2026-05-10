from core.clip_generator import parse_clip_timestamps

test_analysis = """CLIP 1 - HOOK FOCUS:
REASONING: Best opening moment.
START:5 END:30

CLIP 2 - EMOTIONAL PEAK:
REASONING: Impactful content.
START:60 END:85

CLIP 3 - VIRAL MOMENT:
START:120 END:145"""

result = parse_clip_timestamps(test_analysis, 30, 3)
print("Parsed clips:")
for r in result:
    print(f"  Clip {r['number']}: {r['start']}-{r['end']} = {r['duration']}s")

if not result:
    print("No clips parsed!")
    print("Analysis text:", test_analysis)