import streamlit as st
import openai
from moviepy.editor import VideoFileClip
import tempfile
import os
import re
from dotenv import load_dotenv

load_dotenv()

st.title("Convert Videos Into Short Clips")

openai.api_key = os.getenv("api")

with st.sidebar:
    st.header("Settings")
    clip_count = st.slider(
        "Number of clips to generate",
        min_value=1,
        max_value=3,
        value=3,
        help="Generate 1-3 clips from the video"
    )
    
    st.divider()
    
    caption_style = st.selectbox(
        "Caption Style",
        options=["default", "minimal", "highlight"],
        index=0,
        help="default: Solid black background | minimal: Transparent, subtle | highlight: Bold, word emphasis"
    )

video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

if video_file:
    
    if st.button("Process Video"):
        with st.spinner("Processing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                tmp_video.write(video_file.read())
                video_path = tmp_video.name
            
            video = VideoFileClip(video_path)
            audio_path = tempfile.mktemp(suffix='.mp3')
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()
            
            with open(audio_path, 'rb') as audio_file:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            
            st.subheader("Transcription")
            
            full_text = ""
            for segment in transcript.segments:
                full_text += f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}\n"
            
            st.text_area("", full_text, height=200)
            
            clip_count_placeholder = clip_count
            
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": """You are an Expert Viral Content Strategist. Your goal is to identify the best segments from a video transcript for YouTube Shorts. You must identify exactly 3 different clips based on these categories:

1. **HOOK FOCUS**: The strongest opening grab - first words that stop the scroll
2. **EMOTIONAL PEAK**: Laughter, surprise, awe, or a powerful insight moment  
3. **VIRAL MOMENT**: Most shareable, quotable, or memorable segment

IMPORTANT RULES:
- Each clip must be between 15-30 seconds
- Clips CANNOT overlap (use different timestamps for each)
- Each clip must be complete and make sense on its own
- Prioritize clips with clear beginnings and endings
- Look for universal human experiences that resonate broadly"""
                }, {
                    "role": "user",
                    "content": f"""Analyze this transcript and identify the best {clip_count_placeholder} clips for YouTube Shorts.

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
{full_text}"""
                }]
            )
            
            st.subheader("Multi-Clip Analysis")
            analysis = response.choices[0].message.content
            st.write(analysis)
            
            pattern = r'(?i)clip\s*(\d+).*?start[:\s]*\*?(\d+(?:\.\d+)?).*?end[:\s]*\*?(\d+(?:\.\d+)?)'
            all_matches = re.findall(pattern, analysis, re.DOTALL)
            
            clips_data = []
            for match in all_matches:
                clip_num = int(match[0])
                start_time = float(match[1])
                end_time = float(match[2])
                
                if 15 <= (end_time - start_time) <= 35:
                    clips_data.append({
                        'number': clip_num,
                        'start': start_time,
                        'end': end_time
                    })
            
            clips_data.sort(key=lambda x: x['start'])
            
            deduplicated = []
            for clip in clips_data:
                is_overlap = False
                for existing in deduplicated:
                    overlap_start = max(clip['start'], existing['start'])
                    overlap_end = min(clip['end'], existing['end'])
                    overlap_duration = overlap_end - overlap_start
                    clip_duration = clip['end'] - clip['start']
                    
                    if overlap_duration > 0 and (overlap_duration / clip_duration) > 0.5:
                        is_overlap = True
                        break
                
                if not is_overlap:
                    deduplicated.append(clip)
            
            clips_data = deduplicated[:clip_count]
            
            if len(clips_data) == 0:
                st.error("Could not find any valid clip timestamps in the analysis.")
                with st.expander("Debug Info - Raw Analysis Output"):
                    st.code(analysis)
                os.unlink(video_path)
                os.unlink(audio_path)
            else:
                st.success(f"Found {len(clips_data)} valid clip(s)")
                
                clip_categories = {
                    1: "Hook Focus",
                    2: "Emotional Peak", 
                    3: "Viral Moment"
                }
                
                final_clips = {}
                
                for idx, clip_info in enumerate(clips_data):
                    category = clip_categories.get(clip_info['number'], f"Clip {idx + 1}")
                    clip_num_display = idx + 1
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text(f"Processing {category} ({clip_num_display}/{len(clips_data)})...")
                    progress_bar.progress(10)
                    
                    start_time = clip_info['start']
                    end_time = clip_info['end']
                    duration = end_time - start_time
                    
                    video = VideoFileClip(video_path)
                    clip = video.subclip(start_time, end_time)
                    clip_path = tempfile.mktemp(suffix='.mp4')
                    clip.write_videofile(clip_path, verbose=False, logger=None)
                    clip.close()
                    video.close()
                    
                    progress_bar.progress(30)
                    status_text.text(f"Cropping {category} to vertical...")
                    
                    try:
                        from crop_videos import SmartVerticalCropper
                        
                        cropped_path = tempfile.mktemp(suffix='.mp4')
                        cropper = SmartVerticalCropper(smoothing_factor=0.1)
                        crop_success = cropper.crop_video(clip_path, cropped_path)
                        
                        if crop_success:
                            progress_bar.progress(60)
                            status_text.text(f"Adding captions to {category}...")
                            
                            from captions import create_captions_video
                            
                            captioned_path = tempfile.mktemp(suffix='.mp4')
                            captions_success = create_captions_video(
                                cropped_path, 
                                captioned_path, 
                                api_key=openai.api_key, 
                                style_preset=caption_style
                            )
                            
                            if captions_success:
                                final_clips[clip_num_display] = {
                                    'category': category,
                                    'original': clip_path,
                                    'final': captioned_path,
                                    'start': start_time,
                                    'end': end_time,
                                    'duration': duration
                                }
                                progress_bar.progress(100)
                                status_text.text(f"{category} complete!")
                            else:
                                final_clips[clip_num_display] = {
                                    'category': category,
                                    'original': clip_path,
                                    'final': cropped_path,
                                    'start': start_time,
                                    'end': end_time,
                                    'duration': duration,
                                    'no_captions': True
                                }
                                progress_bar.progress(100)
                                status_text.text(f"{category} complete (no captions)")
                        else:
                            final_clips[clip_num_display] = {
                                'category': category,
                                'original': clip_path,
                                'final': clip_path,
                                'start': start_time,
                                'end': end_time,
                                'duration': duration,
                                'no_crop': True
                            }
                            progress_bar.progress(100)
                            status_text.text(f"{category} complete (no crop)")
                            
                    except Exception as e:
                        st.error(f"Error processing {category}: {e}")
                        final_clips[clip_num_display] = {
                            'category': category,
                            'original': clip_path,
                            'final': clip_path,
                            'start': start_time,
                            'end': end_time,
                            'duration': duration,
                            'error': str(e)
                        }
                        progress_bar.progress(100)
                    
                    progress_bar.empty()
                    status_text.empty()
                
                st.divider()
                st.header("Generated Clips")
                
                with st.expander("View All Clips", expanded=True):
                    tabs = st.tabs([f"{clip['category']}" for clip in final_clips.values()])
                    
                    for tab, (clip_num, clip_data) in zip(tabs, final_clips.items()):
                        with tab:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Start", f"{clip_data['start']:.1f}s")
                            with col2:
                                st.metric("End", f"{clip_data['end']:.1f}s")
                            with col3:
                                st.metric("Duration", f"{clip_data['duration']:.1f}s")
                            
                            st.subheader("Final Video")
                            st.video(clip_data['final'])
                            
                            col_download1, col_download2 = st.columns(2)
                            with col_download1:
                                with open(clip_data['original'], 'rb') as f:
                                    st.download_button(
                                        f"Original ({clip_data['category']})",
                                        f,
                                        f"clip_{clip_num}_{clip_data['category'].lower().replace(' ', '_')}_original.mp4"
                                    )
                            with col_download2:
                                suffix = "_captioned" if not clip_data.get('no_captions') else "_vertical"
                                with open(clip_data['final'], 'rb') as f:
                                    st.download_button(
                                        f"Final ({clip_data['category']})",
                                        f,
                                        f"clip_{clip_num}_{clip_data['category'].lower().replace(' ', '_')}{suffix}.mp4"
                                    )
                
                if len(final_clips) > 1:
                    st.divider()
                    st.subheader("Quick Download All")
                    
                    all_cols = st.columns(min(len(final_clips), 3))
                    for idx, (clip_num, clip_data) in enumerate(final_clips.items()):
                        with all_cols[idx]:
                            with open(clip_data['final'], 'rb') as f:
                                st.download_button(
                                    f"Download {clip_data['category']}",
                                    f,
                                    f"clip_{clip_num}_{clip_data['category'].lower().replace(' ', '_')}_final.mp4",
                                    use_container_width=True
                                )
                
                if len(final_clips) == 3:
                    zip_buffer = None
                    
                    if len(final_clips) == 3:
                        import io
                        import zipfile
                        
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for clip_num, clip_data in final_clips.items():
                                with open(clip_data['final'], 'rb') as f:
                                    filename = f"clip_{clip_num}_{clip_data['category'].lower().replace(' ', '_')}_final.mp4"
                                    zip_file.writestr(filename, f.read())
                        
                        zip_buffer.seek(0)
                        
                        st.download_button(
                            "Download All as ZIP",
                            zip_buffer.getvalue(),
                            "all_clips.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                
                for clip_data in final_clips.values():
                    if clip_data.get('original') and clip_data['original'] != clip_data['final']:
                        try:
                            os.unlink(clip_data['original'])
                        except:
                            pass
                    if not clip_data.get('no_crop') and not clip_data.get('error'):
                        try:
                            if clip_data.get('final'):
                                pass
                        except:
                            pass
                
                if len(clips_data) < clip_count:
                    st.warning(f"Only found {len(clips_data)} valid clips (requested {clip_count}). Some segments may have been too short, too long, or overlapping.")
            
            os.unlink(video_path)
            os.unlink(audio_path)

print("hello from someone u don't know lol")
