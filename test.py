import streamlit as st
import openai
from moviepy.editor import VideoFileClip
import tempfile
import os
import re
from dotenv import load_dotenv

load_dotenv()

st.title("Convert Videos Into Short Clips")

# Set your API key here
openai.api_key = os.getenv("api")

video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

if video_file:
    
    if st.button("Process Video"):
        with st.spinner("Processing..."):
            # Save video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                tmp_video.write(video_file.read())
                video_path = tmp_video.name
            
            # Extract audio
            video = VideoFileClip(video_path)
            audio_path = tempfile.mktemp(suffix='.mp3')
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()
            
            # Transcribe with timestamps
            with open(audio_path, 'rb') as audio_file:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            
            st.subheader("Transcription")
            
            # Format transcript with timestamps
            full_text = ""
            for segment in transcript.segments:
                full_text += f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}\n"
            
            st.text_area("", full_text, height=200)
            
            # Analyze with GPT
            # Analyze with GPT
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": "You are an Expert Viral Content Strategist. Your goal is to identify the single most shareable, engaging, and inclusive segment from a video transcript."
                }, {
                    "role": "user",
                    "content": f"""Analyze this transcript and identify the best 30-second clip (maximum 30 seconds) that would make viewers stop scrolling.

                            CRITERIA:
                            1. **The Hook**: Must grab attention in the first 3 seconds.
                            2. **Inclusivity & Appeal**: Look for universal human experiences, diverse perspectives, or content that brings people together. Avoid offensive stereotypes or niche jargon that excludes the general audience.
                            3. **Emotional/Intellectual Payoff**: The clip must contain a clear value add - a laugh, a "wow" moment, a surprising fact, or a deep insight.

                            OUTPUT FORMAT:
                            Provide a brief "REASONING" section explaining why you chose this clip and how it meets the criteria.
                            Then, on a new line, provide the exact timestamps in this format:
                            START:X END:Y

                            TRANSCRIPT:
                            {full_text}"""
                }]
            )
            
            st.subheader("Best Scene Analysis")
            analysis = response.choices[0].message.content
            st.write(analysis)
            
            # Extract timestamps and create clip
            try:
                # Regex to extract timestamps (handles various formats, case-insensitive, bolding, units like 's')
                # Matches: START: 10.5s END: 20.5s, **START**: 10.5, Start: 10.5, etc.
                pattern = r'(?i)start[:\s]*\*?(\d+(?:\.\d+)?).*?end[:\s]*\*?(\d+(?:\.\d+)?)'
                times = re.search(pattern, analysis, re.DOTALL)
                if times:
                    start_time = float(times.group(1))
                    end_time = float(times.group(2))
                    
                    # Create clip
                    video = VideoFileClip(video_path)
                    clip = video.subclip(start_time, end_time)
                    clip_path = tempfile.mktemp(suffix='.mp4')
                    clip.write_videofile(clip_path, verbose=False, logger=None)
                    clip.close()
                    video.close()
                    
                    st.subheader("Best Scene Clip (Original)")
                    st.video(clip_path)
                    
                    
# Apply smart cropping
                    st.write("Applying smart vertical cropping...")
                    try:
                        from crop_videos import SmartVerticalCropper
                        from captions import create_captions_video
                        
                        cropped_path = tempfile.mktemp(suffix='.mp4')
                        # Initialize with lower smoothing for shorter clips
                        cropper = SmartVerticalCropper(smoothing_factor=0.1)
                        crop_success = cropper.crop_video(clip_path, cropped_path)
                        
                        if crop_success:
                            # Add Captions
                            st.write("Generating captions for the vertical clip...")
                            captioned_path = tempfile.mktemp(suffix='.mp4')
                            
                            # Use the API key from st.secrets or environment if available, 
                            # otherwise it will try to load from .env inside the function, 
                            # but we can pass it explicitly since we have it here.
                            captions_success = create_captions_video(cropped_path, captioned_path, api_key=openai.api_key)
                            
                            if captions_success:
                                st.subheader("Final Video (Vertical + Captions)")
                                st.video(captioned_path)
                                
                                # Download buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    with open(clip_path, 'rb') as f:
                                        st.download_button("Download Original", f, "best_scene_original.mp4")
                                with col2:
                                    with open(captioned_path, 'rb') as f:
                                        st.download_button("Download Final Video", f, "best_scene_captioned.mp4")
                                        
                                # Clean up
                                os.unlink(captioned_path)
                            else:
                                st.error("Failed to add captions. Showing vertical video without captions.")
                                st.subheader("Best Scene Clip (Vertical)")
                                st.video(cropped_path)
                                with open(cropped_path, 'rb') as f:
                                    st.download_button("Download Vertical (No Captions)", f, "best_scene_vertical.mp4")

                            # Clean up cropped file
                            os.unlink(cropped_path)
                        else:
                            st.error("Failed to crop video automatically.")
                            with open(clip_path, 'rb') as f:
                                st.download_button("Download Clip", f, "best_scene.mp4")
                                
                    except Exception as e:
                        st.error(f"Error during post-processing: {e}")
                        with open(clip_path, 'rb') as f:
                            st.download_button("Download Clip", f, "best_scene.mp4")
                    
                    os.unlink(clip_path)
                else:
                    st.error("Could not find START/END timestamps in the analysis.")
                    with st.expander("Debug Info - Raw Analysis Output"):
                        st.code(analysis)
            except Exception as e:
                st.error(f"Could not create clip: {e}")
            
            # Cleanup
            os.unlink(video_path)
            os.unlink(audio_path)