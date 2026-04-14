import streamlit as st
import openai
import os
import traceback
from dotenv import load_dotenv

from ui.sidebar import render_sidebar
from ui.progress import ProgressUI, get_step_message
from ui.results import (
    render_transcription,
    render_analysis,
    render_clip_results,
    render_debug_info,
    render_error
)
from core.processor import VideoProcessor
from core.analysis import CLIP_CATEGORIES

load_dotenv()

st.title("Convert Videos Into Short Clips")

openai.api_key = os.getenv("api")

clip_count, clip_length, caption_style = render_sidebar()

video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

if video_file:
    if st.button("Process Video"):
        progress = ProgressUI()
        progress.init()
        
        processor = VideoProcessor(
            api_key=openai.api_key,
            clip_count=clip_count,
            clip_length=clip_length,
            caption_style=caption_style
        )
        
        try:
            progress.update(0)
            progress.set_status(get_step_message(0))
            processor.load_video(video_file.read())
            
            progress.update(1)
            progress.set_status(get_step_message(1))
            processor.extract_audio()
            processor.transcribe()
            
            transcript_text = processor.get_transcript_text()
            render_transcription(transcript_text)
            
            progress.update(2)
            progress.set_status(get_step_message(2))
            analysis = processor.analyze_clips()
            render_analysis(analysis)
            
            progress.update(3)
            progress.set_status(get_step_message(3))
            clips_data = processor.extract_clips(analysis)
            
            if len(clips_data) == 0:
                st.error("Could not find any valid clip timestamps in the analysis.")
                render_debug_info(analysis)
                progress.clear()
            else:
                st.success(f"Found {len(clips_data)} valid clip(s) (target: {clip_length}s)")
                
                for idx, clip_info in enumerate(clips_data):
                    category = CLIP_CATEGORIES.get(clip_info['number'], f"Clip {idx + 1}")
                    clip_num_display = idx + 1
                    
                    progress.progress_bar = st.progress(0)
                    progress.status_text = st.empty()
                    
                    status_msg = f"Processing {category} ({clip_num_display}/{len(clips_data)})..."
                    progress.status_text.text(status_msg)
                    progress.progress_bar.progress(10)
                    
                    start_time = clip_info['start']
                    end_time = clip_info['end']
                    duration = end_time - start_time
                    
                    clip_path = processor.extract_clip_video(start_time, end_time)
                    
                    progress.update(4, total_steps=6)
                    progress.set_status(f"Step 5/6: Cropping {category} to vertical...")
                    
                    try:
                        cropped_path = processor.crop_video(clip_path)
                        
                        if cropped_path:
                            progress.update(5, total_steps=6)
                            progress.set_status(f"Step 6/6: Adding captions to {category}...")
                            
                            captioned_path = processor.add_captions(cropped_path)
                            
                            if captioned_path:
                                processor.final_clips[clip_num_display] = {
                                    'category': category,
                                    'original': clip_path,
                                    'final': captioned_path,
                                    'start': start_time,
                                    'end': end_time,
                                    'duration': duration
                                }
                                progress.progress_bar.progress(100)
                                progress.status_text.text(f"{category} complete!")
                            else:
                                processor.final_clips[clip_num_display] = {
                                    'category': category,
                                    'original': clip_path,
                                    'final': cropped_path,
                                    'start': start_time,
                                    'end': end_time,
                                    'duration': duration,
                                    'no_captions': True
                                }
                                progress.progress_bar.progress(100)
                                progress.status_text.text(f"{category} complete (no captions)")
                        else:
                            processor.final_clips[clip_num_display] = {
                                'category': category,
                                'original': clip_path,
                                'final': clip_path,
                                'start': start_time,
                                'end': end_time,
                                'duration': duration,
                                'no_crop': True
                            }
                            progress.progress_bar.progress(100)
                            progress.status_text.text(f"{category} complete (no crop)")
                            
                    except Exception as e:
                        st.error(f"Error processing {category}: {e}")
                        processor.final_clips[clip_num_display] = {
                            'category': category,
                            'original': clip_path,
                            'final': clip_path,
                            'start': start_time,
                            'end': end_time,
                            'duration': duration,
                            'error': str(e)
                        }
                        progress.progress_bar.progress(100)
                
                render_clip_results(processor.final_clips, clip_length)
        
        except Exception as e:
            render_error(f"Processing failed: {e}", traceback.format_exc())
        
        finally:
            processor.cleanup()
            progress.clear()
