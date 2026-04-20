import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

from ui.sidebar import render_sidebar
from ui.components import ProgressDisplay, STEP_MESSAGES, render_clip_card, render_download_buttons
from ui.main_view import (
    render_upload_tab,
    render_transcription_view,
    render_analysis_view,
    render_error_view,
    render_debug_view
)
from core.video_processor import VideoProcessor
from utils.file_utils import cleanup_files


st.title("Convert Videos Into Short Clips")

openai_api_key = os.getenv("api")
if not openai_api_key:
    st.error("API key not found. Please set 'api' in .env file.")
    st.stop()

import openai
openai.api_key = openai_api_key

clip_count, clip_length, caption_style = render_sidebar()

video_file = render_upload_tab()

if video_file and st.button("Process Video"):
    progress = ProgressDisplay()
    progress.init()
    
    processor = VideoProcessor(
        api_key=openai_api_key,
        clip_count=clip_count,
        clip_length=clip_length,
        caption_style=caption_style
    )
    
    try:
        progress.set_status(STEP_MESSAGES[0])
        progress.update(0)
        
        processor.load_video(video_file.read())
        
        progress.set_status(STEP_MESSAGES[1])
        progress.update(1)
        
        processor.extract_audio()
        
        progress.set_status("Step 2/6: Transcribing audio...")
        processor.transcribe()
        
        transcript_text = processor.get_transcript_text()
        render_transcription_view(transcript_text)
        
        progress.set_status(STEP_MESSAGES[2])
        progress.update(2)
        
        analysis = processor.analyze_clips()
        render_analysis_view(analysis)
        
        progress.set_status(STEP_MESSAGES[3])
        progress.update(3)
        
        clips_data = processor.extract_clips(analysis)
        
        if not clips_data:
            st.error("Could not find any valid clip timestamps in the analysis.")
            render_debug_view(analysis)
            progress.clear()
            st.stop()
        
        st.success(f"Found {len(clips_data)} valid clip(s) (target: {clip_length}s)")
        
        final_clips = {}
        
        for idx, clip_info in enumerate(clips_data):
            clip_num_display = idx + 1
            category = processor.clips_data[idx].get('number', clip_num_display)
            
            progress.set_status(f"Processing clip {clip_num_display}/{len(clips_data)}...")
            progress.update(4)
            
            start_time = clip_info['start']
            end_time = clip_info['end']
            
            clip_path = processor.extract_clip_video(start_time, end_time)
            
            progress.set_status(f"Step 5/6: Cropping clip {clip_num_display} to vertical...")
            
            cropped_path = processor.crop_video(clip_path)
            final_path = cropped_path if cropped_path else clip_path
            no_captions = False
            
            progress.set_status(f"Step 6/6: Adding captions to clip {clip_num_display}...")
            
            captioned_path = processor.add_captions(final_path)
            if captioned_path:
                final_path = captioned_path
            else:
                no_captions = True
            
            final_clips[clip_num_display] = {
                'category': f"Clip {clip_num_display}",
                'original': clip_path,
                'final': final_path,
                'start': start_time,
                'end': end_time,
                'duration': clip_info['duration'],
                'no_captions': no_captions
            }
            
            progress.update(6)
        
        st.divider()
        st.header("Generated Clips")
        
        tabs = st.tabs([f"{clip['category']}" for clip in final_clips.values()])
        
        for tab, (clip_num, clip_data) in zip(tabs, final_clips.items()):
            with tab:
                render_clip_card(clip_num, clip_data, clip_length)
        
        render_download_buttons(final_clips)
        
        if len(clips_data) < clip_count:
            st.warning(f"Only found {len(clips_data)} valid clips (requested {clip_count}).")
    
    except Exception as e:
        render_error_view(f"Processing failed: {e}")
        import traceback
        render_debug_view(traceback.format_exc())
    
    finally:
        progress.clear()
        processor.cleanup()


def render_clip_card(clip_num, clip_data, clip_length):
    from ui.components import render_clip_card as _render
    _render(clip_num, clip_data, clip_length)