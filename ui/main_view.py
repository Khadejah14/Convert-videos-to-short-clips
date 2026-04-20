import streamlit as st
from ui.components import render_clip_card, render_progress_step, render_download_buttons


def render_upload_tab():
    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    return video_file


def render_processing_tab(progress_ui, step, step_message):
    render_progress_step(progress_ui, step, step_message)


def render_results_tab(final_clips, clip_length):
    if not final_clips:
        st.info("No clips generated yet.")
        return
    
    st.header("Generated Clips")
    
    tabs = st.tabs([f"{clip['category']}" for clip in final_clips.values()])
    
    for tab, (clip_num, clip_data) in zip(tabs, final_clips.items()):
        with tab:
            render_clip_card(clip_num, clip_data, clip_length)
    
    render_download_buttons(final_clips)
    
    if len(final_clips) < 3:
        st.warning(f"Only found {len(final_clips)} valid clips (requested 3).")


def render_transcription_view(transcript_text):
    st.subheader("Transcription")
    st.text_area("Transcription", transcript_text, height=200, key="transcription_display", disabled=True, label_visibility="collapsed")


def render_analysis_view(analysis_text):
    st.subheader("Clip Analysis")
    st.write(analysis_text)


def render_debug_view(analysis):
    with st.expander("Debug Info - Raw Analysis Output"):
        st.code(analysis)


def render_error_view(message, details=None):
    st.error(message)
    if details:
        with st.expander("Error Details"):
            st.code(details)