import streamlit as st
import io
import zipfile


def render_transcription(transcript_text):
    st.subheader("Transcription")
    st.text_area("", transcript_text, height=200, key="transcription_display")


def render_analysis(analysis_text):
    st.subheader("Multi-Clip Analysis")
    st.write(analysis_text)


def render_clip_results(final_clips, clip_length, show_expanded=True):
    st.divider()
    st.header("Generated Clips")
    
    with st.expander("View All Clips", expanded=show_expanded):
        tabs = st.tabs([clip['category'] for clip in final_clips.values()])
        
        for tab, (clip_num, clip_data) in zip(tabs, final_clips.items()):
            with tab:
                _render_single_clip(clip_num, clip_data, clip_length)
    
    _render_quick_downloads(final_clips)
    _render_zip_download(final_clips)
    
    if len(final_clips) < 3:
        st.warning(f"Only found {len(final_clips)} valid clips (requested 3). Some segments may have been too short, too long, or overlapping.")


def _render_single_clip(clip_num, clip_data, clip_length):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Start", f"{clip_data['start']:.1f}s")
    with col2:
        st.metric("End", f"{clip_data['end']:.1f}s")
    with col3:
        st.metric("Duration", f"{clip_data['duration']:.1f}s")
    with col4:
        st.metric("Target", f"{clip_length}s")
    
    st.subheader("Final Video")
    st.video(clip_data['final'])
    
    col_download1, col_download2 = st.columns(2)
    
    with col_download1:
        category_slug = clip_data['category'].lower().replace(' ', '_')
        with open(clip_data['original'], 'rb') as f:
            st.download_button(
                f"Original ({clip_data['category']})",
                f,
                f"clip_{clip_num}_{category_slug}_original.mp4"
            )
    
    with col_download2:
        suffix = "_captioned" if not clip_data.get('no_captions') else "_vertical"
        category_slug = clip_data['category'].lower().replace(' ', '_')
        with open(clip_data['final'], 'rb') as f:
            st.download_button(
                f"Final ({clip_data['category']})",
                f,
                f"clip_{clip_num}_{category_slug}{suffix}.mp4"
            )


def _render_quick_downloads(final_clips):
    if len(final_clips) > 1:
        st.divider()
        st.subheader("Quick Download All")
        
        cols = st.columns(min(len(final_clips), 3))
        for idx, (clip_num, clip_data) in enumerate(final_clips.items()):
            with cols[idx]:
                category_slug = clip_data['category'].lower().replace(' ', '_')
                with open(clip_data['final'], 'rb') as f:
                    st.download_button(
                        f"Download {clip_data['category']}",
                        f,
                        f"clip_{clip_num}_{category_slug}_final.mp4",
                        use_container_width=True
                    )


def _render_zip_download(final_clips):
    if len(final_clips) == 3:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for clip_num, clip_data in final_clips.items():
                category_slug = clip_data['category'].lower().replace(' ', '_')
                with open(clip_data['final'], 'rb') as f:
                    filename = f"clip_{clip_num}_{category_slug}_final.mp4"
                    zip_file.writestr(filename, f.read())
        
        zip_buffer.seek(0)
        
        st.download_button(
            "Download All as ZIP",
            zip_buffer.getvalue(),
            "all_clips.zip",
            mime="application/zip",
            use_container_width=True
        )


def render_debug_info(analysis):
    with st.expander("Debug Info - Raw Analysis Output"):
        st.code(analysis)


def render_error(message, details=None):
    st.error(message)
    if details:
        with st.expander("Error Details"):
            st.code(details)
