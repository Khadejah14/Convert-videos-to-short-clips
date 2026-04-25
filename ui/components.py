import streamlit as st
import io
import zipfile


class ProgressDisplay:
    def __init__(self):
        self.progress_bar = None
        self.status_text = None
    
    def init(self):
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
    
    def update(self, step, total_steps=6):
        if self.progress_bar:
            progress = int((step / total_steps) * 100)
            self.progress_bar.progress(progress)
    
    def set_status(self, text):
        if self.status_text:
            self.status_text.text(text)
    
    def clear(self):
        if self.progress_bar:
            self.progress_bar.empty()
        if self.status_text:
            self.status_text.empty()


STEP_MESSAGES = [
    "Step 1/6: Extracting audio...",
    "Step 2/6: Transcribing audio...",
    "Step 3/6: GPT analyzing content...",
    "Step 4/6: Extracting clips...",
    "Step 5/6: Cropping to vertical...",
    "Step 6/6: Adding captions..."
]


def render_progress_step(progress_ui, step, total_steps=6):
    if progress_ui:
        progress_ui.update(step, total_steps)
        if 0 <= step < len(STEP_MESSAGES):
            progress_ui.set_status(STEP_MESSAGES[step])


def render_clip_card(clip_num, clip_data, clip_length, is_winner=False, use_vision=False, vision_data=None):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Start", f"{clip_data['start']:.1f}s")
    with col2:
        st.metric("End", f"{clip_data['end']:.1f}s")
    with col3:
        st.metric("Duration", f"{clip_data['duration']:.1f}s")
    with col4:
        st.metric("Target", f"{clip_length}s")
    
    if use_vision and vision_data:
        score_col1, score_col2 = st.columns(2)
        with score_col1:
            st.metric("Text Score", f"{vision_data.get('text_score', 0):.1f}")
        with score_col2:
            st.metric("Visual Score", f"{vision_data.get('visual_score', 0):.1f}")
    
    if is_winner and vision_data and vision_data.get('visual_hook_description'):
        st.success(f"**Visual Hook:** {vision_data['visual_hook_description']}")
    
    st.subheader("Final Video")
    st.video(clip_data['final'])
    
    render_clip_downloads(clip_num, clip_data)


def render_clip_downloads(clip_num, clip_data):
    col_download1, col_download2 = st.columns(2)
    category_slug = clip_data['category'].lower().replace(' ', '_')
    
    with col_download1:
        with open(clip_data['original'], 'rb') as f:
            st.download_button(
                f"Original ({clip_data['category']})",
                f,
                f"clip_{clip_num}_{category_slug}_original.mp4"
            )
    
    with col_download2:
        suffix = "_captioned" if not clip_data.get('no_captions') else "_vertical"
        with open(clip_data['final'], 'rb') as f:
            st.download_button(
                f"Final ({clip_data['category']})",
                f,
                f"clip_{clip_num}_{category_slug}{suffix}.mp4"
            )


def render_download_buttons(final_clips):
    if len(final_clips) <= 1:
        return
    
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
    
    if len(final_clips) == 3:
        render_zip_download(final_clips)


def render_zip_download(final_clips):
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