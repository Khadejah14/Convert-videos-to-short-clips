import streamlit as st


class ProgressUI:
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
    
    def update_clip_progress(self, current, total):
        if self.progress_bar:
            self.progress_bar.progress(int((current / total) * 100))
    
    def clear(self):
        if self.progress_bar:
            self.progress_bar.empty()
        if self.status_text:
            self.status_text.empty()


def get_step_message(step):
    messages = [
        "Step 1/6: Extracting audio...",
        "Step 2/6: Transcribing audio...",
        "Step 3/6: GPT analyzing content...",
        "Step 4/6: Extracting clips...",
        "Step 5/6: Cropping to vertical...",
        "Step 6/6: Adding captions..."
    ]
    if 0 <= step < len(messages):
        return messages[step]
    return f"Step {step + 1}/6..."
