import streamlit as st


def render_sidebar():
    with st.sidebar:
        st.header("Settings")
        
        clip_count = st.slider(
            "Number of clips to generate",
            min_value=1,
            max_value=3,
            value=3,
            help="Generate 1-3 clips from the video"
        )
        
        clip_length = st.slider(
            "Clip Length (seconds)",
            min_value=15,
            max_value=60,
            step=15,
            value=30,
            help="Target duration for each clip: 15s, 30s, or 60s"
        )
        
        st.divider()
        
        caption_style = st.selectbox(
            "Caption Style",
            options=["default", "minimal", "highlight"],
            index=0,
            help="default: Solid black background | minimal: Transparent, subtle | highlight: Bold, word emphasis"
        )
        
        st.divider()
        
        st.subheader("Advanced")
        
        use_vision = st.checkbox(
            "Enable GPT-4o Vision Analysis",
            value=False,
            help="Uses GPT-4o vision to analyze keyframes for visual hooks. More accurate but uses more API credits."
        )
    
    return clip_count, clip_length, caption_style, use_vision
