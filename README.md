# Convert Videos to Short Clips

A Python application that automatically converts long videos into engaging vertical short clips with smart cropping and auto-generated captions using AI.

## Features

- **Multi-Clip Generation**: Generate 1-3 clips from a single video simultaneously
- **AI-Powered Clip Categorization**: GPT-4 identifies three types of clips:
  - **Hook Focus**: Strongest opening grab - stops the scroll
  - **Emotional Peak**: Laughter, surprise, awe, or powerful insights
  - **Viral Moment**: Most shareable and quotable segment
- **Smart Vertical Cropping**: Automatically detects and follows subjects (faces or motion) to create 9:16 vertical videos
- **Auto-Captions**: Generates captions using OpenAI Whisper and burns them into the video
- **Flexible Download Options**: Download individual clips or all clips as a ZIP file
- **Streamlit UI**: Easy-to-use web interface with sidebar controls

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up environment variables:
   Create a `.env` file with your OpenAI API key:
   ```
   api=your_openai_api_key
   ```
3. Install ImageMagick (required for moviepy text clips):
   - Download from https://imagemagick.org/script/download.php
   - During installation, check "Install legacy utilities (e.g. convert)"

## Usage

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

## Project Structure

```
├── test.py             # Main Streamlit application
├── crop_videos.py      # Smart vertical video cropping
├── captions.py         # Caption generation and burn-in
├── requirements.txt    # Python dependencies
├── assets/
│   └── prompts/        # AI prompt templates
└── .env                # API keys (create this)
```

## How It Works

1. **Configure**: Select number of clips (1-3) and caption style in the sidebar
2. **Upload**: User uploads a video file
3. **Transcription**: Audio is extracted and transcribed using OpenAI Whisper
4. **AI Analysis**: GPT-4 analyzes the transcript to identify multiple clip segments:
   - Hook Focus (strongest opening)
   - Emotional Peak (most impactful moment)
   - Viral Moment (most shareable content)
5. **Smart Cropping**: Each clip is cropped to vertical format using face detection and motion tracking
6. **Caption Generation**: Captions are generated and burned into each final video
7. **Download**: Choose individual clips or download all as a ZIP

## Usage

1. Adjust clip count slider (1-3) and caption style in the sidebar
2. Upload a video file (mp4, avi, mov)
3. Click "Process Video" to generate clips
4. View clips in tabs, compare side-by-side
5. Download individual clips or all at once

## Requirements

- Python 3.8+
- Streamlit
- OpenCV
- MoviePy
- NumPy
- OpenAI
- python-dotenv
- ImageMagick

## Notes

- Each clip is automatically validated to be 15-30 seconds
- Overlapping clips are automatically deduplicated
- If fewer valid clips are found than requested, available clips are still generated
