# Convert Videos to Short Clips

A Python application that automatically converts long videos into engaging vertical short clips with smart cropping and auto-generated captions using AI.

## Features

- **Smart Vertical Cropping**: Automatically detects and follows subjects (faces or motion) to create 9:16 vertical videos
- **AI-Powered Clip Selection**: Uses GPT-4o to analyze transcripts and identify the most engaging 30-second segment with viral potential
- **Auto-Captions**: Generates captions using OpenAI Whisper and burns them into the video
- **Streamlit UI**: Easy-to-use web interface for uploading and processing videos

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
streamlit run test.py
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

1. **Upload**: User uploads a video file
2. **Transcription**: Audio is extracted and transcribed using OpenAI Whisper
3. **AI Analysis**: GPT-4o analyzes the transcript to find the most engaging 30-second segment
4. **Smart Cropping**: The selected clip is cropped to vertical format using face detection and motion tracking
5. **Caption Generation**: Captions are generated from the transcript and burned into the final video

## Requirements

- Python 3.8+
- Streamlit
- OpenCV
- MoviePy
- NumPy
- OpenAI
- python-dotenv
- ImageMagick
