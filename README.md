# Convert Videos to Short Clips

A Python application that automatically converts long videos into engaging vertical short clips with smart cropping and auto-generated captions.

## Features

- **Smart Vertical Cropping**: Automatically detects and follows the subject (faces or motion) to create 9:16 vertical videos
- **AI-Powered Clip Selection**: Uses GPT-4 to analyze transcripts and identify the most engaging 30-second segment
- **Auto-Captions**: Generates captions using OpenAI Whisper and burns them into the video
- **Streamlit UI**: Easy-to-use web interface for uploading and processing videos

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.env` file with your OpenAI API key:
   ```
   api=your_openai_api_key
   ```

4. Install ImageMagick (required for moviepy text clips):
   - Download from https://imagemagick.org/script/download.php
   - During installation, check "Install legacy utilities (e.g. convert)"

## Usage

### Streamlit Web App

```bash
streamlit run test.py
```

Then open your browser to `http://localhost:8501` and upload a video.

### Command Line

#### Crop video to vertical format:
```bash
python crop_videos.py input_video.mp4 output_vertical.mp4
```

#### Batch process directory:
```bash
python crop_videos.py input_dir/ output_dir/ --batch
```

#### Add captions to video:
```bash
python captions.py input_video.mp4 output_video.mp4
```

## Requirements

- Python 3.8+
- OpenCV (`cv2`)
- MoviePy
- NumPy
- OpenAI
- python-dotenv
- Streamlit
- ImageMagick

## How It Works

1. **Video Upload**: User uploads a video file
2. **Transcription**: Audio is extracted and transcribed using OpenAI Whisper
3. **AI Analysis**: GPT-4 analyzes the transcript to find the most engaging 30-second segment
4. **Smart Cropping**: The selected clip is cropped to vertical format using face detection and motion tracking
5. **Caption Generation**: Captions are generated from the transcript and burned into the final video
