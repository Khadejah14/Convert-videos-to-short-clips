
import os
import tempfile
import textwrap
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
import openai
from dotenv import load_dotenv

# Load environment variables

# Load environment variables
load_dotenv()

# Manually set ImageMagick path if not found in environmental variables
from moviepy.config_defaults import IMAGEMAGICK_BINARY
if os.name == 'nt':
    if not os.path.isfile(IMAGEMAGICK_BINARY):
        # Try common paths
        possible_paths = [
            r"C:\Program Files\ImageMagick\magick.exe",
            r"C:\Program Files (x86)\ImageMagick\magick.exe",
        ]
        for p in possible_paths:
            if os.path.isfile(p):
                try:
                    from moviepy.config import change_settings
                    change_settings({"IMAGEMAGICK_BINARY": p})
                    print(f"Set ImageMagick binary to: {p}")
                    break
                except Exception as e:
                    print(f"Failed to set ImageMagick binary: {e}")


def create_captions_video(video_path, output_path, api_key=None):
    """
    Takes a video file, generates transcripts using OpenAI Whisper,
    and creates a new video with visually appealing captions burned in.
    
    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output video with captions.
        api_key (str, optional): OpenAI API key. defaults to os.getenv("api").
    """
    
    # Set API key
    if api_key:
        openai.api_key = api_key
    elif os.getenv("api"):
        openai.api_key = os.getenv("api")
    else:
        raise ValueError("OpenAI API key not found. Please set 'api' in .env or pass it as an argument.")

    print(f"Processing video: {video_path}")

    # 1. Extract Audio
    try:
        video = VideoFileClip(video_path)
        # Create a temp audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
        
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False

    # 2. Transcribe Audio
    print("Transcribing audio...")
    try:
        with open(temp_audio_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
    except Exception as e:
        print(f"Error during transcription: {e}")
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return False
    finally:
        # Clean up temp audio
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    segments = transcript.segments

    # 3. Create Caption Clips
    print(f"Generating captions for {len(segments)} segments...")
    
    captions = []
    
    # Video dimensions
    W, H = video.size
    
    # Styling Configuration
    font_size = int(H * 0.05) # Dynamic font size based on height
    font = 'Arial-Bold'
    text_color = 'white'
    stroke_color = 'black'
    stroke_width = 2
    bg_color = None # Transparent background for text clip itself
    
    # Create the base video clip
    captions.append(video)

    for segment in segments:
        start_time = segment.start
        end_time = segment.end
        text = segment.text.strip()
        
        # Word wrapping to ensure text fits on screen
        # Estimate chars per line based on width and font size (rough approximation)
        # Assuming avg char width is 0.5 * font_size
        max_chars_line = int((W * 0.8) / (font_size * 0.5))
        wrapped_text = "\n".join(textwrap.wrap(text, width=max_chars_line))

        # Create TextClip
        # Note: TextClip requires ImageMagick to be installed and configured.
        try:
            txt_clip = TextClip(
                wrapped_text, 
                fontsize=font_size, 
                font=font, 
                color=text_color, 
                bg_color='black',
                stroke_width=0,
                align='center'
            )
            
            # Position the text at the bottom center
            txt_clip = txt_clip.set_position(('center', 0.8), relative=True) # 80% down
            txt_clip = txt_clip.set_start(start_time).set_end(end_time)
            
            captions.append(txt_clip)
            
        except Exception as e:
            print(f"Error creating text clip for segment '{text}': {e}")
            # Continue to next segment if one fails
            continue

    # 4. Composite and Write Output
    print("Compositing video...")
    try:
        final_video = CompositeVideoClip(captions, size=video.size)
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac', verbose=False, logger=None)
        print(f"Successfully created captioned video: {output_path}")
        return True
    except Exception as e:
        print(f"Error writing output video: {e}")
        return False
    finally:
        video.close()
        final_video.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add visually appealing captions to a video using OpenAI Whisper.")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("output_video", help="Path to output video file")
    parser.add_argument("--api_key", help="OpenAI API Key (optional if set in .env)", default=None)
    
    args = parser.parse_args()
    
    create_captions_video(args.input_video, args.output_video, args.api_key)

# print("ni hao, zao jiu hao le")
