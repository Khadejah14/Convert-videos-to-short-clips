
import os
import json
import tempfile
import textwrap
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
import openai
from dotenv import load_dotenv


def load_caption_preset(preset_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    preset_path = os.path.join(base_dir, "assets", "templates", "captions", f"{preset_name}.json")
    
    default_preset = {
        "font": "Arial-Bold",
        "font_size_ratio": 0.05,
        "color": "white",
        "bg_color": "black",
        "stroke_width": 0,
        "stroke_color": "black",
        "position": {"x": "center", "y": 0.8},
        "max_width_ratio": 0.8,
        "animation_type": "static"
    }
    
    if not os.path.exists(preset_path):
        print(f"Warning: Preset '{preset_name}' not found. Using default.")
        return default_preset
    
    try:
        with open(preset_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading preset '{preset_name}': {e}. Using default.")
        return default_preset

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


def create_captions_video(video_path, output_path, api_key=None, style_preset="default"):
    """
    Takes a video file, generates transcripts using OpenAI Whisper,
    and creates a new video with visually appealing captions burned in.
    
    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output video with captions.
        api_key (str, optional): OpenAI API key. defaults to os.getenv("api").
        style_preset (str): Name of the caption style preset (without .json).
                          Options: "default", "minimal", "highlight"
    """
    
    preset = load_caption_preset(style_preset)
    
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
    
    # Styling Configuration from preset
    font_size = int(H * preset["font_size_ratio"])
    font = preset["font"]
    text_color = preset["color"]
    stroke_color = preset["stroke_color"]
    stroke_width = preset["stroke_width"]
    bg_color = preset.get("background_color") or preset.get("bg_color", "black")
    max_chars_ratio = preset.get("max_width_ratio", 0.8)
    
    pos_dict = preset.get("position", {})
    pos_y = pos_dict.get("vertical_ratio", pos_dict.get("y", 0.85))
    pos_x_str = pos_dict.get("horizontal", pos_dict.get("x", "center"))
    pos_x = 0.5 if pos_x_str == "center" else float(pos_x_str)
    
    animation_type = preset.get("animation_type", "static")
    
    # Create the base video clip
    captions.append(video)

    for segment in segments:
        start_time = segment.start
        end_time = segment.end
        text = segment.text.strip()
        
        if not text:
            continue
        
        clip_duration = end_time - start_time
        if clip_duration < 0.3:
            end_time = start_time + 0.3
        
        # Word wrapping to ensure text fits on screen
        # Use more conservative char width estimate (avg char ~0.6 * font_size for most fonts)
        max_chars_line = int((W * max_chars_ratio) / (font_size * 0.6))
        max_chars_line = max(max_chars_line, 10)
        
        wrapped_lines = textwrap.wrap(text, width=max_chars_line)
        wrapped_text = "\n".join(wrapped_lines)
        
        # Adjust font size for multi-line text to ensure readability
        num_lines = len(wrapped_lines)
        if num_lines > 2:
            adjusted_font_size = int(font_size * 0.8)
        else:
            adjusted_font_size = font_size
        
        # Create TextClip
        # Note: TextClip requires ImageMagick to be installed and configured.
        try:
            txt_clip = TextClip(
                wrapped_text, 
                fontsize=adjusted_font_size, 
                font=font, 
                color=text_color, 
                bg_color=bg_color,
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                align='center'
            )
            
            # Position the text based on preset (bottom-center with padding)
            txt_clip = txt_clip.set_position(('center', 'bottom'))
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
    parser.add_argument("--style", help="Caption style preset (default, minimal, highlight)", default="default")
    
    args = parser.parse_args()
    
    create_captions_video(args.input_video, args.output_video, args.api_key, args.style)

# print("ni hao, zao jiu hao le")
