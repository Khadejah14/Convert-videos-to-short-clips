import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import tempfile
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_merge_video_audio

class SmartVerticalCropper:
    def __init__(self, smoothing_factor=0.2, min_face_size=0.1):
        """
        Initialize the smart cropper
        
        Args:
            smoothing_factor: How smooth the camera movement should be (0-1)
                             Lower = smoother, Higher = more responsive
            min_face_size: Minimum face size as fraction of frame height
        """
        self.smoothing_factor = smoothing_factor
        self.min_face_size = min_face_size
        
        # Load face detector (using Haar cascade for simplicity)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Alternative: Use DNN-based face detector (more accurate)
        self.dnn_net = None
        self.use_dnn = False
        self.initialize_dnn_detector()
    
    def initialize_dnn_detector(self):
        """Initialize DNN face detector for better accuracy"""
        try:
            # Download model files first:
            # https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
            # https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
            
            prototxt_path = "deploy.prototxt"
            caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"
            
            if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
                self.dnn_net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
                self.use_dnn = True
                print("Using DNN face detector for better accuracy")
            else:
                print("Using Haar cascade face detector")
                print("For better accuracy, download DNN model files:")
                print("1. deploy.prototxt: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt")
                print("2. res10_300x300_ssd_iter_140000.caffemodel: https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")
        except:
            print("Using Haar cascade face detector")
    
    def detect_faces_dnn(self, frame, confidence_threshold=0.5):
        """Detect faces using DNN"""
        if self.dnn_net is None:
            return []
        
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, 
            (300, 300), (104.0, 177.0, 123.0)
        )
        
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                faces.append((x1, y1, x2-x1, y2-y1))
        
        return faces
    
    def detect_faces(self, frame):
        """Detect faces using available method"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.use_dnn:
            faces = self.detect_faces_dnn(frame)
        else:
            # Use Haar cascade
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(int(frame.shape[0] * self.min_face_size), 
                         int(frame.shape[0] * self.min_face_size))
            )
        
        return faces
    
    def detect_subject(self, frame, prev_frame=None):
        """
        Detect main subject (face or motion center)
        
        Returns:
            (x_center, y_center, importance_score)
        """
        faces = self.detect_faces(frame)
        
        if len(faces) > 0:
            # Find the largest face (assuming it's the main subject)
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            return (x + w//2, y + h//2, 1.0)  # High importance for faces
        
        # If no faces detected, try motion detection
        if prev_frame is not None:
            motion_center = self.detect_motion_center(prev_frame, frame)
            if motion_center:
                return (motion_center[0], motion_center[1], 0.5)  # Medium importance
        
        # Fallback to center of frame
        height, width = frame.shape[:2]
        return (width // 2, height // 2, 0.1)  # Low importance
    
    def detect_motion_center(self, prev_frame, current_frame):
        """Detect center of motion using optical flow"""
        try:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Compute magnitude and angle of flow vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Threshold to find significant motion
            motion_mask = magnitude > 1.0
            
            if np.any(motion_mask):
                # Find center of motion
                y_coords, x_coords = np.where(motion_mask)
                if len(x_coords) > 0 and len(y_coords) > 0:
                    center_x = int(np.mean(x_coords))
                    center_y = int(np.mean(y_coords))
                    return (center_x, center_y)
        except:
            pass
        
        return None
    
    def smooth_positions(self, positions, window_size=5):
        """Apply smoothing to positions"""
        if len(positions) < window_size:
            return positions
        
        smoothed = []
        for i in range(len(positions)):
            start = max(0, i - window_size // 2)
            end = min(len(positions), i + window_size // 2 + 1)
            window = positions[start:end]
            
            # Weighted average - recent positions get more weight
            weights = np.linspace(0.5, 1.5, len(window))
            weights = weights / weights.sum()
            
            weighted_avg = np.average(window, weights=weights)
            smoothed.append(int(weighted_avg))
        
        return smoothed
    
    def crop_video(self, input_path, output_path):
        """
        Main function to crop video to vertical format following subject
        
        Args:
            input_path: Path to input video file
            output_path: Path for output vertical video
        """
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {input_path}")
            return False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Original video: {width}x{height}, {fps} FPS")
        print(f"Total frames: {total_frames}")
        
        # Calculate crop dimensions for 9:16 vertical
        crop_height = height
        crop_width = int(crop_height * 9 / 16)  # 9:16 aspect ratio
        
        # If video is already taller than wide, don't crop
        if height >= width:
            print("Video is already vertical or square, adjusting crop...")
            crop_width = int(min(width, height * 9/16))
        
        print(f"Crop dimensions: {crop_width}x{crop_height}")
        
        # Prepare video writer
        try:
            # Create a temporary file for the video-only output
            temp_video_path = tempfile.mktemp(suffix='.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (crop_width, crop_height))
        except:
            print("Warning: avc1 codec failed, falling back to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (crop_width, crop_height))
        
        # Store positions for analysis
        x_positions = []
        y_positions = []
        importance_scores = []
        prev_frame = None
        
        print("\nAnalyzing video and tracking subject...")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect subject position
            subject_pos = self.detect_subject(frame, prev_frame)
            x_center, y_center, importance = subject_pos
            
            x_positions.append(x_center)
            y_positions.append(y_center)
            importance_scores.append(importance)
            
            prev_frame = frame.copy()
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        
        print("\nApplying smoothing to camera movements...")
        
        # Smooth positions (more smoothing for less important frames)
        x_smoothed = []
        for i in range(len(x_positions)):
            # Adjust smoothing based on importance
            window_size = 15 if importance_scores[i] < 0.3 else 5
            start = max(0, i - window_size // 2)
            end = min(len(x_positions), i + window_size // 2 + 1)
            
            window = x_positions[start:end]
            window_weights = importance_scores[start:end]
            
            # Weighted average based on importance
            if sum(window_weights) > 0:
                weighted_avg = np.average(window, weights=window_weights)
            else:
                weighted_avg = np.mean(window)
            
            # Apply smoothing factor
            if x_smoothed:
                smoothed = int(x_smoothed[-1] * (1 - self.smoothing_factor) + 
                              weighted_avg * self.smoothing_factor)
            else:
                smoothed = int(weighted_avg)
            
            x_smoothed.append(smoothed)
        
        print("Applying smart cropping...")
        
        # Second pass: Apply cropping
        cap = cv2.VideoCapture(input_path)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx < len(x_smoothed):
                # Calculate crop position
                target_x = x_smoothed[frame_idx]
                
                # Ensure crop stays within bounds
                x_start = target_x - crop_width // 2
                x_start = max(0, x_start)
                x_start = min(x_start, width - crop_width)
                
                # Apply crop
                cropped_frame = frame[0:crop_height, x_start:x_start + crop_width]
                
                # Resize if necessary (shouldn't be needed but just in case)
                if cropped_frame.shape[1] != crop_width or cropped_frame.shape[0] != crop_height:
                    cropped_frame = cv2.resize(cropped_frame, (crop_width, crop_height))
                
                out.write(cropped_frame)
            
            frame_idx += 1
        
        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Add audio back to the cropped video
        print("Adding audio to cropped video...")
        try:
            # Extract audio from original video
            temp_audio_path = tempfile.mktemp(suffix='.mp3')
            original_clip = VideoFileClip(input_path)
            original_clip.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
            original_clip.close()
            
            # Merge video and audio
            ffmpeg_merge_video_audio(temp_video_path, temp_audio_path, output_path, vcodec='copy', acodec='copy', logger=None)
            
            # Cleanup temp files
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
        except Exception as e:
            print(f"Warning: Failed to add audio: {e}")
            # If audio merge fails, at least move the video file to output
            if os.path.exists(temp_video_path):
                import shutil
                shutil.move(temp_video_path, output_path)

        print(f"\n✓ Successfully created vertical video: {output_path}")
        print(f"  Original: {width}x{height}")
        print(f"  Cropped:  {crop_width}x{crop_height}")
        
        return True
    
    def process_directory(self, input_dir, output_dir):
        """Process all videos in a directory"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv'}
        video_files = [f for f in input_dir.iterdir() 
                      if f.suffix.lower() in video_extensions]
        
        print(f"Found {len(video_files)} video(s) to process")
        
        for video_file in video_files:
            output_path = output_dir / f"vertical_{video_file.name}"
            print(f"\nProcessing: {video_file.name}")
            
            success = self.crop_video(str(video_file), str(output_path))
            
            if success:
                print(f"Saved to: {output_path}")
            else:
                print(f"Failed to process: {video_file.name}")

def main():
    parser = argparse.ArgumentParser(description='Smart video cropper for vertical format')
    parser.add_argument('input', help='Input video file or directory')
    parser.add_argument('output', help='Output video file or directory')
    parser.add_argument('--smooth', type=float, default=0.2, 
                       help='Smoothing factor (0-1, default: 0.2)')
    parser.add_argument('--batch', action='store_true',
                       help='Process all videos in input directory')
    
    args = parser.parse_args()
    
    # Initialize cropper
    cropper = SmartVerticalCropper(smoothing_factor=args.smooth)
    
    if args.batch:
        # Process directory
        cropper.process_directory(args.input, args.output)
    else:
        # Process single video
        success = cropper.crop_video(args.input, args.output)
        if not success:
            print("Failed to process video")

if __name__ == "__main__":
    main()