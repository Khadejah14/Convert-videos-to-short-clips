import cv2
import numpy as np
import os
import subprocess
import json
import tempfile
from pathlib import Path
import argparse


# hellow in new advancement 
print("Hi, supposed to crop better, hope not to ended up ruininng it lol")

class SmartVerticalCropper:
    def __init__(self, smoothing_factor=0.2, min_face_size=0.1):
        """
        Initialize the smart cropper.

        Args:
            smoothing_factor: How smooth the camera movement should be (0-1)
                              Lower = smoother, Higher = more responsive
            min_face_size:    Minimum face size as fraction of frame height
        """
        self.smoothing_factor = smoothing_factor
        self.min_face_size = min_face_size

        # Haar cascade (fast, always available)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # DNN face detector (more accurate, optional)
        self.dnn_net = None
        self.use_dnn = False
        self._initialize_dnn_detector()

    # ------------------------------------------------------------------
    # Face detection
    # ------------------------------------------------------------------

    def _initialize_dnn_detector(self):
        model_dir = Path(__file__).parent / "assets" / "config" / "model_weights"
        root_dir = Path(__file__).parent
        
        prototxt_paths = [
            model_dir / "deploy.prototxt",
            root_dir / "deploy.prototxt"
        ]
        caffemodel_paths = [
            model_dir / "res10_300x300_ssd_iter_140000.caffemodel",
            root_dir / "res10_300x300_ssd_iter_140000.caffemodel"
        ]
        
        prototxt = None
        caffemodel = None
        
        for pp, cp in zip(prototxt_paths, caffemodel_paths):
            if pp.exists() and cp.exists():
                prototxt = str(pp)
                caffemodel = str(cp)
                break
        
        if prototxt and caffemodel:
            try:
                self.dnn_net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
                self.use_dnn = True
                print("Using DNN face detector (higher accuracy)")
                return
            except Exception:
                pass
        print("Using Haar cascade face detector")

    def _detect_faces_dnn(self, frame, confidence_threshold=0.5):
        if self.dnn_net is None:
            return []
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        faces = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces

    def _detect_faces(self, frame):
        if self.use_dnn:
            return self._detect_faces_dnn(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(
                int(frame.shape[0] * self.min_face_size),
                int(frame.shape[0] * self.min_face_size),
            ),
        )

    # ------------------------------------------------------------------
    # Subject / motion detection
    # ------------------------------------------------------------------

    def _detect_motion_center(self, prev_frame, curr_frame):
        try:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_mask = magnitude > 1.0
            if np.any(motion_mask):
                y_coords, x_coords = np.where(motion_mask)
                return int(np.mean(x_coords)), int(np.mean(y_coords))
        except Exception:
            pass
        return None

    def _detect_subject(self, frame, prev_frame=None):
        """Returns (x_center, y_center, importance_score)."""
        faces = self._detect_faces(frame)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            return x + w // 2, y + h // 2, 1.0

        if prev_frame is not None:
            mc = self._detect_motion_center(prev_frame, frame)
            if mc:
                return mc[0], mc[1], 0.5

        height, width = frame.shape[:2]
        return width // 2, height // 2, 0.1

    # ------------------------------------------------------------------
    # Crop position analysis (pure Python / NumPy — no video writing)
    # ------------------------------------------------------------------

    def _analyze_crop_positions(self, input_path, width, height, crop_width):
        """
        First pass: read every frame, detect subject, record x positions.
        Returns a list of smoothed integer x_start values, one per frame.
        """
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        x_positions = []
        importance_scores = []
        prev_frame = None
        frame_count = 0

        print("Analysing video and tracking subject…")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            x_center, _, importance = self._detect_subject(frame, prev_frame)
            x_positions.append(x_center)
            importance_scores.append(importance)
            prev_frame = frame.copy()
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  {frame_count}/{total_frames} frames analysed")

        cap.release()

        # Smooth x positions
        print("Smoothing camera movement…")
        x_smoothed = []
        for i, (xp, imp) in enumerate(zip(x_positions, importance_scores)):
            window_size = 15 if imp < 0.3 else 5
            start = max(0, i - window_size // 2)
            end = min(len(x_positions), i + window_size // 2 + 1)
            window = x_positions[start:end]
            weights = importance_scores[start:end]
            weighted_avg = (
                np.average(window, weights=weights)
                if sum(weights) > 0
                else np.mean(window)
            )
            if x_smoothed:
                smoothed = int(
                    x_smoothed[-1] * (1 - self.smoothing_factor)
                    + weighted_avg * self.smoothing_factor
                )
            else:
                smoothed = int(weighted_avg)
            x_smoothed.append(smoothed)

        # Convert to x_start values, clamped to valid range
        x_starts = []
        for xc in x_smoothed:
            xs = xc - crop_width // 2
            xs = max(0, min(xs, width - crop_width))
            x_starts.append(xs)

        return x_starts

    # ------------------------------------------------------------------
    # FFmpeg helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_ffmpeg():
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    @staticmethod
    def _get_video_info(input_path):
        """Use ffprobe to get video metadata."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", input_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                return stream
        return {}

    def _build_crop_filter(self, x_starts, crop_width, crop_height, fps):
        """
        Build an FFmpeg sendcmd filter that sets crop x for every frame.
        Falls back to a simpler approach for very long videos.
        """
        # Use the sendcmd filter: one command per frame
        # Format: TIME crop x X; (newline) ...
        lines = []
        for frame_idx, x in enumerate(x_starts):
            t = frame_idx / fps
            lines.append(f"{t:.6f} crop x {x};")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crop_video(self, input_path, output_path, crf=18, preset="slow"):
        """
        Crop video to 9:16 vertical format in a SINGLE FFmpeg pass.

        Quality controls
        ----------------
        crf    : Constant Rate Factor. 0=lossless, 18=visually lossless,
                 23=default, 28=low quality. Recommended range: 15–22.
        preset : FFmpeg encoding speed preset (ultrafast … veryslow).
                 Slower preset = better compression at same quality.
        """
        if not self._check_ffmpeg():
            print("ERROR: ffmpeg not found. Please install ffmpeg and ensure it is on PATH.")
            return False

        input_path = str(input_path)

        # ---- Get video properties via ffprobe ----
        info = self._get_video_info(input_path)
        width = int(info.get("width", 0))
        height = int(info.get("height", 0))

        # Parse FPS (could be "30000/1001" or "30")
        fps_raw = info.get("r_frame_rate", "30/1")
        try:
            num, den = fps_raw.split("/")
            fps = float(num) / float(den)
        except Exception:
            fps = float(fps_raw)

        if width == 0 or height == 0:
            # Fallback to OpenCV for metadata
            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

        print(f"Original: {width}x{height} @ {fps:.2f} FPS")

        # ---- Calculate crop dimensions ----
        if height >= width:
            # Already vertical or square
            crop_width = int(min(width, height * 9 / 16))
        else:
            crop_width = int(height * 9 / 16)
        crop_height = height

        print(f"Crop window: {crop_width}x{crop_height}")

        # ---- Analyse subject positions (OpenCV read-only, no encode) ----
        x_starts = self._analyze_crop_positions(
            input_path, width, height, crop_width
        )

        # ---- Build per-frame crop filter file ----
        print("Building per-frame crop filter…")
        cmd_lines = []
        for frame_idx, x in enumerate(x_starts):
            t = frame_idx / fps
            cmd_lines.append(f"{t:.6f} crop x {x};")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as cmd_file:
            cmd_file.write("\n".join(cmd_lines))
            cmd_file_path = cmd_file.name

        # ---- Single-pass FFmpeg encode ----
        # sendcmd reads the per-frame x positions and drives the crop filter.
        # Audio is stream-copied (no re-encode) → zero audio quality loss.
        # Video is encoded once with H.264 at the chosen CRF.
        print(f"Encoding with FFmpeg (CRF={crf}, preset={preset})…")
        print("This is a single-pass encode — no quality degradation from double compression.")

        vf_filter = (
            f"sendcmd=f={cmd_file_path},"
            f"crop={crop_width}:{crop_height}"
        )

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", vf_filter,
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", preset,
            "-c:a", "copy",          # Audio: stream copy, zero loss
            "-movflags", "+faststart",
            output_path,
        ]

        try:
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print("FFmpeg error output:")
                print(result.stderr[-3000:])  # last 3000 chars of stderr
                # Clean up
                if os.path.exists(cmd_file_path):
                    os.remove(cmd_file_path)
                return False
        except Exception as e:
            print(f"FFmpeg execution error: {e}")
            if os.path.exists(cmd_file_path):
                os.remove(cmd_file_path)
            return False

        # Clean up temp filter file
        if os.path.exists(cmd_file_path):
            os.remove(cmd_file_path)

        print(f"\n[OK] Successfully created vertical video: {output_path}")
        print(f"  Original  : {width}x{height}")
        print(f"  Output    : {crop_width}x{crop_height}")
        print(f"  CRF used  : {crf}  (lower = better quality)")
        print(f"  Audio     : stream-copied (lossless)")
        return True

    def process_directory(self, input_dir, output_dir, crf=18, preset="slow"):
        """Process all videos in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv"}
        video_files = [
            f for f in input_dir.iterdir()
            if f.suffix.lower() in video_extensions
        ]
        print(f"Found {len(video_files)} video(s) to process")

        for video_file in video_files:
            output_path = output_dir / f"vertical_{video_file.name}"
            print(f"\nProcessing: {video_file.name}")
            success = self.crop_video(
                str(video_file), str(output_path), crf=crf, preset=preset
            )
            if success:
                print(f"Saved: {output_path}")
            else:
                print(f"Failed: {video_file.name}")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Smart vertical video cropper — FFmpeg single-pass, high quality"
    )
    parser.add_argument("input", help="Input video file or directory")
    parser.add_argument("output", help="Output video file or directory")
    parser.add_argument(
        "--smooth", type=float, default=0.2,
        help="Smoothing factor 0–1 (default: 0.2)"
    )
    parser.add_argument(
        "--crf", type=int, default=18,
        help="FFmpeg CRF quality (0=lossless, 18=visually lossless, 23=default). "
             "Range 15–22 recommended. (default: 18)"
    )
    parser.add_argument(
        "--preset", default="slow",
        choices=["ultrafast","superfast","veryfast","faster","fast",
                 "medium","slow","slower","veryslow"],
        help="FFmpeg encoding preset (default: slow)"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Process all videos in input directory"
    )
    args = parser.parse_args()

    cropper = SmartVerticalCropper(smoothing_factor=args.smooth)

    if args.batch:
        cropper.process_directory(
            args.input, args.output, crf=args.crf, preset=args.preset
        )
    else:
        success = cropper.crop_video(
            args.input, args.output, crf=args.crf, preset=args.preset
        )
        if not success:
            print("Failed to process video.")


if __name__ == "__main__":
    main()
