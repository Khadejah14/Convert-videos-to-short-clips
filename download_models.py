import os
import urllib.request
from pathlib import Path


MODEL_DIR = Path(__file__).parent / "assets" / "config" / "model_weights"
MODELS = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
}


def download_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    for filename, url in MODELS.items():
        filepath = MODEL_DIR / filename
        if filepath.exists():
            print(f"Already exists: {filename}")
            continue
        
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  Saved: {filename} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  Failed to download {filename}: {e}")
            if filepath.exists():
                filepath.unlink()


if __name__ == "__main__":
    download_models()
    print("\nDone! Models are in:", MODEL_DIR)
