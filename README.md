# ASL Alphabet + Digits (Landmark-based) — quick project

## What this repo contains
- Preprocessing: extract MediaPipe hand landmarks from images and short videos
- Train static MLP for static letters/digits
- Train LSTM for dynamic letters (J, Z)
- Real-time inference using phone camera (DroidCam/IP Webcam) and OpenCV

## Quick setup
1. Create virtualenv and install:
   ```
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. Prepare data:
   - Put static letter/digit images in `data/static_images/<LABEL>/*.jpg`
   - Put short videos (1–2s) for `J` and `Z` in `data/dynamic_videos/` or frame folders.

3. Preprocess static:
   ```
   python scripts/preprocess_images.py data/static_images data/landmarks/static
   ```
   Preprocess dynamic (label J):
   ```
   python scripts/preprocess_videos.py data/dynamic_videos data/landmarks/dynamic J
   # repeat for Z if separate folder
   ```

4. Train static model:
   ```
   python scripts/train_static.py --landmark_dir data/landmarks/static --save models/static_mlp.pth --epochs 20
   ```

5. Train dynamic model:
   ```
   python scripts/train_dynamic.py --landmark_dir data/landmarks/dynamic --save models/dynamic_lstm.pth --epochs 30
   ```

6. Run realtime demo (replace IP and port shown by DroidCam):
   ```
   python scripts/infer_realtime.py --cam_url "http://192.168.0.5:8080/video" --static_model models/static_mlp.pth --dynamic_model models/dynamic_lstm.pth
   ```

## Notes & tips
- Use single-hand, 640x480 video to speed up.
- If `cv2.VideoCapture` fails, try the JPEG endpoint `http://<ip>:<port>/shot.jpg` and fetch frames manually.
- Tune `motion_thresh` in `infer_realtime.py` to detect J/Z reliably.
- For quick submission: if you don't have time to collect data, use public static ASL datasets for letters and digits (preprocess them), and collect a few short videos of your own for J/Z.
- For demo reliability, use background with good lighting and avoid occlusion.
