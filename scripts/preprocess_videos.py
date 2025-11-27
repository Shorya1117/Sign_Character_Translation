# scripts/preprocess_videos.py
import os, sys, glob
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

def extract_sequence_from_video(video_path, max_frames=60):
    cap = cv2.VideoCapture(video_path)
    seq = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or count >= max_frames:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            arr = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            seq.append(arr)
        count += 1
    cap.release()
    if len(seq) == 0:
        return None
    seq = np.stack(seq, axis=0)  # (T,21,3)
    return seq

def normalize_sequence(seq):
    # seq: (T,21,3)
    seq = seq.copy()
    # subtract wrist per frame and scale by max distance across frames
    seq = seq - seq[:, 0:1, :]
    scale = np.max(np.linalg.norm(seq.reshape(-1,3), axis=1)) + 1e-6
    seq /= scale
    return seq

def main(src_dir, out_dir, label, max_frames=60):
    os.makedirs(out_dir, exist_ok=True)
    files = []
    # accept video files
    for ext in ('mp4','avi','mov','mkv'):
        files.extend(glob.glob(os.path.join(src_dir, f'*.{ext}')))
    if not files:
        print("No video files found.")
        return
    for f in tqdm(files, desc=f"Processing {label} videos"):
        seq = extract_sequence_from_video(f, max_frames=max_frames)
        if seq is None:
            continue
        seq = normalize_sequence(seq)
        base = os.path.splitext(os.path.basename(f))[0]
        np.save(os.path.join(out_dir, f"{label}_{base}.npy"), {'landmarks': seq, 'label': label})
    print("Done.")

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python preprocess_videos.py <video_dir> <out_dir> <label> [max_frames]")
        sys.exit(1)
    src = sys.argv[1]; out = sys.argv[2]; label = sys.argv[3]
    maxf = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    main(src, out, label, maxf)
