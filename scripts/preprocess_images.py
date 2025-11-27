# scripts/preprocess_images.py
import os, sys, json
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=1,
                       min_detection_confidence=0.5)

def extract_landmarks(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0]
    arr = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)  # (21,3)
    return arr

def normalize_landmarks(lm):
    # wrist center as origin (landmark 0)
    lm = lm.copy()
    origin = lm[0].copy()
    lm[:, 0] -= origin[0]
    lm[:, 1] -= origin[1]
    lm[:, 2] -= origin[2]
    # scale by max distance
    scale = np.max(np.linalg.norm(lm, axis=1)) + 1e-6
    lm /= scale
    return lm

def main(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    for cls in classes:
        cls_dir = os.path.join(src_dir, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        out_cls_dir = os.path.join(out_dir, cls)
        os.makedirs(out_cls_dir, exist_ok=True)
        for f in tqdm(files, desc=f"Processing {cls}"):
            path = os.path.join(cls_dir, f)
            img = cv2.imread(path)
            if img is None:
                continue
            lm = extract_landmarks(img)
            if lm is None:
                continue
            lm = normalize_landmarks(lm)  # (21,3)
            save_path = os.path.join(out_cls_dir, os.path.splitext(f)[0] + '.npy')
            np.save(save_path, {'landmarks': lm, 'label': cls})
    print("Done.")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python preprocess_images.py <static_images_dir> <out_landmark_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
