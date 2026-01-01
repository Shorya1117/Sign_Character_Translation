import cv2
import time
import argparse
import numpy as np
import torch
import mediapipe as mp
from collections import deque
from scripts.models import StaticMLP  # static model only

mp_hands = mp.solutions.hands

# ---------- Helper functions ----------
def normalize_landmarks(lm):
    lm = lm.copy()
    origin = lm[0].copy()
    lm[:, 0] -= origin[0]
    lm[:, 1] -= origin[1]
    lm[:, 2] -= origin[2]
    scale = np.max(np.linalg.norm(lm, axis=1)) + 1e-6
    lm /= scale
    return lm

def landmarks_to_feature(lm):
    return lm[:, :2].reshape(-1).astype(np.float32)

# ---------- Load static model ----------
def load_static_model(path, device):
    ckpt = torch.load(path, map_location=device)
    class_map = ckpt.get('class_map')
    n_classes = len(class_map)
    model = StaticMLP(input_dim=42, n_classes=n_classes)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    inv_map = {v: k for k, v in class_map.items()}
    return model, inv_map

# ---------- Camera open helper ----------
def open_camera(cam_url):
    # If full HTTP URL → open directly
    if cam_url.startswith("http://") or cam_url.startswith("https://"):
        print(f" Using HTTP stream: {cam_url}")
        return cv2.VideoCapture(cam_url)

    # Else treat as webcam index
    try:
        idx = int(cam_url)
        print(f"🎥 Using webcam index: {idx}")
        return cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    except:
        print(" Invalid camera URL/index.")
        return None

# ---------- Main ----------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    static_model, static_inv = load_static_model(args.static_model, device)

    # FIXED CAMERA SELECTION (URL + INDEX support)
    cap = open_camera(args.cam_url)

    if cap is None or not cap.isOpened():
        print(" ERROR: Camera failed to open. Check URL or index.")
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    buffer = deque(maxlen=args.max_len)
    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame. Check camera connection.")
            time.sleep(1)
            continue

        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)

        display_label = "No hand"
        conf = 0.0

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            arr = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            arr = normalize_landmarks(arr)

            # Run static model
            feat = landmarks_to_feature(arr)
            with torch.no_grad():
                t = torch.from_numpy(feat).unsqueeze(0).to(device)
                out = static_model(t)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                idx = int(np.argmax(probs))
                display_label = static_inv[idx]
                conf = float(probs[idx])

        # FPS + display
        fps = 1.0 / (time.time() - fps_time + 1e-6)
        fps_time = time.time()

        cv2.putText(frame, f"{display_label}  {conf:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("ASL Realtime", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- Entry ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_url', default='0', help='Webcam index OR DroidCam URL')
    parser.add_argument('--static_model', default='models/static_mlp.pth')
    parser.add_argument('--max_len', type=int, default=60)
    args = parser.parse_args()
    main(args)
