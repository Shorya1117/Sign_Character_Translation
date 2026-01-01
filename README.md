# ASL Sign Language Recognition Project

This project implements a **real-time Sign Language Recognition system** using **MediaPipe Hand Landmarks**, **OpenCV**, and a **PyTorch-based MLP model**. The system can take input from a **laptop webcam or mobile phone camera (via DroidCam)** and predict sign language gestures in real time.

---

## 1. Project Overview

### What this project does

* Detects hands in real time using **MediaPipe**
* Extracts **21 hand landmarks**
* Feeds landmarks into a **trained neural network (Static MLP)**
* Predicts the corresponding **sign language gesture**
* Displays prediction live on the camera feed

### Why this approach

* No GPU required
* Works on low-end laptops (CPU only)
* Faster than image-based CNN models
* Suitable for real-time applications

---

## 2. Technologies Used

* **Python 3.10**
* **MediaPipe** – Hand landmark detection
* **OpenCV** – Camera handling & visualization
* **PyTorch** – Model training and inference
* **NumPy** – Numerical processing
* **DroidCam** – Mobile phone as webcam (optional)

---

## 3. System Requirements

### Hardware

* Laptop / PC (Windows)
* Webcam OR Android phone (for DroidCam)
* Minimum 8 GB RAM recommended

### Software

* Windows 10/11
* Python 3.10.x (important)
* VS Code / PowerShell

> ⚠️ GPU is **NOT required**. Intel HD Graphics is sufficient.

---

## 4. Project Folder Structure

```
asl_project/
│
├── scripts/
│   ├── infer_realtime.py      # Real-time inference
│   ├── train_static.py        # Model training
│   ├── models.py              # MLP model definition
│   ├── dataset.py             # Dataset loader
│   └── __init__.py
│
├── dataset/
│   ├── landmarks/             # Saved landmark .npy files
│
├── models/
│   └── static_mlp.pt           # Trained model
│
├── venv/                      # Virtual environment
├── requirements.txt
└── README.md
```

---

## 5. Python Environment Setup (From Basics)

### Step 1: Install Python 3.10

Download from:
[https://www.python.org/downloads/release/python-31011/](https://www.python.org/downloads/release/python-31011/)

During installation:

* ✅ Check **Add Python to PATH**

Verify installation:

```
python --version
```

---

### Step 2: Create Virtual Environment

From project root:

```
python -m venv venv
```

Activate it:

```
venv\Scripts\activate
```

---

### Step 3: Install Dependencies

```
pip install -r requirements.txt
```

If requirements.txt not available:

```
pip install mediapipe opencv-python torch numpy
```

---

## 6. Training the Model

### Landmark dataset required

* Each class has its own folder
* Each sample is a `.npy` file

### Train command

```
python -m scripts.train_static --landmark_dir dataset/landmarks
```

After training, model will be saved as:

```
models/static_mlp.pt
```

---

## 7. Running Real-Time Inference (Laptop Webcam)

### Command

```
python -m scripts.infer_realtime
```

This will:

* Open default webcam
* Detect hands
* Show predicted sign on screen

---

## 8. Using Mobile Camera with DroidCam (IMPORTANT)

### Step 1: Install DroidCam

* Install **DroidCam Client** on PC
* Install **DroidCam App** on Android phone

Download PC client:
[https://www.dev47apps.com/droidcam/windows/](https://www.dev47apps.com/droidcam/windows/)

---

### Step 2: Connect Phone and Laptop

Make sure:

* Phone and laptop are on **same Wi-Fi network**

Open DroidCam app on phone → you will see:

* Device IP (example: 192.168.1.6)
* Port (example: 4747)

---

### Step 3: Start DroidCam Server

On PC:

* Open **DroidCam Client**
* Enter phone IP
* Click **Start**

---

### Step 4: Run ASL Model with DroidCam

Use this command:

```
python -m scripts.infer_realtime --cam_url http://<PHONE_IP>:4747/video
```

Example:

```
python -m scripts.infer_realtime --cam_url http://192.168.1.6:4747/video
```

---

## 9. Common Errors & Fixes

### Error: ModuleNotFoundError: No module named 'scripts'

✔ Always run using `-m` from project root

### Error: Camera failed to open

✔ Check DroidCam running
✔ Check correct IP
✔ Test URL in browser

### Error: Python not found

✔ Disable App Execution Aliases
✔ Add Python to PATH

---

## 10. Deployment Options

### Option 1: Desktop Executable (Recommended)

```
pip install pyinstaller
pyinstaller --onefile scripts/infer_realtime.py
```

Output:

```
dist/infer_realtime.exe
```

---

### Option 2: Local Web App

* Flask backend
* Browser-based demo

---

## 11. Final Notes

* This system runs fully offline
* No GPU required
* Real-time performance on CPU
* Suitable for academic projects and demos

---

## 12. Author

**ASL Sign Language Recognition Project**
Developed for academic and research purposes

---

If you face any issue, re-check commands and environment setup carefully.
