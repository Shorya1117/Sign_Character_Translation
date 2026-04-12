# 🤟 Sign Language Recognition System

> Real-time ASL gesture recognition using MediaPipe, OpenCV, and PyTorch — no GPU required.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Training the Model](#training-the-model)
- [Running the App](#running-the-app)
- [Using Mobile Camera (DroidCam)](#using-mobile-camera-droidcam)
- [Deployment Options](#deployment-options)
- [Troubleshooting](#troubleshooting)
- [Authors](#authors)

---

## Overview

This project implements a **real-time Sign Language Recognition system** that detects hand gestures via webcam and predicts the corresponding ASL sign using a lightweight neural network. It is designed to run entirely on CPU, making it accessible on standard laptops without any dedicated GPU.

---

## Features

- 🖐️ Real-time hand detection using **MediaPipe** (21 landmarks)
- 🧠 Gesture prediction via a trained **MLP neural network** (PyTorch)
- 📷 Supports **laptop webcam** or **Android phone camera** (via DroidCam)
- 💻 Fully **offline** — no internet connection required at runtime
- ⚡ Fast inference on **CPU only**

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| MediaPipe | Hand landmark detection |
| OpenCV | Camera handling & frame visualization |
| PyTorch | Model training and inference |
| NumPy | Numerical processing |
| DroidCam *(optional)* | Use Android phone as a webcam |

---

## System Requirements

### Hardware
- Laptop or PC (Windows 10/11)
- Webcam **or** Android phone (for DroidCam)
- Minimum **8 GB RAM** recommended

### Software
- Windows 10 or 11
- Python **3.10.x** *(version-sensitive)*
- VS Code or PowerShell

> ⚠️ A dedicated GPU is **not required**. Intel HD Graphics is sufficient.

---

## Project Structure

```
asl_project/
│
├── scripts/
│   ├── infer_realtime.py      # Real-time inference script
│   ├── train_static.py        # Model training script
│   ├── models.py              # MLP model architecture
│   ├── dataset.py             # Dataset loader
│   └── __init__.py
│
├── dataset/
│   └── landmarks/             # Saved landmark .npy files per class
│
├── models/
│   └── static_mlp.pt          # Trained model weights
│
├── venv/                      # Python virtual environment
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Step 1 — Install Python 3.10

Download from the official site:  
🔗 https://www.python.org/downloads/release/python-31011/

During installation, make sure to check:  
✅ **Add Python to PATH**

Verify the installation:
```bash
python --version
```

---

### Step 2 — Create a Virtual Environment

From the project root directory:
```bash
python -m venv venv
```

Activate it:
```bash
venv\Scripts\activate
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is unavailable, install manually:
```bash
pip install mediapipe opencv-python torch numpy
```

---

## Training the Model

The dataset uses `.npy` landmark files, organized one folder per class.

Run the training script:
```bash
python -m scripts.train_static --landmark_dir dataset/landmarks
```

The trained model will be saved to:
```
models/static_mlp.pt
```

---

## Running the App

### Laptop Webcam (Default)

```bash
python -m scripts.infer_realtime
```

This opens your default webcam, detects hand landmarks, and displays the predicted sign in real time.

---

## Using Mobile Camera (DroidCam)

You can use your Android phone as a webcam using **DroidCam**.

### Step 1 — Install DroidCam

- **PC Client:** https://www.dev47apps.com/droidcam/windows/
- **Android App:** Available on the Google Play Store

### Step 2 — Connect Phone and Laptop

- Ensure both devices are on the **same Wi-Fi network**
- Open the DroidCam app on your phone to see your **Device IP** and **Port** (e.g., `192.168.1.6:4747`)

### Step 3 — Start DroidCam on PC

- Open **DroidCam Client**
- Enter the phone IP and click **Start**

### Step 4 — Run Inference with DroidCam

```bash
python -m scripts.infer_realtime --cam_url http://<PHONE_IP>:4747/video
```

**Example:**
```bash
python -m scripts.infer_realtime --cam_url http://192.168.1.6:4747/video
```

---

## Deployment Options

### Option 1 — Desktop Executable

Package the app into a standalone `.exe` using PyInstaller:
```bash
pip install pyinstaller
pyinstaller --onefile scripts/infer_realtime.py
```

Output will be at:
```
dist/infer_realtime.exe
```

### Option 2 — Local Web App

Wrap the inference script in a **Flask** backend to serve a browser-based demo.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'scripts'` | Always run commands using `-m` from the project root |
| Camera failed to open | Ensure DroidCam is running; verify the IP and test the URL in a browser |
| `Python not found` | Disable App Execution Aliases in Windows settings and add Python to PATH |

---

## Authors

Developed by **Riddhi** and **Shorya** for academic and research purposes.

---

> 💡 *If you encounter any issues, double-check your environment setup and ensure commands are run from the project root directory.*