# 🚗 Driver Drowsiness Detection

A real-time driver drowsiness detection system using **MediaPipe Face Mesh** and computer vision. Monitors eye aspect ratio (EAR) from a video feed and triggers an audio alarm when prolonged eye closure (drowsiness) is detected.

---

## 📌 How It Works

The system uses **MediaPipe's 468-point Face Mesh** to track facial landmarks frame-by-frame.

- **Eye Aspect Ratio (EAR)** — calculated from 6 landmark points per eye (left & right)
- If EAR drops below `0.25` for **10 consecutive frames**, the driver is marked **DROWSY**
- An **audio alarm** (`alarm.wav`) is triggered via `aplay`
- A **live EAR graph** is rendered on the video feed for visual feedback
- **Accuracy metric** is computed as the percentage of alert frames vs total frames

---

## 🛠️ Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Mounish2311/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection
```

### 2. Install dependencies

```bash
pip install opencv-python mediapipe numpy
```

### 3. Add required files

Place the following in the project root:
- `face.mp4` — input video file of the driver
- `alarm.wav` — audio file to play when drowsiness is detected

### 4. Run the project

```bash
python drowsiness.py
```

Press **`q`** to quit the video window.

---

## 📁 Project Structure

```
Driver-Drowsiness-Detection/
│
├── drowsiness.py       # Main detection script
├── alarm.wav           # Alert sound (add manually)
├── face.mp4            # Input video (add manually)
├── .gitignore
└── README.md
```

---

## 🚨 Features

- MediaPipe Face Mesh — no external `.dat` model file needed
- EAR-based drowsiness detection (left + right eye average)
- Blink counter tracking
- Real-time EAR graph plotted on video frame
- Audio alarm via system `aplay` command (Linux)
- Live accuracy display on screen
- Colored status overlay — 🟢 ALERT / 🔴 DROWSY / 🟡 NO FACE

---

## 📐 EAR Formula

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

Where p1–p6 are the 6 eye landmark coordinates. EAR drops significantly when the eye closes.

**Threshold used:** `EAR < 0.25` for `10 consecutive frames` → DROWSY
