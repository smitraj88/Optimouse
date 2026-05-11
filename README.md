# 🚀 OptiMouse: AI-Powered Gesture & Voice Control System

OptiMouse is a cutting-edge accessibility and productivity tool that allows users to control their computer using facial gestures (eye-tracking) and voice commands. Leveraging MediaPipe for computer vision and SpeechRecognition for vocal interaction, it provides a hands-free computing experience.

---

## ✨ Key Features

### 👁️ Intelligent Gaze Tracking
- **Eye-Tracking Cursor:** Move your mouse cursor just by moving your eyes. The system maps your iris center to screen coordinates.
- **Precision Smoothing:** Uses Exponential Moving Average (EMA) to ensure fluid cursor movement without jitter.
- **Blink-to-Click:** Perform mouse clicks by blinking. The system calculates the Eye Aspect Ratio (EAR) for accurate detection.

### 🎙️ Advanced Voice Assistant
- **Navigation:** Open YouTube, Google, or Gmail instantly.
- **Search:** Say "Search for [query]" to launch a Google search.
- **Application Control:** Open Notepad, Calculator, Command Prompt, or VS Code.
- **System Utilities:** Control volume (Up/Down/Mute), scroll pages, and perform clicks via voice.
- **Calibration:** Automatically adjusts for background noise to improve recognition accuracy.

### 🖥️ Real-time Feedback
- **Overlay HUD:** Visual indicators for EAR values and click detections.
- **Iris Tracking:** Visual marker on the camera feed to verify tracking accuracy.

---

## 🛠️ Tech Stack

- **Python 3.11+**
- **MediaPipe:** For high-performance face landmarker and iris tracking.
- **OpenCV:** For camera feed processing and visual feedback.
- **PyAutoGUI:** For system-level mouse and keyboard control.
- **SpeechRecognition:** For processing vocal commands.
- **PyTTSx3:** For text-to-speech system feedback.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python installed. It is recommended to use a virtual environment.

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/smitraj88/Optimouse.git
cd Optimouse
pip install -r requriment.txt
```

### 3. Setup Model
The project requires the `face_landmarker.task` file. Ensure it is placed in the same directory as `final.py`.

### 4. Running the Project
```bash
python final.py
```

---

## 🎮 How to Use
- **Mouse Movement:** Look around the screen to move the cursor.
- **Left Click:** Blink both eyes firmly or say "Click".
- **Voice Commands:** 
  - "Open YouTube"
  - "Search for [topic]"
  - "Scroll down"
  - "Volume up"
  - "Exit" (to stop the program)
- **Emergency Stop:** Press `Q` on your keyboard while the camera window is focused.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

## 📄 License
This project is open source. Feel free to use and modify it for your own needs.
