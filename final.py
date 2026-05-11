import cv2
import pyautogui
import numpy as np
import time
import threading
import os
import webbrowser
import pyttsx3
import speech_recognition as sr
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

# ================= SETTINGS =================
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0 
SCREEN_W, SCREEN_H = pyautogui.size()

# Landmark indices for EAR (Eye Aspect Ratio)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
IRIS_CENTER = 468 # Refined iris center landmark

class OptiMouse:
    def __init__(self, model_path):
        self.program_running = True
        self.smooth_factor = 0.15 # Sensitivity: 0.05 (very slow) to 0.4 (fast/jittery)
        self.prev_x, self.prev_y = SCREEN_W // 2, SCREEN_H // 2
        
        # Audio Initialization
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        
        # MediaPipe Tasks Initialization
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            output_face_blendshapes=True,
            num_faces=1
        )
        self.detector = FaceLandmarker.create_from_options(options)
        
        # Camera Initialization
        print("📸 Booting Camera...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Camera Access Denied. Check permissions.")

    def speak(self, text):
        print(f"🔊: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def get_ear(self, landmarks, eye_indices):
        """Calculates Eye Aspect Ratio (EAR) for blink detection."""
        p = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
        # Vertical distances
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        # Horizontal distance
        h = np.linalg.norm(p[0] - p[3])
        return (v1 + v2) / (2.0 * h)

    def voice_loop(self):
        """Dedicated thread for listening to commands."""
        with sr.Microphone() as source:
            print("🎤 Calibrating for background noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
            print("✅ Voice Assistant Active.")
            
            while self.program_running:
                try:
                    audio = self.recognizer.listen(source, phrase_time_limit=4)
                    cmd = self.recognizer.recognize_google(audio).lower()
                    print(f"🎤 Heard: {cmd}")
                    
                    if "youtube" in cmd:
                        self.speak("Opening YouTube")
                        webbrowser.open("https://www.youtube.com")
                    elif "google" in cmd:
                        self.speak("Opening Google")
                        webbrowser.open("https://www.google.com")
                    elif "gmail" in cmd:
                        self.speak("Opening Gmail")
                        webbrowser.open("https://mail.google.com")
                    elif "search for" in cmd:
                        query = cmd.split("search for")[-1].strip()
                        self.speak(f"Searching for {query}")
                        webbrowser.open(f"https://www.google.com/search?q={query}")
                    elif "click" in cmd:
                        pyautogui.click()
                    elif "scroll up" in cmd:
                        pyautogui.scroll(600)
                    elif "scroll down" in cmd:
                        pyautogui.scroll(-600)
                    elif "open notepad" in cmd:
                        os.system("notepad")

                    elif "open calculator" in cmd:
                        os.system("calc")

                    elif "open command prompt" in cmd:
                        os.system("start cmd")

                    elif "open vscode" in cmd:
                        os.system("code")

                    elif "volume up" in cmd:
                        pyautogui.press("volumeup")

                    elif "volume down" in cmd:
                        pyautogui.press("volumedown")

                    elif "mute" in cmd:
                        pyautogui.press("volumemute")
                    elif "stop program" in cmd or "exit" in cmd:
                        self.speak("System shutting down. Goodbye.")
                        self.program_running = False
                        
                except sr.UnknownValueError:
                    continue # Ignore garbled speech
                except Exception as e:
                    print(f"Voice Thread Error: {e}")

    def run(self):
        # Start Voice Module in background
        threading.Thread(target=self.voice_loop, daemon=True).start()
        
        print("🚀 OptiMouse Engine Started. Press 'Q' to Emergency Stop.")
        self.speak("Opti Mouse system online.")

        try:
            while self.cap.isOpened() and self.program_running:
                success, frame = self.cap.read()
                if not success: continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Detect Face & Eyes
                timestamp_ms = int(time.time() * 1000)
                result = self.detector.detect_for_video(mp_image, timestamp_ms)

                if result.face_landmarks:
                    face = result.face_landmarks[0]
                    
                    # 1. Gaze Mapping
                    iris = face[IRIS_CENTER]
                    # Map 0.0-1.0 to Screen Resolution
                    target_x = iris.x * SCREEN_W
                    target_y = iris.y * SCREEN_H
                    
                    # Apply Exponential Moving Average Smoothing
                    curr_x = (self.smooth_factor * target_x) + ((1 - self.smooth_factor) * self.prev_x)
                    curr_y = (self.smooth_factor * target_y) + ((1 - self.smooth_factor) * self.prev_y)
                    
                    pyautogui.moveTo(int(curr_x), int(curr_y))
                    self.prev_x, self.prev_y = curr_x, curr_y

                    # 2. Blink Detection (using EAR formula)
                    l_ear = self.get_ear(face, LEFT_EYE)
                    r_ear = self.get_ear(face, RIGHT_EYE)
                    avg_ear = (l_ear + r_ear) / 2.0

                    # 3. Visual Feedback Overlay
                    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 40), 1, 1.5, (255, 255, 0), 2)
                    
                    # Blink threshold (typically 0.20 to 0.22)
                    if avg_ear < 0.21:
                        pyautogui.click()
                        cv2.putText(frame, "CLICK DETECTED", (10, 90), 1, 2, (0, 255, 0), 3)
                        time.sleep(0.2) # Debounce to prevent multiple clicks

                    # Draw iris marker for tracking verification
                    h, w, _ = frame.shape
                    cv2.circle(frame, (int(iris.x * w), int(iris.y * h)), 5, (0, 0, 255), -1)

                cv2.imshow('OptiMouse Control Hub', frame)
                
                # Exit with 'Q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.program_running = False
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the task model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(current_dir, 'face_landmarker.task')
    
    if os.path.exists(model_file):
        app = OptiMouse(model_file)
        app.run()
    else:
        print(f"❌ CRITICAL ERROR: '{model_file}' not found.")
        print("Please place the face_landmarker.task file in the same directory.")