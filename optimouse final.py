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
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

# ================= SETTINGS =================
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
IRIS_CENTER = 468


class OptiMouse:
    def __init__(self, model_path):
        self.program_running = True
        self.smooth_factor = 0.12
        self.prev_x, self.prev_y = SCREEN_W // 2, SCREEN_H // 2
        self.last_blink_time = 0

        # Audio
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()

        # MediaPipe
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            output_face_blendshapes=True,
            num_faces=1
        )
        self.detector = FaceLandmarker.create_from_options(options)

        # Camera
        print("📸 Booting Camera...")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            raise Exception("Camera Access Denied.")

    def speak(self, text):
        print(f"🔊: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def get_ear(self, landmarks, eye_indices):
        try:
            p = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]

            if len(p) != 6:
                return 0.3

            v1 = np.linalg.norm(p[1] - p[5])
            v2 = np.linalg.norm(p[2] - p[4])
            h = np.linalg.norm(p[0] - p[3])

            if h == 0:
                return 0.3

            return (v1 + v2) / (2.0 * h)

        except:
            return 0.3

    def voice_loop(self):
        try:
            with sr.Microphone() as source:
                print("🎤 Calibrating...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("✅ Voice Assistant Active")

                while self.program_running:
                    try:
                        audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=4)
                        cmd = self.recognizer.recognize_google(audio).lower()
                        print(f"🎤 Heard: {cmd}")

                        if "youtube" in cmd:
                            webbrowser.open("https://www.youtube.com")
                        elif "google" in cmd:
                            webbrowser.open("https://www.google.com")
                        elif "gmail" in cmd:
                            webbrowser.open("https://mail.google.com")
                        elif "search for" in cmd:
                            query = cmd.split("search for")[-1].strip()
                            webbrowser.open(f"https://www.google.com/search?q={query}")
                        elif "click" in cmd:
                            pyautogui.click()
                        elif "scroll up" in cmd:
                            pyautogui.scroll(600)
                        elif "scroll down" in cmd:
                            pyautogui.scroll(-600)
                        elif "exit" in cmd or "stop" in cmd:
                            self.speak("Shutting down")
                            self.program_running = False
                            break

                    except sr.UnknownValueError:
                        continue
                    except Exception as e:
                        print("Voice Error:", e)
        except:
            print("⚠️ PyAudio not working")

    def run(self):
        threading.Thread(target=self.voice_loop, daemon=True).start()

        print("🚀 OptiMouse Started (Press Q to quit)")
        self.speak("System online")

        while self.cap.isOpened() and self.program_running:
            success, frame = self.cap.read()
            if not success or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp = int(time.time() * 1000)
            result = self.detector.detect_for_video(mp_image, timestamp)

            if result.face_landmarks:
                face = result.face_landmarks[0]

                # Cursor movement
                iris = face[IRIS_CENTER]
                tx, ty = iris.x * SCREEN_W, iris.y * SCREEN_H

                cx = (self.smooth_factor * tx) + ((1 - self.smooth_factor) * self.prev_x)
                cy = (self.smooth_factor * ty) + ((1 - self.smooth_factor) * self.prev_y)

                pyautogui.moveTo(int(cx), int(cy))
                self.prev_x, self.prev_y = cx, cy

                # Blink detection
                l_ear = self.get_ear(face, LEFT_EYE)
                r_ear = self.get_ear(face, RIGHT_EYE)
                avg_ear = (l_ear + r_ear) / 2.0

                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 40), 1, 1.5, (255, 255, 0), 2)

                if avg_ear < 0.21:
                    current_time = time.time()
                    if current_time - self.last_blink_time > 0.8:
                        pyautogui.click()
                        self.last_blink_time = current_time
                        cv2.putText(frame, "CLICK", (10, 80), 1, 2, (0, 255, 0), 3)

                # Draw iris
                h, w, _ = frame.shape
                cv2.circle(frame, (int(iris.x * w), int(iris.y * h)), 5, (0, 0, 255), -1)

            cv2.imshow("OptiMouse", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "face_landmarker.task"

    if not os.path.exists(model_path):
        print("❌ Model file not found!")
    else:
        app = OptiMouse(model_path)
        app.run()