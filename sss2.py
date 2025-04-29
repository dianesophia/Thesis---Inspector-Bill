import cv2
import time
import pyttsx3
import threading
import streamlit as st
import av
import numpy as np
import os
import queue
import torch
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ----------------- Text-to-Speech (TTS) Handling -----------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

speech_queue = queue.Queue()
last_speech_time = 0
speech_cooldown = 2  # 2-second cooldown between speeches

def speak_thread():
    """Continuously speaks text from the queue with a cooldown."""
    global last_speech_time
    while True:
        text = speech_queue.get()
        if text == "STOP":
            break
        current_time = time.time()
        if current_time - last_speech_time >= speech_cooldown:
            engine.say(text)
            engine.runAndWait()
            last_speech_time = current_time

speech_thread = threading.Thread(target=speak_thread, daemon=True)
speech_thread.start()

def speak_detection(label):
    """Queue text for speech with cooldown."""
    speech_queue.put(label)

# ----------------- Load YOLO Model -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(os.getcwd(), "oldMoney100.pt")

if not os.path.exists(model_path):
    st.error("❌ Model file 'oldMoney100.pt' not found. Please check the path!")
    st.stop()

try:
    model = YOLO(model_path).to(device)
    st.success(f"Model loaded on: {device}")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ----------------- Class Names & Detected Counts -----------------
classNames = {
    "One Hundred": 100,
    "One Thousand": 1000,
    "Twenty": 20,
    "Two Hundred": 200,
    "Fifty": 50,
    "Five Hundred": 500
}

if "detected_counts" not in st.session_state:
    st.session_state.detected_counts = {key: 0 for key in classNames}

# ----------------- Video Processor -----------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.prev_frame_time = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Processes frames in real-time for detection and returns video feed."""
        img = frame.to_ndarray(format="bgr24")
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - self.prev_frame_time) if self.prev_frame_time else 0
        self.prev_frame_time = new_frame_time

        # Run YOLO object detection with error handling
        try:
            results = model(img, device=device)
        except Exception as e:
            st.warning(f"Inference failed: {e}. Switching to CPU.")
            try:
                results = model(img, device='cpu')
            except Exception as e2:
                st.error(f"CPU inference failed: {e2}")
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        if results is None or len(results) == 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])

                if cls < len(classNames):  # Ensure class index is within range
                    label = list(classNames.keys())[cls]
                    st.session_state.detected_counts[label] += 1  # Update count

                    # Draw bounding box and label
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(img, f"{label} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Queue speech
                    speak_detection(f"Detected {label}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ----------------- Streamlit UI -----------------
st.title("💵 Inspector Bill - Cash Denomination Detection System")
st.write("Using YOLO for real-time cash recognition with voice feedback.")

# Reset counts button
if st.button("Reset Counts"):
    st.session_state.detected_counts = {key: 0 for key in classNames}
    st.success("Counts reset!")

# Display webcam with WebRTC
webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    video_html_attrs={"style": "width: 100vw; height: 80vh; object-fit: cover;"}
)

# ----------------- Table for Detected Denominations -----------------
st.write("### Detected Cash Denominations")

cols = st.columns(3)
cols[0].write("Denomination")
cols[1].write("Count")
cols[2].write("Total Amount")

overall_total = 0
for denomination, value in classNames.items():
    count = st.session_state.detected_counts[denomination]
    total = count * value
    overall_total += total
    cols[0].write(denomination)
    cols[1].write(str(count))
    cols[2].write(str(total))

st.write("### 💰 Overall Total: `₱{}`".format(overall_total))

# ----------------- Stop Speech Thread on Exit -----------------
if st.button("Stop Speech"):
    speech_queue.put("STOP")
    speech_thread.join()
    st.success("Speech stopped.")