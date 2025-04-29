import cv2
import time
import pyttsx3
import threading
import streamlit as st

import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

def speak_detection(label):
    engine.say(label)
    engine.runAndWait()

# Load YOLO model
model = YOLO('oldMoney100.pt')

# Class names and their respective peso values
classNames = {
    "One Hundred": 100,
    "One Thousand": 1000,
    "Twenty": 20,
    "Two Hundred": 200,
    "Fifty": 50,
    "Five Hundred": 500
}

detected_counts = {key: 0 for key in classNames}

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.prev_frame_time = 0

    def transform(self, frame):
        global detected_counts
        img = frame.to_ndarray(format="bgr24")
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - self.prev_frame_time)
        self.prev_frame_time = new_frame_time

        results = model(img)
        detected_objects = []
        overall_total = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])
                label = list(classNames.keys())[cls]
                detected_objects.append(f"{label} ({conf:.2f})")
                detected_counts[label] += 1

                # Draw bounding box and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Speak detected object
                threading.Thread(target=speak_detection, args=(f"Detected {label}",)).start()

        return img

# Streamlit UI
st.title("Inspector Bill - Cash Denomination Detection System")
st.write("Using YOLO for real-time cash recognition with voice feedback.")

# Display webcam
#webrtc_streamer(key="object-detection", video_transformer_factory=VideoTransformer)
webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": True},  # Ensures video is enabled
    video_html_attrs={"style": "width: 100vw; height: 80vh; object-fit: cover;"}  # Full width, large height
)


#webrtc_streamer(key="object-detection", video_processor_factory=VideoTransformer)


# Table for detected denominations
st.write("### Detected Cash Denominations")

cols = st.columns(3)
cols[0].write("Denomination")
cols[1].write("Count")
cols[2].write("Total Amount")

overall_total = 0
for denomination, value in classNames.items():
    count = detected_counts[denomination]
    total = count * value
    overall_total += total
    cols[0].write(denomination)
    cols[1].write(str(count))
    cols[2].write(str(total))

st.write("### Overall Total: `₱{}`".format(overall_total))
