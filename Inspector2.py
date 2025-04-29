import cv2
import time
import pyttsx3
import threading
from ultralytics import YOLO
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtCore import QTimer, Qt

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Load YOLO model
model = YOLO('YoloScratchTrains/yoloExtention.pt')
#model = YOLO('OriginalDatasetTrains/Train1.pt')

# New Datasets
classNames = ["Unknown", "One Hundred", "One Thousand", "Twenty", "Two Hundred", "Fifty", "Five Hundred"]

#old dataset
#classNames = ['Real Fifty', 'Real Five Hundred', 'Real One Hundred','Real One Thousand', 'Real Twenty', 'Real Two Hundred']

# Webcam Setup
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 640)

# Time variables for FPS calculation
prev_frame_time = 0
detected_cache = set()  # Cache to prevent redundant speech
last_speech_time = 0  # Track last speech time to avoid repetition

# Thread-safe function for text-to-speech
def speak_detection(label):
    global last_speech_time
    current_time = time.time()
    if label not in detected_cache or (current_time - last_speech_time) > 3:
        detected_cache.add(label)
        last_speech_time = current_time
        engine.say(label)
        engine.runAndWait()

class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection with YOLO")
        self.setGeometry(100, 100, 800, 700)

        self.setWindowIcon(QIcon("icon.png"))

        # Layout
        layout = QVBoxLayout()

        # Video feed label (centered)
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Detected objects label
        self.detection_label = QLabel("Detected Objects: ")
        self.detection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.detection_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)

        # Timer for updating frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

    def update_frame(self):
        global prev_frame_time
        success, img = cap.read()
        if not success:
            print("Failed to capture frame.")
            return

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time else 0
        prev_frame_time = new_frame_time
        self.setWindowTitle(f"Inspector Bill - FPS: {fps:.2f}")

        detected_objects = []

        # Resize image for faster inference
        resized_img = cv2.resize(img, (320, 320))  # Downscaling improves speed

        # Run YOLO detection with **higher confidence and IoU thresholds**
        results = model(resized_img, conf=0.70, iou=0.6)  # Higher confidence, higher IoU

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])

                # **Stricter filtering for precision**
                if conf < 0.70 or cls >= len(classNames) or cls == 0:
                    continue  # Ignore weak detections and "Unknown" class

                label = f"{classNames[cls]} {conf:.2f}"
                detected_objects.append(f"{classNames[cls]} ({conf:.2f})")

                # Rescale coordinates back to original image
                x1 = int(x1 * img.shape[1] / 320)
                y1 = int(y1 * img.shape[0] / 320)
                x2 = int(x2 * img.shape[1] / 320)
                y2 = int(y2 * img.shape[0] / 320)

                # Draw bounding box and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Speak only new detections (avoid repetition)
                threading.Thread(target=speak_detection, args=(f"Detected {classNames[cls]}",)).start()

        # Update detected objects label
        self.detection_label.setText("Detected Objects: " + ", ".join(detected_objects) if detected_objects else "No objects detected")

        # Convert frame to Qt image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qt_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

# Run Application
app = QApplication([])
window = ObjectDetectionApp()
window.show()
app.exec()



cap.release()
cv2.destroyAllWindows()
