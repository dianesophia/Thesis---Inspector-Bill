import cv2
import time
import pyttsx3
import threading
import queue  # ✅ Added queue to manage speech requests
from ultralytics import YOLO
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt

# ✅ Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# ✅ Create a queue for handling speech requests
speech_queue = queue.Queue()

# ✅ Function to handle text-to-speech in a single thread
def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break  # Stop thread if None is received
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# ✅ Start speech thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# Load YOLO model
#model = YOLO('oldMoney100.pt')
model = YOLO('nano40.pt')

#classNames = ["One Hundred", "One Thousand", "Twenty", "Two Hundred", "Fifty", "Five Hundred"]
classNames = ['Real Fifty', 'Real Five Hundred', 'Real One Hundred',
              'Real One Thousand', 'Real Twenty', 'Real Two Hundred']

# Set up webcam
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 640)

prev_frame_time = 0

class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection with YOLO")
        self.setGeometry(100, 100, 800, 700)

        # Layout
        layout = QVBoxLayout()

        # Video feed label (centered)
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Detected objects label (centered)
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
            print("Failed to capture frame from webcam.")
            return

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        self.setWindowTitle(f"Inspector Bill - FPS: {fps:.2f}")

        detected_objects = []

        # Run YOLOv8 detection
        results = model(img)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])
                label = f"{classNames[cls]} {conf:.2f}"

                detected_objects.append(f"{classNames[cls]} ({conf:.2f})")

                # Draw bounding box and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # ✅ Add speech request to queue instead of creating multiple threads
                speech_queue.put(f"Detected {classNames[cls]}")

        # Update detected objects label
        self.detection_label.setText("Detected Objects: " + ", ".join(detected_objects))

        # Convert frame to Qt image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))


app = QApplication([])
window = ObjectDetectionApp()
window.show()
app.exec()

# ✅ Stop the speech thread when closing the app
speech_queue.put(None)
speech_thread.join()

cap.release()
cv2.destroyAllWindows()
