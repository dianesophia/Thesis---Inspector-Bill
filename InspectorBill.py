import cv2
import time
import pyttsx3
from ultralytics import YOLO
import threading
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Load YOLO model

model = YOLO('OriginalDatasetTrains/Train1.pt')

#Original Dataset`
classNames = ['Real Fifty', 'Real Five Hundred', 'Real One Hundred',
              'Real One Thousand', 'Real Twenty', 'Real Two Hundred']
#model = YOLO('nano40.pt')
model = YOLO('oldMoney100.pt')


#classNames = ['Real Fifty', 'Real Five Hundred', 'Real One Hundred','Real One Thousand', 'Real Twenty', 'Real Two Hundred']
classNames = ["One Hundred", "One Thousand", "Twenty", "Two Hundred", "Fifty", "Five Hundred"]
# Set up webcam

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://192.168.160.193:4747/video")

cap.set(3, 640)
cap.set(4, 640)

prev_frame_time = 0


# Handle text-to-speech in a separate thread
def speak_detection(label):
    engine.say(label)
    engine.runAndWait()


class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection with YOLO")
        self.setGeometry(100, 100, 800, 700)  # Adjusted window size

        # Layout
        layout = QVBoxLayout()

        # Video feed label (centered)
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center the image
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Detected objects label (centered)
        self.detection_label = QLabel("Detected Objects: ")
        self.detection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.detection_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)


        # Timer for updating frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)  # Update every 10ms

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

                # Start text-to-speech in a separate thread
                threading.Thread(target=speak_detection, args=(f"Detected {classNames[cls]}",)).start()

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

cap.release()
cv2.destroyAllWindows()
