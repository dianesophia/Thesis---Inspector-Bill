import cv2
import time
import pyttsx3
from ultralytics import YOLO
import threading
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt6.QtGui import QImage, QPixmap, QColor
from PyQt6.QtCore import QTimer, Qt

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Load YOLO model
#model = YOLO('oldMoney100.pt')
model = YOLO('oldMoney100.pt')

# Class names and their respective peso values and colors
classNames = {
    "One Hundred": (100, "#005BAC"),
    "One Thousand": (1000, "#004B87"),
    "Twenty": (20, "#E87722"),
    "Two Hundred": (200, "#00843D"),
    "Fifty": (50, "#C8102E"),
    "Five Hundred": (500, "#FFD700")
}

detected_counts = {key: 0 for key in classNames}
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)
prev_frame_time = 0

def speak_detection(label):
    engine.say(label)
    engine.runAndWait()

class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection with YOLO")
        self.setGeometry(100, 100, 800, 800)

        # Layout
        layout = QVBoxLayout()

        # Video feed label
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Table for detected denominations
        self.table = QTableWidget()
        self.table.setRowCount(len(classNames))
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Denomination", "Count", "Total Amount"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Initialize table with denominations and colors
        for row, (denomination, (value, color)) in enumerate(classNames.items()):
            item = QTableWidgetItem(denomination)
            item.setBackground(QColor(color))
            item.setForeground(QColor("white"))
            self.table.setItem(row, 0, item)
            self.table.setItem(row, 1, QTableWidgetItem("0"))  # Count
            self.table.setItem(row, 2, QTableWidgetItem("0"))  # Total

        layout.addWidget(self.table)
        self.setLayout(layout)

        # Timer for updating frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

    def update_frame(self):
        global prev_frame_time, detected_counts
        success, img = cap.read()
        if not success:
            return

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        self.setWindowTitle(f"Inspector Bill - FPS: {fps:.2f}")

        results = model(img)
        detected_objects = []

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

        # Update table
        for row, (denomination, (value, _)) in enumerate(classNames.items()):
            count = detected_counts[denomination]
            self.table.setItem(row, 1, QTableWidgetItem(str(count)))  # Count
            self.table.setItem(row, 2, QTableWidgetItem(str(count * value)))  # Total

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
