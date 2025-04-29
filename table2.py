import cv2
import time
import pyttsx3
from ultralytics import YOLO
import threading
from collections import defaultdict
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem
from PyQt6.QtGui import QImage, QPixmap, QColor, QBrush
from PyQt6.QtCore import QTimer, Qt

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Load YOLO model
model = YOLO('oldMoney100.pt')

# Correctly indexed class names based on the YOLO dataset
classNames = {
    1: "100 pesos",
    2: "1000 pesos",
    3: "20 pesos",
    4: "200 pesos",
    5: "50 pesos",
    6: "500 pesos",
    7: "not money"
}

# Correct color mapping based on class indexes
bill_colors = {
    "20 pesos": "#FFA500",   # Orange
    "50 pesos": "#FF0000",   # Red
    "100 pesos": "#800080",  # Purple
    "200 pesos": "#008000",  # Green
    "500 pesos": "#FFD700",  # Gold
    "1000 pesos": "#0000FF"  # Blue
}

# Store bill detections for tracking
bill_history = defaultdict(list)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)
prev_frame_time = 0

# Speech function in a separate thread
def speak_detection(label):
    engine.say(label)
    engine.runAndWait()

class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Peso Bill Detection")
        self.setGeometry(100, 100, 800, 750)

        layout = QVBoxLayout()

        # Video Feed
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Table Title
        self.table_title = QLabel("Detected Denominations")
        self.table_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.table_title)

        # Table for tracking denominations
        self.table = QTableWidget()
        self.table.setRowCount(len(bill_colors))
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Denomination", "Count", "Total Amount"])
        self.populate_table()
        layout.addWidget(self.table)

        self.setLayout(layout)

        # Timer for updating frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

    def populate_table(self):
        """Initialize the table with denomination names and colors."""
        for row, (bill, color) in enumerate(bill_colors.items()):
            item = QTableWidgetItem(bill)
            item.setBackground(QBrush(QColor(color)))
            item.setForeground(QBrush(QColor(Qt.GlobalColor.black)))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, item)

            count_item = QTableWidgetItem("0")
            count_item.setBackground(QBrush(QColor(color)))
            count_item.setForeground(QBrush(QColor(Qt.GlobalColor.black)))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, count_item)

            total_item = QTableWidgetItem("0")
            total_item.setBackground(QBrush(QColor(color)))
            total_item.setForeground(QBrush(QColor(Qt.GlobalColor.black)))
            total_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, total_item)

    def update_frame(self):
        global prev_frame_time
        success, img = cap.read()
        if not success:
            print("Failed to capture frame from webcam.")
            return

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        self.setWindowTitle(f"Peso Bill Detection - FPS: {fps:.2f}")

        detected_objects = []
        results = model(img)

        bill_count = {key: 0 for key in bill_colors.keys()}

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])

                if cls in classNames and conf > 0.6:  # Confidence filter
                    label = classNames[cls]  # Map class index to bill name
                    bill_count[label] += 1
                    bill_history[label].append(time.time())  # Store timestamp

                    detected_objects.append(f"{label} ({conf:.2f})")

                    # Draw bounding box and label
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(img, f"{label} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Speak detected bill asynchronously
                    threading.Thread(target=speak_detection, args=(f"Detected {label}",), daemon=True).start()

        # Update table with new counts
        for row, (bill, _) in enumerate(bill_colors.items()):
            self.table.item(row, 1).setText(str(bill_count[bill]))
            self.table.item(row, 2).setText(str(bill_count[bill] * int(bill.split()[0])))

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
