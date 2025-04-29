import cv2
import time
import pyttsx3
import threading
import numpy as np
from ultralytics import YOLO
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem
from PyQt6.QtGui import QImage, QPixmap, QColor
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg  # For real-time graphing

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Load YOLO model
model = YOLO('oldMoney100.pt')

# Define class names based on dataset YAML order
classNames = ["-", "100 pesos", "1000 pesos", "20 pesos", "200 pesos", "50 pesos", "500 pesos", "not money"]

# Define denomination colors (based on actual peso bill colors)
denomination_colors = {
    "100 pesos": "#005BAC",  # Blue
    "1000 pesos": "#0071BB",  # Deep Blue
    "20 pesos": "#FF671F",  # Orange
    "200 pesos": "#128807",  # Green
    "50 pesos": "#B81140",  # Red
    "500 pesos": "#F2A900"  # Yellow
}

# Set up webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)

prev_frame_time = 0

# Dictionary to store detected bill counts
bill_counts = {denom: 0 for denom in denomination_colors.keys()}


# Function to handle text-to-speech
def speak_detection(label):
    engine.say(label)
    engine.runAndWait()


class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Peso Bill Detector - AI Vision")
        self.setGeometry(100, 100, 900, 750)

        # Layout
        layout = QVBoxLayout()

        # Video feed label
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # FPS Display
        self.fps_label = QLabel("FPS: 0.00 | Latency: 0.00s")
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.fps_label)

        # **Denomination Table**
        self.table = QTableWidget(6, 3)  # 6 rows, 3 columns
        self.table.setHorizontalHeaderLabels(["Denomination", "Count", "Total Value"])
        layout.addWidget(QLabel("💰 Detected Peso Bills Summary"), alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.table)

        # Set denomination names and colors in the first column
        for i, denom in enumerate(["20 pesos", "50 pesos", "100 pesos", "200 pesos", "500 pesos", "1000 pesos"]):
            self.table.setItem(i, 0, QTableWidgetItem(denom))
            self.table.item(i, 0).setBackground(QColor(denomination_colors[denom]))

        # **Graph for real-time bill tracking**
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.setTitle("Real-Time Peso Bill Detection", color="black", size="12pt")
        self.plot_widget.setLabel("left", "Count")
        self.plot_widget.setLabel("bottom", "Denominations")
        self.plot_widget.showGrid(x=True, y=True)
        layout.addWidget(self.plot_widget)

        # Bar chart data
        self.bar_chart = pg.BarGraphItem(x=np.arange(6), height=[0]*6, width=0.5, brush="b")
        self.plot_widget.addItem(self.bar_chart)

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
        latency = new_frame_time - prev_frame_time
        prev_frame_time = new_frame_time
        self.fps_label.setText(f"FPS: {fps:.2f} | Latency: {latency:.3f}s")

        detected_objects = []
        results = model(img)

        # Reset bill counts before processing
        global bill_counts
        bill_counts = {denom: 0 for denom in denomination_colors.keys()}

        # Process detected objects
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])

                if cls >= len(classNames) or classNames[cls] == "-":
                    continue  # Skip invalid detections

                label = f"{classNames[cls]} ({conf:.2f})"
                detected_objects.append(label)

                # Update bill count
                bill_counts[classNames[cls]] += 1

                # Draw bounding box
                color = (255, 0, 255)  # Default color
                if classNames[cls] in denomination_colors:
                    hex_color = denomination_colors[classNames[cls]]
                    color = tuple(int(hex_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Start text-to-speech in a separate thread
                threading.Thread(target=speak_detection, args=(f"Detected {classNames[cls]}",)).start()

        # Update table with counts and total values
        for i, denom in enumerate(["20 pesos", "50 pesos", "100 pesos", "200 pesos", "500 pesos", "1000 pesos"]):
            count = bill_counts[denom]
            total_value = count * int(denom.split()[0])  # Extract numerical value

            self.table.setItem(i, 1, QTableWidgetItem(str(count)))
            self.table.setItem(i, 2, QTableWidgetItem(f"₱{total_value}"))

            # Apply color background
            self.table.item(i, 1).setBackground(QColor(denomination_colors[denom]))
            self.table.item(i, 2).setBackground(QColor(denomination_colors[denom]))

        # Update real-time bar chart
        self.bar_chart.setOpts(height=[bill_counts[denom] for denom in ["20 pesos", "50 pesos", "100 pesos", "200 pesos", "500 pesos", "1000 pesos"]])

        # Convert frame to Qt image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qt_img = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.strides[0], QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))


app = QApplication([])
window = ObjectDetectionApp()
window.show()
app.exec()

cap.release()
cv2.destroyAllWindows()
