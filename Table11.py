import cv2
import time
import pyttsx3
from ultralytics import YOLO
import threading
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem
from PyQt6.QtGui import QImage, QPixmap, QColor, QBrush
from PyQt6.QtCore import QTimer, Qt

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Load YOLO model
model = YOLO('oldMoney100.pt')

# Class names based on dataset
classNames = ["-", "100 pesos", "1000 pesos", "20 pesos", "200 pesos", "50 pesos", "500 pesos", "not money"]

# Philippine peso bill colors
bill_colors = {
    "20 pesos": "#FFA500",  # Orange
    "50 pesos": "#FF0000",  # Red
    "100 pesos": "#800080",  # Violet
    "200 pesos": "#008000",  # Green
    "500 pesos": "#FFD700",  # Yellow
    "1000 pesos": "#0000FF"  # Blue
}

# Initialize denomination count
bill_count = {key: 0 for key in bill_colors.keys()}

# Set up webcam
cap = cv2.VideoCapture(0)
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
        self.setWindowTitle("Peso Bill Detection")
        self.setGeometry(100, 100, 800, 750)

        # Layout
        layout = QVBoxLayout()

        # Video feed label
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Title for the table
        self.table_title = QLabel("Detected Denominations")
        self.table_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.table_title)

        # Table for bill count
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
        """Initializes the table with denomination names and applies background colors."""
        for row, (bill, color) in enumerate(bill_colors.items()):
            # Denomination Column
            item = QTableWidgetItem(bill)
            item.setBackground(QBrush(QColor(color)))  # Set background color
            item.setForeground(QBrush(QColor(Qt.GlobalColor.black)))  # Set text color
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, item)

            # Count Column
            count_item = QTableWidgetItem("0")
            count_item.setBackground(QBrush(QColor(color)))  # Apply background color
            count_item.setForeground(QBrush(QColor(Qt.GlobalColor.black)))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, count_item)

            # Total Amount Column
            total_item = QTableWidgetItem("0")
            total_item.setBackground(QBrush(QColor(color)))  # Apply background color
            total_item.setForeground(QBrush(QColor(Qt.GlobalColor.black)))
            total_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, total_item)

    def update_frame(self):
        global prev_frame_time, bill_count
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

        # Reset bill count for each frame
        bill_count = {key: 0 for key in bill_colors.keys()}

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])

                if cls in range(1, 7):  # Exclude class 0 ('-') and 7 ('not money')
                    label = classNames[cls]
                    bill_count[label] += 1
                    detected_objects.append(f"{label} ({conf:.2f})")

                    # Draw bounding box and label
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Start text-to-speech in a separate thread
                    threading.Thread(target=speak_detection, args=(f"Detected {label}",)).start()

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
