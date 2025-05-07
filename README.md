# 💵 Inspector Bill

A Machine Learning-powered model that detects and identifies **currency denominations** in real-time using your **webcam**, with **audio feedback** for accessibility. Built with **Python**, **YOLOv8**, **OpenCV**, and **PyQt6**.

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV** – for video frame capture and image processing  
- **YOLOv8** – for real-time currency detection  
- **PyQt6** – for GUI development  
- **pyttsx3** – for offline text-to-speech feedback  
- **Threading & Queue** – to optimize speech handling

---

## 🚀 How It Works

1. Launch the application.
2. The webcam starts capturing video.
3. The YOLOv8 model detects and classifies currency denominations.
4. Detected denominations are:
   - Displayed with bounding boxes on the GUI.
   - Announced via text-to-speech.
5. Repetitions are avoided with a detection cache and time delay.

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/inspector-bill.git
cd inspector-bill
```

### 2. (Optional) Create a Virtual Environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```
opencv-python
pyttsx3
pyqt6
ultralytics
```

### 4. Add Required Files

–- Trained YOLOv8 model file --
* Train1.pt
* Train2.pt
* Train3.pt
* Train4.pt
* Train5.pt
* Train6.pt
* Train7.pt
* Train8.pt
* Train9.pt
* Train10.pt 
  
- `icon.png` application icon

---

## 🧪 Running the Application

```bash
python InspectorBill.py
```

> ⚠️ If your webcam does not work, try changing this line:
```python
cap = cv2.VideoCapture(1)
```
to:
```python
cap = cv2.VideoCapture(0)
```

📱 Using a phone as a webcam (e.g., with Iriun Webcam)?
> Make sure the Iriun app is running on your phone and desktop. Then try different camera indices:
```python
cap = cv2.VideoCapture(2)  # or 3, depending on your system
```

---

## 🗣️ Detected Classes

The app is currently trained to recognize:

- One Hundred
- Two Hundred
- Five Hundred
- One Thousand
- Twenty
- Fifty
- Damage Bill
