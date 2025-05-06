🛠️ Tech StackPython
*  OpenCV (for image processing)
*  YOLOv8 (for currency detection)
*  PyQt6 (for GUI)
*  pyttsx3 (for text-to-speech conversion)
* Threading & Queue (for optimized speech processing)

🚀 How It WorksLaunch the application.
1. The webcam starts capturing video.
2. The YOLO model detects and identifies currency denominations.
3. The detected objects are displayed on-screen with bounding boxes.
4. The text-to-speech engine announces the detected denomination.


# 💵 Inspector Bill

An AI-powered desktop application that detects and identifies **currency denominations** in real-time using your **webcam**, with **audio feedback** for accessibility. Built with **Python**, **YOLOv8**, **OpenCV**, and **PyQt6**.

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
