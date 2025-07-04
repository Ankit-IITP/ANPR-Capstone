
# 🔍 Automatic Number Plate Recognition System (ANPR-Capstone)

This is a real-time Automatic Number Plate Recognition (ANPR) system developed as a capstone project by Group 83 at IIT Patna. The system detects license plates from both uploaded images and live camera feeds, using deep learning for detection and OCR for character recognition. It includes a clean web interface built with Flask and supports multiple regional and international plate formats.

## 🚀 Features

- 📷 Upload image or use live webcam for detection
- ✅ Real-time license plate detection using SSD MobileNet
- 🔡 OCR-powered text extraction using EasyOCR
- 🌐 Clean, responsive web UI with Flask and HTML/CSS
- 🇮🇳 Supports Indian plates and basic international plates
- 💻 Works on CPU (no GPU required)

## 🧰 Technologies Used

- Python 3
- Flask
- TensorFlow (SSD MobileNet)
- OpenCV
- EasyOCR
- HTML5 + CSS3
- Jinja2 (for templating)

## 📂 Folder Structure

```
ANPR-Capstone/
├── app.py               # Flask application
├── anpr_model.py        # Detection + OCR logic
├── templates/           # HTML files
│   ├── base.html
│   ├── home.html
│   ├── index.html
│   ├── about.html
│   ├── explore.html
│   └── contact.html
├── static/
│   └── uploads/         # Stores uploaded/test images
├── requirements.txt     # Python dependencies
└── README.md
```

## 🖼️ Demo Preview

*(Insert detection screenshots or UI collage image here)*

## 🛠️ How to Run Locally

1. **Clone the repo:**
```bash
git clone https://github.com/Ankit-IITP/ANPR-Capstone.git
cd ANPR-Capstone
```

2. **Create virtual environment (optional but recommended):**
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. **Install requirements:**
```bash
pip install -r requirements.txt
```

4. **Run the app:**
```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## 📌 Objectives

- Automate vehicle license plate detection
- Support both static images and live camera feeds
- Handle OCR under varied lighting and plate formats
- Deliver a clean and professional frontend experience

## ✨ Team Members (Group 83 - IIT Patna)

- Ankit Kumar Singh
- Aman Kumar Singh
- Ayush Kumar Singh
- Bablu Kumar Singh
- Ayush Kumar Singh

## 📄 License

This project is for educational purposes only.
