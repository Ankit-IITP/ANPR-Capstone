from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os
import cv2
import threading
import time
from anpr_model import detect_plate

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ===== HOME PAGE ROUTE =====
@app.route('/')
def home():
    return render_template('home.html')

# ===== EXPLORE PAGE =====
@app.route('/explore')
def explore():
    return render_template('explore.html')

# ===== ABOUT PAGE =====
@app.route('/about')
def about():
    return render_template('about.html')

# ===== CONTACT PAGE =====
@app.route('/contact')
def contact():
    return render_template('contact.html')


# ===== DETECTION PAGE =====
@app.route("/detections", methods=["GET", "POST"])
def detections():
    result_text = None
    image_url = None

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", results="No image uploaded.")

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", results="Empty file name.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result_text = detect_plate(filepath)
        image_url = filepath

    return render_template("index.html", results=result_text, image_url=image_url)


# ===== LIVE CAMERA DETECTION =====
camera_on = False
capture_thread = None
live_plate_result = None

def capture_from_camera():
    global camera_on, live_plate_result
    cap = cv2.VideoCapture(0)

    while camera_on and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        temp_path = "static/uploads/live_frame.jpg"
        cv2.imwrite(temp_path, frame)

        try:
            plate = detect_plate(temp_path)
        except Exception:
            plate = "Error"

        live_plate_result = plate

        # Annotate
        cv2.putText(frame, f"Detected: {plate}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if plate != "No plate detected" and plate != "Error":
            timestamp = int(time.time())
            save_path = f"static/uploads/detected_{plate}_{timestamp}.jpg"
            cv2.imwrite(save_path, frame)

        cv2.imshow("Live ANPR - Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route("/start_camera", methods=["POST"])
def start_camera():
    global camera_on, capture_thread
    if not camera_on:
        camera_on = True
        capture_thread = threading.Thread(target=capture_from_camera)
        capture_thread.start()
    return redirect("/get_live_result")

@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    global camera_on
    camera_on = False
    return render_template("index.html", results="Camera stopped.")

@app.route("/get_live_result")
def get_live_result():
    global live_plate_result
    return render_template("index.html", live_result=live_plate_result)
if __name__ == '__main__':
    app.run(debug=True)
