import os
import cv2
import face_recognition
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import datetime


app = Flask(__name__)


# Secret key for session management (required for session functionality)
app.secret_key = os.urandom(24)  # Make sure this is unique and persistent


# Directory to store face images and logs
FACE_IMAGES_DIR = 'static/images/faces'
ATTENDANCE_LOG_DIR = 'static/logs/attendance_logs'


# Create directories if they don't exist
os.makedirs(FACE_IMAGES_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_LOG_DIR, exist_ok=True)


# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    # Log session to check if 'registered_name' exists
    print(f"Session at index: {session}")
    return render_template('index.html')


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/register_face', methods=['POST'])
def register_face():
    try:
        # Get the image data and other form data
        image_data = request.form.get('image')
        name = request.form.get('name')
        surname = request.form.get('surname')


        # Decode the base64 image
        image_data = image_data.split(',')[1]
        img = Image.open(BytesIO(base64.b64decode(image_data)))
        img_path = os.path.join(FACE_IMAGES_DIR, f"{name}_{surname}.jpg")
        img.save(img_path)


        # Add face recognition logic here (you may need to save the face encoding for later use)
        face_image = cv2.imread(img_path)
        face_locations = face_recognition.face_locations(face_image)
        if face_locations:
            # Store face encodings for future recognition
            face_encoding = face_recognition.face_encodings(face_image, face_locations)[0]
            encoding_file_path = os.path.join(FACE_IMAGES_DIR, f"{name}_{surname}.npy")
            np.save(encoding_file_path, face_encoding)


            # Store the registered name in session
            session['registered_name'] = f"{name} {surname}"
            print(f"Registered name saved in session: {session['registered_name']}")  # Debugging log


            return jsonify({"status": "success", "message": "Face registered successfully."})
        else:
            return jsonify({"status": "error", "message": "No face detected."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    try:
        # Decode image
        image_data = request.form.get('image')
        image_data = image_data.split(',')[1]
        img = Image.open(BytesIO(base64.b64decode(image_data)))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


        face_locations = face_recognition.face_locations(img)


        if not face_locations:
            return jsonify({"status": "error", "message": "No face detected for attendance."})


        # Get encoding of detected face
        face_encoding = face_recognition.face_encodings(img, face_locations)[0]


        # Load all known face encodings
        known_encodings = []
        known_names = []


        for file in os.listdir(FACE_IMAGES_DIR):
            if file.endswith(".npy"):
                name_parts = file.replace(".npy", "").split('_')
                person_name = f"{name_parts[0]} {name_parts[1]}"
                encoding = np.load(os.path.join(FACE_IMAGES_DIR, file))
                known_encodings.append(encoding)
                known_names.append(person_name)


        # Calculate face distances
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)


        if len(face_distances) == 0:
            return jsonify({"status": "error", "message": "No known faces to compare."})


        # Get best match
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]


        # Set a confidence threshold (lower is stricter)
        threshold = 0.35


        if best_distance < threshold:
            matched_name = known_names[best_match_index]
            session['registered_name'] = matched_name  # Update session with matched name
            return jsonify({"status": "success", "match": True, "name": matched_name})
        else:
            return jsonify({"status": "error", "message": "Face not recognized."})


    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    try:
        # Retrieve the name of the person from the session
        name = session.get('registered_name')


        if not name:
            return jsonify({"status": "error", "message": "No registered user found."})


        # Log the attendance with timestamp
        log_file_path = os.path.join(ATTENDANCE_LOG_DIR, 'attendance_log.txt')


        # Append the name and timestamp to the log file
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"{name} - {datetime.datetime.now()}\n")


        # Respond with success message
        return jsonify({"status": "success", "message": f"Attendance marked for {name}."})


    except Exception as e:
        # Handle any exceptions and respond with an error message
        return jsonify({"status": "error", "message": str(e)})


@app.route('/view_attendance', methods=['GET'])
def view_attendance():
    try:
        log_file_path = os.path.join(ATTENDANCE_LOG_DIR, 'attendance_log.txt')
       
        # Check if the log file exists
        if not os.path.exists(log_file_path):
            return jsonify({"status": "error", "message": "No attendance logs found."})


        # Read the attendance log
        with open(log_file_path, 'r') as log_file:
            logs = log_file.readlines()


        return jsonify({"status": "success", "logs": logs})


    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
