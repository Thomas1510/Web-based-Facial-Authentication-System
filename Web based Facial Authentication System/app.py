from flask import Flask, render_template, request, redirect, url_for, session
import cv2
import os
import numpy as np
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'super_secret_key'

# Define the paths
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
users = {}  # Store usernames and face IDs
next_id = 1  # Start assigning face IDs from 1

# Ensure the user_data directory exists
if not os.path.exists("user_data"):
    os.mkdir('user_data')

# Index Page
@app.route('/')
def index():
    return render_template('index.html')

# Create Account Page
@app.route('/create_account', methods=['GET', 'POST'])
def create_account():
    global next_id
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        face_image_data = request.form['face_image']

        # Check if username already exists
        if username in users:
            return "Username already exists. Please choose another one."

        # Decode the image data
        face_id = next_id  # Assign a new face ID for this user
        next_id += 1  # Increment the ID for the next user

        users[username] = {'password': password, 'face_id': face_id}

        # Decode the Base64 image and save it
        img_data = base64.b64decode(face_image_data.split(',')[1])
        image = Image.open(BytesIO(img_data)).convert('L')
        image.save(f"user_data/face.{face_id}.1.jpg")

        # Train the model with the new data
        train_model()
        return redirect(url_for('login'))

    return render_template('create_account.html')

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        face_image_data = request.form['face_image']

        if authenticate_credentials(username, password):  # Check password
            if authenticate_face(username, face_image_data):  # Check face
                session['username'] = username  # Log the user in
                return redirect(url_for('main'))
            else:
                return "Face authentication failed. Try again."
        else:
            return "Invalid credentials."

    return render_template('login.html')

# Check if username and password match
def authenticate_credentials(username, password):
    return username in users and users[username]['password'] == password

# Authenticate the face of the user
def authenticate_face(username, face_image_data):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trained_data.yml')

    # Decode the Base64 image
    img_data = base64.b64decode(face_image_data.split(',')[1])
    image = Image.open(BytesIO(img_data)).convert('L')
    img_arr = np.array(image, 'uint8')

    faces = face_cascade.detectMultiScale(img_arr, 1.3, 5)

    for (x, y, w, h) in faces:
        face_id, confidence = recognizer.predict(img_arr[y:y+h, x:x+w])

        # Check if the recognized face ID matches the user's stored face ID
        if confidence < 100 and face_id == users[username]['face_id']:
            print(f"Face authenticated with confidence: {confidence}")
            return True
        else:
            print(f"Failed to authenticate face: {confidence}")
            return False

    return False

# Train the face recognizer model
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = [], []
    for imagePath in os.listdir('user_data'):
        gray_image = Image.open(os.path.join('user_data', imagePath)).convert('L')
        img_arr = np.array(gray_image, 'uint8')
        face_id = int(os.path.split(imagePath)[-1].split('.')[1])

        faces.append(img_arr)
        ids.append(face_id)

    # Train the recognizer
    recognizer.train(faces, np.array(ids))
    recognizer.write('trained_data.yml')
    print("Model trained successfully!")

# Main Page (after login)
@app.route('/main')
def main():
    if 'username' in session:
        username = session['username']
        return render_template('mains.html', username=username)
    return redirect(url_for('login'))

# Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
