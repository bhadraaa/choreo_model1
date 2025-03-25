from flask import Flask, request, jsonify
import numpy as np
import tensorflow.lite as tflite
import cv2
from PIL import Image
import io
import mediapipe as mp
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
import os

# Firebase Setup
cred_path = r"C:\Users\bhadr\sec\choreo-flutter-firebase-cred_cert.json"
if os.path.exists(cred_path):  
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
else:
    raise FileNotFoundError(f"❌ Firebase credential file not found at {cred_path}!")

# Firestore reference
db = firestore.client()
app = Flask(__name__)
CORS(app)

# Load MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Load TFLite Model
TFLITE_MODEL = r"C:\\Users\\bhadr\\model choreoscope\\files\\new_model\\gesture_lstm_model.tflite"
LABELS_FILE = r"C:\\Users\\bhadr\\model choreoscope\\files\\new_model\\gesture_labels.txt"

interpreter = tflite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load gesture labels
gesture_labels = {}
with open(LABELS_FILE, "r") as f:
    for line in f:
        idx, label = line.strip().split(",")
        gesture_labels[int(idx)] = label

def extract_landmarks(image):
    """Extract hand landmarks from an image using MediaPipe."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    
    return None

def predict_gesture(image):
    """Predict gesture from image using TFLite model."""
    landmarks = extract_landmarks(image)
    if landmarks is None:
        return None

    X_input = landmarks.reshape((1, 1, 63)).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], X_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output_data)  # Get predicted class index
    confidence = np.max(output_data)  # Get confidence score

    return gesture_labels[predicted_label] if confidence > 0.5 else None  # Return label only if confidence > 50%

@app.route("/", methods=["GET"])
def home():
    return "Flask server is running!"

@app.route('/predict', methods=['POST']) 
def predict():
    """API Endpoint to predict mudra from image."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    label = predict_gesture(image)
    if label is None:
        return jsonify({"error": "No hand detected or low confidence"}), 400

    # Fetch details from Firestore
    print('hello')
    print(label)
    mudra_ref = db.collection('mudra_details').document(label)
    mudra_doc = mudra_ref.get()
    
    if mudra_doc.exists:
        details = mudra_doc.to_dict().get('details', 'No details available.')
        image=mudra_doc.to_dict().get('imageUrl', 'assets/mudras/PATHAKA.jpg')
    else:
        details = 'No details found in Firestore.'

    return jsonify({
        "mudra": label,
        "details": details,
        "imageUrl":image
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
