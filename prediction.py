import cv2
import mediapipe as mp
import numpy as np
import tensorflow.lite as tflite

# File paths
TFLITE_MODEL = r"C:\\Users\bhadr\\model choreoscope\\files\\new_model\\gesture_lstm_model.tflite"
LABELS_FILE = r"C:\\Users\bhadr\\model choreoscope\\files\\new_model\\gesture_labels.txt"

# Load gesture labels
gesture_labels = {}
with open(LABELS_FILE, "r") as f:
    for line in f:
        idx, label = line.strip().split(",")
        gesture_labels[int(idx)] = label  # Map index to gesture name

# Load TFLite model
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def extract_landmarks(image_path):
    """Extract hand landmarks from an image using MediaPipe."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image with MediaPipe
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    
    print("No hand detected in image.")
    return None

def predict_gesture(image_path):
    """Predict gesture from image using TFLite model."""
    landmarks = extract_landmarks(image_path)
    if landmarks is None:
        return None

    # Reshape input for LSTM (batch_size=1, time_steps=1, features=63)
    X_input = landmarks.reshape((1, 1, 63)).astype(np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], X_input)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output_data)  # Get predicted class index
    confidence = np.max(output_data)  # Get confidence score

    print(f"Predicted Gesture: {gesture_labels[predicted_label]} (Confidence: {confidence:.2f})")
    return gesture_labels[predicted_label]

# Test with an image
TEST_IMAGE = r"C:\Users\bhadr\model choreoscope\files\new_model\etho.jpg"  # Change this path
predict_gesture(TEST_IMAGE)
