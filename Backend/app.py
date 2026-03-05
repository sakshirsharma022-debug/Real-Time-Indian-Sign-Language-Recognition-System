from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from utils import get_normalized_rays
import mediapipe as mp
import json

app = Flask(__name__)
CORS(app) # This allows React to talk to Flask

# 1. Load Model and Labels
model = tf.keras.models.load_model('isl_model.h5')
with open("label_map.json", "r") as f:
    label_map = json.load(f)
actions = list(label_map.keys())

# 2. Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

@app.route('/predict', methods=['POST'])
def predict():
    # React will send a list of 30 frames (hand landmarks)
    data = request.json
    sequence = data.get('sequence') # Expecting a list of 42-feature vectors
    
    if len(sequence) != 30:
        return jsonify({"error": "Need 30 frames"}), 400

    # Convert to numpy and predict
    res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
    prediction = actions[np.argmax(res)]
    confidence = float(np.max(res))

    return jsonify({
        "prediction": prediction,
        "confidence": confidence
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)