import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

app = Flask(__name__)

# Load your machine learning model (replace this with your own model)
model = keras.models.load_model('../model27.h5')

# Define a function to preprocess input images
def load_preprocess_video(video):
  cap = cv2.VideoCapture(video)
  frames = []
  frame_count = 0
  while frame_count < 8:
    ret, frame = cap.read()
    frame_count+=1
    if not ret:
        break

    # Preprocess frame (resize, normalize, etc.)
    frame = cv2.resize(frame, (112, 112))
    frames.append(frame)

    # cv2.imshow("Frame", frames)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

  cap.release()

  return frames

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded file
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Preprocess the image
        test_x = load_preprocess_video(file_path)
        predictions = model.predict(np.expand_dims(test_x, axis=0))
        predicted_class = np.argmax(predictions)
        print("The predicted class for the test video is:",predicted_class)

        # Format the predictions
        #result = [{'label': label, 'probability': float(prob)} for (_, label, prob) in decoded_predictions]
	#result = {'0':{'label': 'Normal'}, '1':{'label': 'Burglary'}, '2':{'label': 'Robbery'}, '3':{'label': 'Shoplifting'} ,'4':{'label': 'Vandelism'}}
        result = {0:'Normal', 1:'Burglary', 2:'Robbery', 3:'Shoplifting', 4:'Vandelism'}
        return result[predicted_class]

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)

