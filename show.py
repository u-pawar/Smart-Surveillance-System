import cv2
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import os
from sklearn.model_selection import train_test_split

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

model = keras.models.load_model('model27.h5')
video_paths = ['n.mp4', 'b.mp4', 'r.mp4', 'v.mp4']
for video_path in video_paths:
    test_x = load_preprocess_video(video_path)
    predictions = model.predict(np.expand_dims(test_x, axis=0))
    predicted_class = np.argmax(predictions)
    print("The predicted class for the test video is:",predicted_class)
    label = {0:'Normal', 1:'Burglary', 2:'Robbery', 3:'Shoplifting', 4:'Vandalism'}
    cap = cv2.VideoCapture(video_path)
    while True:
        # Read a frame from the video source
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            break  # Break the loop if the video has ended or an error occurred
        cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', 720,720)
        # Display the frame in a window
        cv2.imshow('Video', frame)
    
        text = label[predicted_class]  # The text you want to display
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (10, 50)  # Position (x, y) where you want to place the text
        font_scale = 1
        font_color = (0, 0, 255)
        if text=='Normal':
            font_color = (0, 255, 0)  # Text color in BGR format (green in this case)
        font_thickness = 2

        cv2.putText(frame, text, position, font, font_scale, font_color, font_thickness)
        cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', 720,720)
        cv2.imshow('Video', frame)
        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

