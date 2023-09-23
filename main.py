import cv2
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import os
from sklearn.model_selection import train_test_split

def create_c3d_model(input_shape, num_classes):
    model = keras.Sequential()

    # 1st Convolutional Layer
    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='conv1', input_shape=input_shape))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='pool1'))

    # 2nd Convolutional Layer
    model.add(layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='conv2'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='pool2'))

    # 3rd Convolutional Layer
    model.add(layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3'))
    model.add(layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv4'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='pool3'))

    # Flatten the feature maps
    model.add(layers.Flatten(name='flatten'))

    # Fully Connected Layers
    model.add(layers.Dense(4096, activation='relu', name='fc1'))
    model.add(layers.Dropout(0.5, name='dropout1'))
    model.add(layers.Dense(4096, activation='relu', name='fc2'))
    model.add(layers.Dropout(0.5, name='dropout2'))

    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))

    return model

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

x = []
y = []
class_labels = {'Normal': [1,0,0,0,0], 'Burglary': [0,1,0,0,0], 'Robbery': [0,0,1,0,0], 'Shoplifting': [0,0,0,1,0], 'Vandalism': [0,0,0,0,1]}
video_dir = 'data'

for class_name in os.listdir(video_dir):
    class_dir = os.path.join(video_dir, class_name)
    for video_file in os.listdir(class_dir):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(class_dir, video_file)
            video_frames = load_preprocess_video(video_path)
            x.append(video_frames)
            y.append(class_labels[class_name])
            
x = np.array(x)
y = np.array(y)

np.save('input_data_2', x)
np.save('input_labels_2', y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(x_train.shape)

# Compile the model
model = create_c3d_model(input_shape = x_train.shape[1:], num_classes = 5)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.summary()

model.fit(np.array(x_train), np.array(y_train), epochs=10)

#model.save('first_model.h5')

model = keras.models.load_model('model27.h5')

test_x = load_preprocess_video('v.mp4')
predictions = model.predict(np.expand_dims(test_x, axis=0))
predicted_class = np.argmax(predictions)
print("The predicted class for the test video is:",predicted_class)

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')
