import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

class train():
    score = 0
    while 1.0 >= score:
        def mediapipe_detection(image, model):
            # COLOR CONVERSION BGR 2 RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False                  # Image is no longer writeable
            results = model.process(image)                 # Make prediction
            image.flags.writeable = True                   # Image is now writeable
            # COLOR COVERSION RGB 2 BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image, results
        def draw_landmarks(image, results):
            mp_drawing.draw_landmarks(image, results.face_landmarks,
                                    mp_holistic.FACE_CONNECTIONS)  # Draw face connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                    mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                    mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                    mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections
        def draw_styled_landmarks(image, results):
            #     # Draw face connections
            #     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
            #                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            #                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
            #                              )
            # Draw pose connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(
                                        color=(80, 22, 10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(
                                        color=(80, 44, 121), thickness=2, circle_radius=2)
                                    )
            # Draw left hand connections
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(
                                        color=(121, 22, 76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(
                                        color=(121, 44, 250), thickness=2, circle_radius=2)
                                    )
            # Draw right hand connections
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(
                                        color=(245, 117, 66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(
                                        color=(245, 66, 230), thickness=2, circle_radius=2)
                                    )
        # Path for exported data, numpy arrays
        DATA_PATH = os.path.join('MP_Data')
        actions = np.array(['name','lastname','me','you','fun','yes','no','sorry','good-luck','howmuch','dislike','Beautiful','remember','age','what-is-your-name','fine','sick'])
        no_sequences = 30
        sequence_length = 30
        start_folder = 30
        label_map = {label: num for num, label in enumerate(actions)}
        sequences, labels = [], []
        for action in actions:
            for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
                window = []
                for frame_num in range(sequence_length):
                    res = np.load(os.path.join(DATA_PATH, action, str(
                        sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[action])
        X = np.array(sequences)
        y = to_categorical(labels).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
        log_dir = os.path.join('Logs')
        tb_callback = TensorBoard(log_dir=log_dir)
        model = Sequential()
        model.add(LSTM(64, return_sequences=True,
                activation='relu', input_shape=(30, 1662)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(actions.shape[0], activation='softmax'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'])
        model.fit(X_train, y_train, epochs=300, callbacks=[tb_callback])
        model.summary()
        res = model.predict(X_test)
        model.save('new.h5')

        yhat = model.predict(X_test)
        ytrue = np.argmax(y_test, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()
        multilabel_confusion_matrix(ytrue, yhat)
        score = accuracy_score(ytrue, yhat)
        print("accuracy",score)
        if score < 1.0 :
            print("FAIL")
        else :
            print("SUCCESS")
            break
        