from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import glob
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report, plot_confusion_matrix
from tensorflow import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2 
import numpy as np
import os 
import matplotlib.pyplot as plt 
import time 
import mediapipe as mp
import tensorflow as tf
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report, plot_confusion_matrix
from tensorflow import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report, plot_confusion_matrix
from tensorflow import math
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from numpy import *
import sys
import sys
import os
from typing_extensions import *
import cv2
import time
import glob
import os
import cv2 
import numpy as np
import os 
import matplotlib.pyplot as plt 
import time 
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

mp_holistic = mp.solutions.holistic # Holistic model 
mp_drawing = mp.solutions.drawing_utils # Drawing utilities 
class train():
        DATA_PATH = os.path.join('MP_Data') 
        actions = ([])
        n = int(input("Enter number of elements : "))
        for i in range(0, n):
            v = str(input("model train:  "))
            actions.append(v)      
        file = str(input("file name:  "))
        actions = append(actions, v)
        score = 0
        while 1.0 >= score:
            def mediapipe_detection(image, model):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
                image.flags.writeable = False                  # Image is no longer writeable
                results = model.process(image)                 # Make prediction
                image.flags.writeable = True                   # Image is now writeable 
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
                return image, results
            def draw_landmarks(image, results):
                #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
            def draw_styled_landmarks(image, results):
                # Draw face connections
                #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                #                         mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                #                         mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                #                         ) 
                # Draw pose connections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                        ) 
                # Draw left hand connections
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                        ) 
                # Draw right hand connections  
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
            def extract_keypoints(results):
                pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
                #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
                lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
                rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
                return np.concatenate([pose, lh, rh])
            # Path for exported data, numpy arrays

            # Actions that we try to detect
            # ADD ACTION HERE 
            #actions = np.array(['nothing', 'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
            #['nothing', '???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???', '???', '???',
            #'???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???']
            #actions = np.array(['nothing', '???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???', '???', '???',
            #'???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','a','b','c','d','e','f_???','g','h_???','i_???','j','k','l_???','m_???','n_???','o','p','q','r_???','s_???','t','u','v','w_???','x','y_???','z'])
            # '???_1' , 'v_2'
            #actions = np.array(['1000000'])

            # adjust ??? 
            # general
            # general = ['nothing','name','lastname','me','you','cute','fun','remember','yes','no','sick','age','same','sorry','fine','how-much','good-luck','hello','like','dislike','beautiful']           
            # general = ['nothing','day','monday','tuesday','wednesday','thursday','friday','saturday','sunday','week','today-now','tomorrow','the-day-after-tomorrow','yesterday',
            #             'the-other-day','month','january','february','march','june','july','august','september','october','november','december','year']
            #??????????????????           
            # general  = ['nothing','age','Do-you-understand','eat','fine','go','have','how-much',
            # # 'hungry','me','mhai','miss','name','nevermind','no','now','or','question',
            # # 'rice','school','sorry','thank-you','time','toothache','what','worry','yang','you']


            # general  = ['nothing','you','age','how-much','have','question','mhai','Do-you-understand','name','what','eat','rice','or','yang','now','time','fine']
            
            #sentence
            # general  = ['nothing','me','no','worry','thank-you','miss','nevermind','fine','sorry','toothache','hungry']

            #general  = ['nothing','me','Do-you-understand','eat','no','have','question','go','school','nevermind','thank-you']
            # Thirty videos worth of data
            no_sequences = 20
            # Videos are going to be 30 frames in length
            sequence_length = 30
            for action in actions: 
                for sequence in range(no_sequences):
                    try: 
                        os.makedirs(os.path.join(DATA_PATH, action, str(sequence+10)))
                    except:
                        pass
            label_map = {label:num for num, label in enumerate(actions)}
            random.seed(10)
            sequences, labels = [], []
            for action in actions:
                for sequence in range(no_sequences):
                    window = []
                    for frame_num in range(sequence_length):
                            res = np.load(os.path.join(DATA_PATH, action, str(sequence+10), "{}.npy".format(frame_num)))
                            window.append(res)
                    sequences.append(window)
                    labels.append(label_map[action])
            X = np.array(sequences)
            print(X.shape)
            y = to_categorical(labels).astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
            model = Sequential()
            model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(30,258)))
            model.add(LSTM(256, return_sequences=True, activation='relu'))
            model.add(LSTM(128, return_sequences=False, activation='relu'))
            #model.add(Dropout(0.2))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(actions.shape[0], activation='softmax'))
            callback = EarlyStopping(monitor='loss', patience=10)
            model.compile(optimizer=Adam(lr=0.00005),loss='categorical_crossentropy',metrics=['accuracy'])
            model.summary()
            history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test,y_test))
            model.summary()
            res = model.predict(X_test)
            model.save(file)
            yhat = model.predict(X_test)
            ytrue = np.argmax(y_test, axis=1).tolist()
            yhat = np.argmax(yhat, axis=1).tolist()
            multilabel_confusion_matrix(ytrue, yhat)
            score = accuracy_score(ytrue, yhat)
            print("accuracy",score)
            if score < 1.0 :
                    print("FAIL")
            else:
                    print("SUCCESS")
                    break