from ast import If, Return
from cgi import test
import sys
import sys
import os
from typing_extensions import *
import cv2
import time
from time import sleep
import os
import asyncio
from PySide2 import *
import cv2
from matplotlib.backend_bases import MouseEvent 
import numpy as np
from numpy import *
import os 
import matplotlib.pyplot as plt 
import time 
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
from pyparsing import Word
import tensorflow as tf
import speech_recognition as stt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from IPython.display import Audio
import gtts
from gtts import gTTS
from playsound import playsound
from threading import Thread
import multiprocessing as mps

from zmq import PUSH
mp_holistic = mp.solutions.holistic # Holistic model 
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
def draw_landmarks(image, results):
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections   
def draw_styled_landmarks(image, results):
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
def testcamera():
	def mediapipe_detection(image, model):
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
			image.flags.writeable = False                  # Image is no longer writeable
			results = model.process(image)                 # Make prediction
			image.flags.writeable = True                   # Image is now writeable 
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
			return image, results
	def extract_keypoints(results, model_name):
			pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
                #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
			lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
			rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
			if model_name ==  'test.h5':
				return np.concatenate([pose, lh, rh])
			else: 
				return np.concatenate([rh]) 

	def get_action(name):
			if name == 'thai.h5':
				return thai
			elif name =='test.h5':
				return new
			else:
				return time		
	thai = ['nothing','korkai','khorkhai','khorkyai','khorrakung','ngorngu','jorjan',
                'chorching','chorchang','zorzoh','chorcher','yorying','dorchada','torbantak',
                'thortan','thornanmunto','torputow','nornean','dordek','dhordhow','thorthung',
                'thorthahan','thorthong','nornu','borbaimai','phorpa','phorphung','forfa',
                'porpan','forfun','phorsampao','morma','yoryak','rorruea','lorling','worwaen',
                'sorsara','sorruesi','sorsuea','horheep','lorchula','orang','hornokhook']
	model_name = {'thai':'thai.h5',
                    'eng':'alphabet.h5',
                    'time' : 'time.h5',
                    'number': 'number.h5',
					'new' : 'test.h5'}
	new = ['nothing','day','my']
	name = model_name['thai']   
	actions = np.array(get_action(name))
	model = tf.keras.models.load_model(name)
	colors=[]
	for i in range(100):
		colors.append((np.random.randint(256),np.random.randint(256),np.random.randint(256)))  
	def prob_viz(res, actions, input_frame, colors):
			scale = 0.25
			output_frame = input_frame.copy()
			for num, prob in enumerate(res):
				cv2.rectangle(output_frame, (0,int(40+num*35*scale)), (int(prob*100*scale), int(50+num*35*scale)), colors[num], -1)
				cv2.putText(output_frame, actions[num], (0, int(50+num*35*scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
			return output_frame
	sequence = []
	sentence = []
	predictions = []
	threshold = 0.8
	predicted = ''
	text = ""
	count_same_frame = 0
	keypress = cv2.waitKey(1)
	wCam, hCam = 1280,720
	cap = cv2.VideoCapture(0)
	cap.set(3, wCam)
	cap.set(4, hCam)
        # Set mediapipe model 
	with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
		while cap.isOpened():
                # Read feed
			ret, frame = cap.read()
			old_text = predicted
                # Make detections
			image, results = mediapipe_detection(frame, holistic)
			#print(results)
			cv2.rectangle(image, (0,0), (100, 640), (0, 0, 0), -1)
                # Draw landmarks
			draw_styled_landmarks(image, results)
                # 2. Prediction logic
                #keypoints = extract_keypoints(results)
			keypoints = extract_keypoints(results,model_name=name)
			sequence.append(keypoints)
			sequence = sequence[-30:]
			if len(sequence) == 30:
				res = model.predict(np.expand_dims(sequence, axis=0))[0]
				predicted = actions[np.argmax(res)]
				predictions.append(np.argmax(res))  
				old_text = predicted
				if old_text == word:
					count_same_frame += 1
				elif old_text == "nothing" :
					count_same_frame = 0			

				if predicted == "nothing":
					if count_same_frame >= 100 :
						print("test")
						text = ''
				elif count_same_frame > 50:
					if len(predicted) == 1:
						Thread(args=(predicted, )).start()
					#tts = gTTS(predicted, lang='th')
					#tts.save('speech.mp3')
					text = predicted + ' ' + text
					#playsound('./speech.mp3')						
					print(text)
					count_same_frame = 0
				
				# if  keypress == ord('t'):
				# 	Thread(args=(text, )).start()
				# 	word1 = predicted + text
				# 	print(word1)
                #3. Viz logic
				# if np.unique(predictions[-30:])[0]==np.argmax(res): 
				# 	if res[np.argmax(res)] > threshold: 
							
				# 		if len(sentence) > 0: 
				# 			if actions[np.argmax(res)] != sentence[-1]:
				# 				sentence.append(actions[np.argmax(res)])
				# 		else:
				# 			sentence.append(actions[np.argmax(res)])
							
				# if len(sentence) > 20: 
				# 	sentence = sentence[-5:] 
			
                # Viz probabilities
				image = prob_viz(res, actions, image, colors) 
			word = predicted
			#print(word)
			cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
			cv2.putText(image, word, (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  
			
			cv2.imshow('OpenCV Feed', image)              
                # Break gracefully
			if cv2.waitKey(10) & 0xFF == ord('q'):
				break
            
		cap.release()
		cv2.destroyAllWindows()
	return word


if(__name__=='__main__'):
    p = mps.Process(target=testcamera)	 
    p.start()
	