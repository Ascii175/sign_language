from cgi import test
import sys
import os
from tkinter import Image, font
from PySide2 import *
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from ui_untitled import *
from threading import Thread
from PyQt5.QtWidgets import QApplication,QMainWindow,QPushButton,QLabel,QFileDialog
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from gtts import gTTS
from playsound import *
from function_sound_file import function_sound
import speech_recognition as stt

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
class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)	
		self.timer = QTimer()
		self.timer.timeout.connect(self.viewCam)
		
		self.ui.train2.clicked.connect(self.train)
		self.ui.pushButton_13.clicked.connect(self.reload)

		self.ui.playSo.clicked.connect(self.playSo)
		self.ui.mic.clicked.connect(self.microphone)
		self.ui.showpic.clicked.connect(self.showpic)

		self.ui.time.clicked.connect(self.time)
		self.ui.general.clicked.connect(self.general_1662)
		self.ui.day.clicked.connect(self.day)
		self.ui.num.clicked.connect(self.number)
		self.ui.question.clicked.connect(self.quest)
		self.ui.sentence.clicked.connect(self.sentence)
		self.ui.deny.clicked.connect(self.deny)

	def train(self):
		import train_258 
		print("ไม่ต้องเทรนอีกครั้งเเล้ว")
#################################################################################################################################################################
	def reload(self):
		return 0
	def showpic(self):
		img = cv2.imread("./A_Z.png")
		self.ui.image_label.append()
		cv2.imshow("A-Z",img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		# fname = QFileDialog.getOpenFileName(self, "Open File", "./A_Z.png","All Files(*);; PNG")
		# self.pixmap = QPixmap(fname[0])
		# self.labe
#################################################################################################################################################################
	def time(self):
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
				if model_name ==  'time.h5':
					return np.concatenate([pose, lh, rh])
				else: 
					return np.concatenate([rh]) 

		def get_action(name):
				if name == 'time.h5':
					return time
				elif name =='test.h5':
					return new

		time = ['nothing','เวลา','บ่าย','เย็น','เช้า','กลางวัน','ค่ำ','กลางคืน','เที่ยงคืน']
		#time = ['nothing','time','afternoon','evening','morning','midday','twilight','night-time','midnight']
		new = ['nothing','day','my']
		model_name = {'time':'time.h5',
						'new' : 'test.h5'}
		name = model_name['time']   
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
					cv2.putText(output_frame, actions[num], (0, int(50+num*35*scale)),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
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
					if old_text == "nothing":
						count_same_frame = 0
					elif old_text == word:
						count_same_frame += 1			

					if predicted == "nothing":
						if count_same_frame >= 150 :
							text = ''
					elif count_same_frame > 50:
						if len(predicted) == 1:
							Thread(args=(predicted, )).start()
						
						text = text + ' ' + predicted
						tts = gTTS(text, lang='th')
						tts.save('speech.mp3')
						self.ui.textBrowser.append(text) 
						print(text)
											
						count_same_frame = 0
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
##############################################################################################################################################################
	def general_1662(self):
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
				if model_name ==  'general.h5':
					return np.concatenate([pose, lh, rh])
				else: 
					return np.concatenate([rh]) 

		def get_action(name):
				if name == 'general.h5':
					return general
				elif name =='general.h5':
					return new

		general = ['name','lastname','me','you','fun','yes','no','sorry','good-luck','howmuch','dislike','Beautiful','remember','age','what-is-your-name','fine','sick']
		#question = ['nothing','you','age','how-much','have','question','mhai','Do-you-understand',
		#			'name','what','eat','rice','or','yang','now','time','fine']
		new = ['nothing','day','my']
		model_name = {'general':'general.h5',
						'new' : 'test.h5'}
		name = model_name['general']   
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
					cv2.putText(output_frame, actions[num], (0, int(50+num*35*scale)),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
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
					if old_text == "nothing":
						count_same_frame = 0
					elif old_text == word:
						count_same_frame += 1			

					if predicted == "nothing":
						if count_same_frame >= 150 :
							text = ''
					elif count_same_frame > 50:
						if len(predicted) == 1:
							Thread(args=(predicted, )).start()
						
						text = text + ' ' + predicted
						tts = gTTS(text, lang='th')
						tts.save('speech.mp3')
						self.ui.textBrowser.append(text) 
						print(text)
											
						count_same_frame = 0
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
#################################################################################################################################################################
	def day(self):##########
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
				if model_name ==  'day.h5':
					return np.concatenate([pose, lh, rh])
				else: 
					return np.concatenate([rh]) 
		def get_action(name):
				if name == 'day.h5':
					return day
				elif name =='general.h5':
					return new
		day = ['nothing','วัน','วันจันทร์','วันอังคาร','วันพุธ','วันพฤหัสบดี','วันศุกร์','วันเสาร์','วันอาทิตย์','สัปดาห์','วันนี้','พรุ่งนี้','มะรืนนี้','เมื่อวานนี้','เมื่อวานซืน','เดือน','มกราคม','กุมภาพันธ์',
			'มีนาคม','เมษาคม','พฤษภาคม','มิถุนายม','กรกฎาคม','สิงหาคม','กันยายน','ตุลาคม','พฤศจิกายน','ธันวาคม','ปี']
		#question = ['nothing','you','age','how-much','have','question','mhai','Do-you-understand',
		#			'name','what','eat','rice','or','yang','now','time','fine']
		new = ['nothing','day','my']
		model_name = {'day':'day.h5',
						'new' : 'test.h5'}
		name = model_name['day']   
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
					cv2.putText(output_frame, actions[num], (0, int(50+num*35*scale)),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
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
					if old_text == "nothing":
						count_same_frame = 0
					elif old_text == word:
						count_same_frame += 1			

					if predicted == "nothing":
						if count_same_frame >= 150 :
							text = ''
					elif count_same_frame > 50:
						if len(predicted) == 1:
							Thread(args=(predicted, )).start()
						
						text = text + ' ' + predicted
						tts = gTTS(text, lang='th')
						tts.save('speech.mp3')
						self.ui.textBrowser.append(text) 
						print(text)
											
						count_same_frame = 0
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
#################################################################################################################################################################
	def number(self):
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
				if model_name ==  'number.h5':
					return np.concatenate([pose, lh, rh])
				else: 
					return np.concatenate([rh]) 

		def get_action(name):
				if name == 'number.h5':
					return number
				elif name =='number.h5':
					return new

		number = ['nothing','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15',
                '16','17','18','19','20','30','40','50','60','70','80','90','100','1000',
                '10000','100000','1000000']
		#question = ['nothing','you','age','how-much','have','question','mhai','Do-you-understand',
		#			'name','what','eat','rice','or','yang','now','time','fine']
		new = ['nothing','day','my']
		model_name = {'number':'number.h5',
						'new' : 'test.h5'}
		name = model_name['number']   
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
					cv2.putText(output_frame, actions[num], (0, int(50+num*35*scale)),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
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
					if old_text == "nothing":
						count_same_frame = 0
					elif old_text == word:
						count_same_frame += 1			

					if predicted == "nothing":
						if count_same_frame >= 150 :
							text = ''
					elif count_same_frame > 50:
						if len(predicted) == 1:
							Thread(args=(predicted, )).start()
						
						text = text + ' ' + predicted
						tts = gTTS(text, lang='th')
						tts.save('speech.mp3')
						self.ui.textBrowser.append(text) 
						print(text)
											
						count_same_frame = 0
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
##################################################################################################################################################################
	def microphone(self):
		recog = stt.Recognizer()
		with stt.Microphone() as mic:
			print("กำลังอัดเสียง")
			audio = recog.listen( mic )
			try:
				self.ui.textBrowser.append(recog.recognize_google(audio,None,'th'))
			except stt.UnknownValueError:
				print("ไม่เข้าใจเสียงที่นำเข้า")
			except stt.RequestError as e:
				print("ไม่สามารถนำข้อมูลมาจากบริการของ Google: {0}".format(e))
#################################################################################################################################################################
	def quest(self):
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
				if model_name ==  'question.h5':
					return np.concatenate([pose, lh, rh])
				else: 
					return np.concatenate([rh]) 

		def get_action(name):
				if name == 'question.h5':
					return question
				elif name =='test.h5':
					return new

		question = ['nothing','คุณ','อายุ','เท่าไหร่','มี','คำถาม','ไหม','เข้าใจ',
					'ชื่อ','อะไร','กิน','ข้าว','หรือ','ยัง','ตอนนี้','เวลา','สบายดี']
		#question = ['nothing','you','age','how-much','have','question','mhai','Do-you-understand',
		#			'name','what','eat','rice','or','yang','now','time','fine']
		new = ['nothing','day','my']

		model_name = {'question':'question.h5',
						'new' : 'test.h5'}
		name = model_name['question']   
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
					cv2.putText(output_frame, actions[num], (0, int(50+num*35*scale)),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
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
					if old_text == "nothing":
						count_same_frame = 0
					elif old_text == word:
						count_same_frame += 1			

					if predicted == "nothing":
						if count_same_frame >= 150 :
							text = ''
					elif count_same_frame > 50:
						if len(predicted) == 1:
							Thread(args=(predicted, )).start()
						
						text = text + ' ' + predicted
						tts = gTTS(text, lang='th')
						tts.save('speech.mp3')
						self.ui.textBrowser.append(text) 
						print(text)
											
						count_same_frame = 0
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
#################################################################################################################################################################
	def sentence(self):
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
				if model_name ==  'sentence.h5':
					return np.concatenate([pose, lh, rh])
				else: 
					return np.concatenate([rh]) 

		def get_action(name):
				if name == 'sentence.h5':
					return sentence
				elif name =='test.h5':
					return new

		sentence = ['nothing','ฉัน','ไม่','ไม่สบายใจ','ขอบคุณ','คิดถึง','ไม่เป็นไร','สบายดี','ขอโทษ','ปวดฟัน','หิว']
		#sentence = ['nothing','me','no','worry','thank-you','miss','nevermind','fine','sorry','toothache','hungry']
		new = ['nothing','day','my']

		model_name = {'sentence':'sentence.h5',
						'new' : 'test.h5'}
		name = model_name['sentence']   
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
					cv2.putText(output_frame, actions[num], (0, int(50+num*35*scale)),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
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
					if old_text == "nothing":
						count_same_frame = 0
					elif old_text == word:
						count_same_frame += 1			

					if predicted == "nothing":
						if count_same_frame >= 150 :
							text = ''
					elif count_same_frame > 50:
						if len(predicted) == 1:
							Thread(args=(predicted, )).start()
						
						text = text + ' ' + predicted
						tts = gTTS(text, lang='th')
						tts.save('speech.mp3')
						self.ui.textBrowser.append(text) 
						print(text)
											
						count_same_frame = 0
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
#################################################################################################################################################################
	def deny(self):
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
				if model_name ==  'deny.h5':
					return np.concatenate([pose, lh, rh])
				else: 
					return np.concatenate([rh]) 

		def get_action(name):
				if name == 'deny.h5':
					return deny
				elif name =='deny.h5':
					return new

		deny = ['nothing','ฉัน','เข้าใจ','ทาน','ไม่','มี','คำถาม','ไป','โรงเรียน','ไม่เป็นไร','ขอบคุณ']
		#deny = ['nothing','me','Do-you-understand','eat','no','have','question','go','school','nevermind','thank-you']
		new = ['nothing','day','my']

		model_name = {'deny':'deny.h5',
						'new' : 'test.h5'}
		name = model_name['deny']   
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
					cv2.putText(output_frame, actions[num], (0, int(50+num*35*scale)),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
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
					if old_text == "nothing":
						count_same_frame = 0
					elif old_text == word:
						count_same_frame += 1			

					if predicted == "nothing":
						if count_same_frame >= 150 :
							text = ''
					elif count_same_frame > 50:
						if len(predicted) == 1:
							Thread(args=(predicted, )).start()
						
						text = text + ' ' + predicted
						tts = gTTS(text, lang='th')
						tts.save('speech.mp3')
						self.ui.textBrowser.append(text) 
						print(text)
											
						count_same_frame = 0
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
################################################################################################################################################################
	def playSo(self):
		function_sound()
#################################################################################################################################################################
	def viewCam(self):
        # read image in BGR format
		ret, image = self.cap.read()
        # convert image to RGB format
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get image infos
		height, width, channel = image.shape
		step = channel * width
        # create QImage from image
		qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
		self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))
    # start/stop timer
	def controlTimer(self):
			# if timer is stopped
		if not self.timer.isActive():
				# create video capture
			self.cap = cv2.VideoCapture(0)
				# start timer
			self.timer.start(20)
				# update control_bt text
			# if timer is started
		else:
				# stop timer
			self.timer.stop()
				# release video capture
			self.cap.release()
				# update control_bt text          

			
########################################################################
## EXECUTE APP
########################################################################
if __name__ == "__main__":
	
	app = QApplication(sys.argv)
	window = MainWindow()
	window.show()
	sys.exit(app.exec_())
########################################################################
## END===>
########################################################################
