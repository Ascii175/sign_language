########################################################################
## SPINN DESIGN CODE
# YOUTUBE: (SPINN TV) https://www.youtube.com/spinnTv
# WEBSITE: spinndesign.com
########################################################################

########################################################################
## IMPORTS
########################################################################
from ast import Return
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
import multiprocessing as mps
########################################################################
# IMPORT GUI FILE
from ui_interface import *
########################################################################


from Custom_Widgets.Widgets import *

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
start = time.time()
async def async_sleep():
    	await asyncio.sleep(1)
########################################################################
## MAIN WINDOW CLASS
########################################################################
class MainWindow(QMainWindow):
	def __init__(self, parent=None):
		QMainWindow.__init__(self)
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)
		
		########################################################################
		# APPLY JSON STYLESHEET
		########################################################################
		# self = QMainWindow class
		# self.ui = Ui_MainWindow / user interface class
		# loadJsonStyle(self, self.ui)
		########################################################################

		#######################################################################
		# SHOW WINDOW
		#######################################################################

        
		
		########################################################################
		##
		########################################################################
		# self.ui.pushButton_8.clicked.connect(lambda: self.ui.notification.expandMenu())
		# self.ui.microphone.clicked.connect(lambda: self.microphone())
		
		self.ui.opencamera.clicked.connect(lambda: self.testcamera())
		# self.ui.opencamera.clicked.connect(lambda: self.commonword())
		self.ui.addSing.clicked.connect(lambda: self.addSing())

		self.show()

	def microphone(self):
		recog = stt.Recognizer()
		with stt.Microphone() as mic:
			print("กำลังอัดเสียง")
			audio = recog.listen( mic )
			try:
				self.ui.textBrowser.append(recog.recognize_google(audio,None,'th'))
			except stt.UnknownValueError:
				print("Google ไม่เข้าใจเสียงที่นำเข้า")
			except stt.RequestError as e:
				print("ไม่สามารถนำข้อมูลมาจากบริการของ Google: {0}".format(e))

	# def train(self):
	# 	from train import train
	# 	return train

	def addSing(self):
		# from testarry import addtext
		# if __name__ == '__main__':
			while True:
				actions = np.array([])
				v = str(input("Element:  "))
				actions = append(actions, v)
				print(actions)
				break
			# except:
			# 	pass
				# actions = np.array([])
				# v = str(input("Element:  "))
				# actions = append(actions, v)
				# print(actions)
###################################################################################################################################################################################
	def testcamera(self):
		
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
			elif name == 'number.h5':
				return num
			elif name =='alphabet.h5':
				return az
			elif name =='test.h5':
				return new
			else:
				return time		
		az = ['nothing','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o',
                'p','q','r','s','t','u','v','w','x','y','z']     
		time = ['nothing','lunch','midday','midnight','12pm','dinner','almost-midnight']
		thai = ['nothing','korkai','khorkhai','khorkyai','khorrakung','ngorngu','jorjan',
                'chorching','chorchang','zorzoh','chorcher','yorying','dorchada','torbantak',
                'thortan','thornanmunto','torputow','nornean','dordek','dhordhow','thorthung',
                'thorthahan','thorthong','nornu','borbaimai','phorpa','phorphung','forfa',
                'porpan','forfun','phorsampao','morma','yoryak','rorruea','lorling','worwaen',
                'sorsara','sorruesi','sorsuea','horheep','lorchula','orang','hornokhook']
		num = ['nothing','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15',
                '16','17','18','19','20','30','40','50','60','70','80','90','100','1000',
                '10000','100000','1000000']
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
		predicted = 'nothing'
		wCam, hCam = 1280,720
		cap = cv2.VideoCapture(0)
		cap.set(3, wCam)
		cap.set(4, hCam)
        # Set mediapipe model 
		with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
			while cap.isOpened():
                # Read feed
				ret, frame = cap.read()
        
                # Make detections
				image, results = mediapipe_detection(frame, holistic)
				print(results)
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
					
                #3. Viz logic
					if np.unique(predictions[-30:])[0]==np.argmax(res): 
						if res[np.argmax(res)] > threshold: 
							
							if len(sentence) > 0: 
								if actions[np.argmax(res)] != sentence[-1]:
									sentence.append(actions[np.argmax(res)])
							else:
								sentence.append(actions[np.argmax(res)])
							
					if len(sentence) > 5: 
						sentence = sentence[-5:] 
                # Viz probabilities
					image = prob_viz(res, actions, image, colors) 
				word = predicted
				cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
				cv2.putText(image, word, (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  
							
				##################### time ###########################
				if word == "nothing":	
					self.ui.textBrowser.append("ไม่สามารถอ่านได้") 
					print(word)
					# try:
					# 	playsound('./speech.mp3')	
					# except:
					# 	print("tt")			
				# elif word == "lunch" :
				# 	self.ui.textBrowser.append("กลางวัน")
# 				elif word == "midday":
# 					self.ui.textBrowser.append("เที่ยงวัน")
# 				elif word == "12pm":
# 					self.ui.textBrowser.append("12pm")
# 				elif word == "midnight":
# 					self.ui.textBrowser.append("เที่ยงคืน")
# 				elif word == "dinner":
# 					self.ui.textBrowser.append("ตอนเย็น")
# 				elif word == "almost-midnight":
# 					self.ui.textBrowser.append("เกือบเที่ยงคืน")
# 				#################### thai ############################
				elif word == "korkai":
					self.ui.textBrowser.append("ก")
				elif word == "khorkhai":
					self.ui.textBrowser.append("ข")
				elif word == "khorkyai":
					self.ui.textBrowser.append("ค")
				elif word == "khorrakung":
					self.ui.textBrowser.append("ฆ")
				elif word == "ngorngu":
					self.ui.textBrowser.append("ง")
				elif word == "jorjan":
					self.ui.textBrowser.append("จ")
				elif word == "chorching":
					self.ui.textBrowser.append("ฉ")
				elif word == "chorchang":
					self.ui.textBrowser.append("ช")
				elif word == "zorzoh":
					self.ui.textBrowser.append("ซ")
				elif word == "chorcher":
					self.ui.textBrowser.append("ฌ")
				elif word == "yorying":
					self.ui.textBrowser.append("ช")
				elif word == "dorchada":
					self.ui.textBrowser.append("ฎ")
				elif word == "torbantak":
					self.ui.textBrowser.append("ฏ")
				elif word == "thortan":
					self.ui.textBrowser.append("ฐ")
				elif word == "thornanmunto":
					self.ui.textBrowser.append("ฑ")
				elif word == "torputow":
					self.ui.textBrowser.append("ฒ")
				elif word == "nornean":
					self.ui.textBrowser.append("ณ")
				elif word == "dordek":
					self.ui.textBrowser.append("ด")
				elif word == "dhordhow":
					self.ui.textBrowser.append("ต")
				elif word == "thorthung":
					self.ui.textBrowser.append("ถ")
				elif word == "thorthahan":
					self.ui.textBrowser.append("ท")
				elif word == "thorthong":
					self.ui.textBrowser.append("ธ")
				elif word == "nornu":
					self.ui.textBrowser.append("น")
				elif word == "borbaimai":
					self.ui.textBrowser.append("บ")
				elif word == "phorpa":
					self.ui.textBrowser.append("ป")
				elif word == "forfa":
					self.ui.textBrowser.append("ฝ")
				elif word == "porpan":
					self.ui.textBrowser.append("พ")
				elif word == "forfun":
					self.ui.textBrowser.append("ฟ")
				elif word == "phorsampao":
					self.ui.textBrowser.append("ภ")
				elif word == "morma":
					self.ui.textBrowser.append("ม")
					self.ui.microphone.clicked.connect(lambda: playsound('./speech.mp3'))
				elif word == "yoryak":
					self.ui.textBrowser.append("ย")
				elif word == "rorruea":
					self.ui.textBrowser.append("ร")
				elif word == "lorling":
					self.ui.textBrowser.append("ล")
				elif word == "worwaen":
					self.ui.textBrowser.append("ว")
				elif word == "sorsara":
					self.ui.textBrowser.append("ศ")
				elif word == "sorruesi":
					self.ui.textBrowser.append("ษ")
				elif word == "sorsuea":
					self.ui.textBrowser.append("ส")
				elif word == "horheep":
					self.ui.textBrowser.append("ห")
				elif word == "lorchula":
					self.ui.textBrowser.append("ฬ")
				elif word == "orang":
					self.ui.textBrowser.append("อ")
				elif word == "hornokhook":
					self.ui.textBrowser.append("ฮ")
# 	#################################Eng###################################################
# 				elif word == "a":
# 					self.ui.textBrowser.append("A")
# 				elif word == "b":
# 					self.ui.textBrowser.append("B")
# 				elif word == "c":
# 					self.ui.textBrowser.append("C")
# 				elif word == "d":
# 					self.ui.textBrowser.append("D")
# 				elif word == "e":
# 					self.ui.textBrowser.append("E")
# 				elif word == "f":
# 					self.ui.textBrowser.append("F")
# 				elif word == "g":
# 					self.ui.textBrowser.append("G")
# 				elif word == "h":
# 					self.ui.textBrowser.append("H")
# 				elif word == "i":
# 					self.ui.textBrowser.append("I")
# 				elif word == "j":
# 					self.ui.textBrowser.append("J")
# 				elif word == "k":
# 					self.ui.textBrowser.append("K")
# 				elif word == "l":
# 					self.ui.textBrowser.append("L")
# 				elif word == "m":
# 					self.ui.textBrowser.append("M")
# 				elif word == "n":
# 					self.ui.textBrowser.append("N")
# 				elif word == "o":
# 					self.ui.textBrowser.append("O")
# 				elif word == "p":
# 					self.ui.textBrowser.append("P")
# 				elif word == "q":
# 					self.ui.textBrowser.append("Q")
# 				elif word == "r":
# 					self.ui.textBrowser.append("R")
# 				elif word == "s":
# 					self.ui.textBrowser.append("S")
# 				elif word == "t":
# 					self.ui.textBrowser.append("T")
# 				elif word == "u":
# 					self.ui.textBrowser.append("U")
# 				elif word == "v":
# 					self.ui.textBrowser.append("V")
# 				elif word == "w":
# 					self.ui.textBrowser.append("W")
# 				elif word == "x":
# 					self.ui.textBrowser.append("X")
# 				elif word == "y":
# 					self.ui.textBrowser.append("Y")
# 				elif word == "z":
# 					self.ui.textBrowser.append("Z")
# ####################################Num#######################################################
# 				elif word == "1":
# 					self.ui.textBrowser.append("1")
# 				elif word == "2":
# 					self.ui.textBrowser.append("2")
# 				elif word == "3":
# 					self.ui.textBrowser.append("3")
# 				elif word == "4":
# 					self.ui.textBrowser.append("4")
# 				elif word == "5":
# 					self.ui.textBrowser.append("5")
# 				elif word == "6":
# 					self.ui.textBrowser.append("6")
# 				elif word == "7":
# 					self.ui.textBrowser.append("7")
# 				elif word == "8":
# 					self.ui.textBrowser.append("8")
# 				elif word == "9":
# 					self.ui.textBrowser.append("9")
# 				elif word == "10":
# 					self.ui.textBrowser.append("10")
# 				elif word == "11":
# 					self.ui.textBrowser.append("11")
# 				elif word == "12":
# 					self.ui.textBrowser.append("12")	
# 				elif word == "13":
# 					self.ui.textBrowser.append("13")
# 				elif word == "14":
# 					self.ui.textBrowser.append("14")
# 				elif word == "15":
# 					self.ui.textBrowser.append("15")
# 				elif word == "16":
# 					self.ui.textBrowser.append("16")
# 				elif word == "17":
# 					self.ui.textBrowser.append("17")
# 				elif word == "18":
# 					self.ui.textBrowser.append("18")
# 				elif word == "19":
# 					self.ui.textBrowser.append("19")
# 				elif word == "20":
# 					self.ui.textBrowser.append("20")
# 				elif word == "30":
# 					self.ui.textBrowser.append("30")
# 				elif word == "40":
# 					self.ui.textBrowser.append("40")
# 				elif word == "50":
# 					self.ui.textBrowser.append("50")
# 				elif word == "60":
# 					self.ui.textBrowser.append("60")
# 				elif word == "70":
# 					self.ui.textBrowser.append("70")
# 				elif word == "80":
# 					self.ui.textBrowser.append("80")
# 				elif word == "90":
# 					self.ui.textBrowser.append("90")
# 				elif word == "100":
# 					self.ui.textBrowser.append("100")
# 				elif word == "1000":
# 					self.ui.textBrowser.append("1,000")
# 				elif word == "10000":
# 					self.ui.textBrowser.append("10,000")
# 				elif word == "100000":
# 					self.ui.textBrowser.append("100,000")
# 				elif word == "1,000,000":
# 					self.ui.textBrowser.append("1,000,000")				
                # Show to screen
				
				cv2.imshow('OpenCV Feed', image)

                # Break gracefully
				if cv2.waitKey(10) & 0xFF == ord('q'):
					break
			cap.release()
			cv2.destroyAllWindows()


	
########################################################################
## EXECUTE APP
########################################################################
if __name__ == "__main__":
	app = QApplication(sys.argv)
	########################################################################
	##
	########################################################################
	window = MainWindow()
	window.show()
	sys.exit(app.exec_())
########################################################################
## END===>
########################################################################
