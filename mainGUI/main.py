from cgi import test
import sys
import os
from PySide2 import *
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from ui_untitled import *

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
import tensorflow as tf


class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)	

		self.timer = QTimer()
		self.timer.timeout.connect(self.viewCam)
		self.ui.pushButton_3.clicked.connect(self.controlTimer)

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
