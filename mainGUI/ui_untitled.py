# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'untitledpoamMp.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import icons_rc
import icons_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(895, 715)
        MainWindow.setStyleSheet(u"")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setStyleSheet(u"*{\n"
"	border: none;\n"
"	color: #fff;\n"
"}\n"
"#centralwidget{\n"
"	background-color: #09102a;\n"
"}\n"
"\n"
"QPushButton{\n"
"	text-align: left;\n"
"	background-color: #08112a;\n"
"	padding: 10px 2px;\n"
"	border-radius: 10px;\n"
"}\n"
"\n"
"\n"
"\n"
"")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.sid_menu_container = QFrame(self.centralwidget)
        self.sid_menu_container.setObjectName(u"sid_menu_container")
        self.sid_menu_container.setMaximumSize(QSize(300, 16777215))
        self.sid_menu_container.setFrameShape(QFrame.StyledPanel)
        self.sid_menu_container.setFrameShadow(QFrame.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.sid_menu_container)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.side_menu = QFrame(self.sid_menu_container)
        self.side_menu.setObjectName(u"side_menu")
        self.side_menu.setFrameShape(QFrame.StyledPanel)
        self.side_menu.setFrameShadow(QFrame.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.side_menu)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.frame_8 = QFrame(self.side_menu)
        self.frame_8.setObjectName(u"frame_8")
        self.frame_8.setStyleSheet(u"background-color: rgb(0, 0, 0);")
        self.frame_8.setFrameShape(QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.frame_8)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(-1, 5, 0, -1)
        self.label = QLabel(self.frame_8)
        self.label.setObjectName(u"label")
        font = QFont()
        font.setFamily(u"TH Mali Grade 6")
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)

        self.horizontalLayout_4.addWidget(self.label)

        self.label_2 = QLabel(self.frame_8)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setPixmap(QPixmap(u":/icons/icons/git-branch.svg"))

        self.horizontalLayout_4.addWidget(self.label_2)


        self.verticalLayout_3.addWidget(self.frame_8, 0, Qt.AlignTop)

        self.frame_9 = QFrame(self.side_menu)
        self.frame_9.setObjectName(u"frame_9")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_9.sizePolicy().hasHeightForWidth())
        self.frame_9.setSizePolicy(sizePolicy)
        self.frame_9.setStyleSheet(u"*{\n"
"	border:none;\n"
"}")
        self.frame_9.setFrameShape(QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QFrame.Raised)
        self.verticalLayout_4 = QVBoxLayout(self.frame_9)
        self.verticalLayout_4.setSpacing(9)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 9, 9)
        self.toolBox = QToolBox(self.frame_9)
        self.toolBox.setObjectName(u"toolBox")
        self.toolBox.setFont(font)
        self.toolBox.setStyleSheet(u"menu2,QWidget{\n"
"	background-color: rgb(0, 0, 0);\n"
"	border:none;\n"
"}\n"
"")
        self.menu1 = QWidget()
        self.menu1.setObjectName(u"menu1")
        self.menu1.setGeometry(QRect(0, 0, 213, 534))
        self.menu1.setStyleSheet(u"border:none;\n"
"")
        self.verticalLayout_5 = QVBoxLayout(self.menu1)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.frame_11 = QFrame(self.menu1)
        self.frame_11.setObjectName(u"frame_11")
        self.frame_11.setFrameShape(QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QFrame.Raised)
        self.verticalLayout_6 = QVBoxLayout(self.frame_11)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.thai = QPushButton(self.frame_11)
        self.thai.setObjectName(u"thai")
        font1 = QFont()
        font1.setFamily(u"TH Mali Grade 6")
        font1.setPointSize(16)
        font1.setBold(True)
        font1.setWeight(75)
        self.thai.setFont(font1)
        icon = QIcon()
        icon.addFile(u":/icons/icons/camera.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.thai.setIcon(icon)
        self.thai.setIconSize(QSize(32, 32))

        self.verticalLayout_6.addWidget(self.thai)

        self.eng = QPushButton(self.frame_11)
        self.eng.setObjectName(u"eng")
        self.eng.setFont(font1)
        self.eng.setIcon(icon)
        self.eng.setIconSize(QSize(32, 32))

        self.verticalLayout_6.addWidget(self.eng)

        self.time = QPushButton(self.frame_11)
        self.time.setObjectName(u"time")
        self.time.setMaximumSize(QSize(999999, 2000))
        self.time.setSizeIncrement(QSize(50, 50))
        self.time.setBaseSize(QSize(200, 200))
        self.time.setFont(font1)
        self.time.setStyleSheet(u"")
        self.time.setIcon(icon)
        self.time.setIconSize(QSize(32, 32))

        self.verticalLayout_6.addWidget(self.time, 0, Qt.AlignBottom)

        self.general = QPushButton(self.frame_11)
        self.general.setObjectName(u"general")
        self.general.setFont(font1)
        self.general.setStyleSheet(u"")
        self.general.setIcon(icon)
        self.general.setIconSize(QSize(32, 32))

        self.verticalLayout_6.addWidget(self.general, 0, Qt.AlignBottom)

        self.day = QPushButton(self.frame_11)
        self.day.setObjectName(u"day")
        self.day.setFont(font1)
        self.day.setStyleSheet(u"")
        self.day.setIcon(icon)
        self.day.setIconSize(QSize(32, 32))

        self.verticalLayout_6.addWidget(self.day, 0, Qt.AlignBottom)

        self.num = QPushButton(self.frame_11)
        self.num.setObjectName(u"num")
        font2 = QFont()
        font2.setFamily(u"TH Mali Grade 6")
        font2.setPointSize(16)
        font2.setBold(True)
        font2.setItalic(False)
        font2.setUnderline(False)
        font2.setWeight(75)
        font2.setStrikeOut(False)
        self.num.setFont(font2)
        self.num.setStyleSheet(u"")
        self.num.setIcon(icon)
        self.num.setIconSize(QSize(32, 32))

        self.verticalLayout_6.addWidget(self.num, 0, Qt.AlignBottom)

        self.deny = QPushButton(self.frame_11)
        self.deny.setObjectName(u"deny")
        self.deny.setFont(font1)
        icon1 = QIcon()
        icon1.addFile(u":/icons/icons/github.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.deny.setIcon(icon1)
        self.deny.setIconSize(QSize(32, 32))

        self.verticalLayout_6.addWidget(self.deny)

        self.question = QPushButton(self.frame_11)
        self.question.setObjectName(u"question")
        self.question.setFont(font1)
        icon2 = QIcon()
        icon2.addFile(u":/icons/icons/thumbs-up.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.question.setIcon(icon2)
        self.question.setIconSize(QSize(32, 32))

        self.verticalLayout_6.addWidget(self.question)

        self.sentence = QPushButton(self.frame_11)
        self.sentence.setObjectName(u"sentence")
        self.sentence.setFont(font1)
        icon3 = QIcon()
        icon3.addFile(u":/icons/icons/user-check.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.sentence.setIcon(icon3)
        self.sentence.setIconSize(QSize(32, 32))

        self.verticalLayout_6.addWidget(self.sentence)


        self.verticalLayout_5.addWidget(self.frame_11)

        icon4 = QIcon()
        icon4.addFile(u":/icons/icons/chevron-down.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.toolBox.addItem(self.menu1, icon4, u"\u0e2b\u0e21\u0e27\u0e14\u0e04\u0e33-\u0e1b\u0e23\u0e30\u0e42\u0e22\u0e04")
        self.menu2 = QWidget()
        self.menu2.setObjectName(u"menu2")
        self.menu2.setGeometry(QRect(0, -23, 213, 476))
        self.verticalLayout_7 = QVBoxLayout(self.menu2)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.frame_12 = QFrame(self.menu2)
        self.frame_12.setObjectName(u"frame_12")
        palette = QPalette()
        brush = QBrush(QColor(255, 255, 255, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.WindowText, brush)
        brush1 = QBrush(QColor(0, 0, 0, 255))
        brush1.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Button, brush1)
        palette.setBrush(QPalette.Active, QPalette.Text, brush)
        palette.setBrush(QPalette.Active, QPalette.ButtonText, brush)
        palette.setBrush(QPalette.Active, QPalette.Base, brush1)
        palette.setBrush(QPalette.Active, QPalette.Window, brush1)
        brush2 = QBrush(QColor(85, 255, 0, 128))
        brush2.setStyle(Qt.SolidPattern)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Active, QPalette.PlaceholderText, brush2)
#endif
        palette.setBrush(QPalette.Inactive, QPalette.WindowText, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.Text, brush)
        palette.setBrush(QPalette.Inactive, QPalette.ButtonText, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Base, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.Window, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush2)
#endif
        palette.setBrush(QPalette.Disabled, QPalette.WindowText, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Text, brush)
        palette.setBrush(QPalette.Disabled, QPalette.ButtonText, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Window, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush2)
#endif
        self.frame_12.setPalette(palette)
        self.frame_12.setStyleSheet(u"frame_12,QFrame{\n"
"	background-color: rgb(0, 0, 0)\n"
"}")
        self.frame_12.setFrameShape(QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QFrame.Raised)
        self.verticalLayout_8 = QVBoxLayout(self.frame_12)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(9, 9, -1, -1)
        self.th_img = QPushButton(self.frame_12)
        self.th_img.setObjectName(u"th_img")
        self.th_img.setFont(font)
        icon5 = QIcon()
        icon5.addFile(u":/icons/icons/edit-2.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.th_img.setIcon(icon5)
        self.th_img.setIconSize(QSize(32, 32))

        self.verticalLayout_8.addWidget(self.th_img)

        self.eng_img = QPushButton(self.frame_12)
        self.eng_img.setObjectName(u"eng_img")
        self.eng_img.setFont(font1)
        icon6 = QIcon()
        icon6.addFile(u":/icons/icons/type.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.eng_img.setIcon(icon6)
        self.eng_img.setIconSize(QSize(32, 32))

        self.verticalLayout_8.addWidget(self.eng_img)

        self.time_img = QPushButton(self.frame_12)
        self.time_img.setObjectName(u"time_img")
        self.time_img.setFont(font)
        icon7 = QIcon()
        icon7.addFile(u":/icons/icons/clock.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.time_img.setIcon(icon7)
        self.time_img.setIconSize(QSize(32, 32))

        self.verticalLayout_8.addWidget(self.time_img)

        self.num1 = QPushButton(self.frame_12)
        self.num1.setObjectName(u"num1")
        self.num1.setFont(font)
        icon8 = QIcon()
        icon8.addFile(u":/icons/icons/italic.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.num1.setIcon(icon8)
        self.num1.setIconSize(QSize(32, 32))

        self.verticalLayout_8.addWidget(self.num1)

        self.num2 = QPushButton(self.frame_12)
        self.num2.setObjectName(u"num2")
        self.num2.setFont(font)
        self.num2.setIcon(icon8)
        self.num2.setIconSize(QSize(32, 32))

        self.verticalLayout_8.addWidget(self.num2)

        self.day_img = QPushButton(self.frame_12)
        self.day_img.setObjectName(u"day_img")
        self.day_img.setFont(font)
        icon9 = QIcon()
        icon9.addFile(u":/icons/icons/calendar.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.day_img.setIcon(icon9)
        self.day_img.setIconSize(QSize(32, 32))

        self.verticalLayout_8.addWidget(self.day_img)

        self.gen_img = QPushButton(self.frame_12)
        self.gen_img.setObjectName(u"gen_img")
        self.gen_img.setFont(font)
        icon10 = QIcon()
        icon10.addFile(u":/icons/icons/user.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.gen_img.setIcon(icon10)
        self.gen_img.setIconSize(QSize(32, 32))

        self.verticalLayout_8.addWidget(self.gen_img)

        self.gen_img2 = QPushButton(self.frame_12)
        self.gen_img2.setObjectName(u"gen_img2")
        self.gen_img2.setFont(font)
        self.gen_img2.setIcon(icon10)
        self.gen_img2.setIconSize(QSize(32, 32))

        self.verticalLayout_8.addWidget(self.gen_img2)


        self.verticalLayout_7.addWidget(self.frame_12)

        self.toolBox.addItem(self.menu2, icon4, u"\u0e23\u0e39\u0e1b\u0e20\u0e32\u0e1e\u0e15\u0e31\u0e27\u0e2d\u0e22\u0e48\u0e32\u0e07")

        self.verticalLayout_4.addWidget(self.toolBox)

        self.newMo = QPushButton(self.frame_9)
        self.newMo.setObjectName(u"newMo")
        self.newMo.setFont(font)
        self.newMo.setStyleSheet(u"background-color: rgb(0, 0, 0);")
        icon11 = QIcon()
        icon11.addFile(u":/icons/icons/user-plus.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.newMo.setIcon(icon11)
        self.newMo.setIconSize(QSize(32, 32))

        self.verticalLayout_4.addWidget(self.newMo)


        self.verticalLayout_3.addWidget(self.frame_9)

        self.frame_10 = QFrame(self.side_menu)
        self.frame_10.setObjectName(u"frame_10")
        self.frame_10.setFrameShape(QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_5 = QHBoxLayout(self.frame_10)
        self.horizontalLayout_5.setSpacing(6)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 9, 6)
        self.pushButton = QPushButton(self.frame_10)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet(u"background-color: rgb(0, 0, 0);")
        icon12 = QIcon()
        icon12.addFile(u":/icons/icons/log-out.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton.setIcon(icon12)
        self.pushButton.setIconSize(QSize(32, 32))

        self.horizontalLayout_5.addWidget(self.pushButton)


        self.verticalLayout_3.addWidget(self.frame_10, 0, Qt.AlignBottom)


        self.verticalLayout_2.addWidget(self.side_menu)


        self.horizontalLayout.addWidget(self.sid_menu_container)

        self.main_body = QFrame(self.centralwidget)
        self.main_body.setObjectName(u"main_body")
        self.main_body.setFrameShape(QFrame.StyledPanel)
        self.main_body.setFrameShadow(QFrame.Raised)
        self.verticalLayout = QVBoxLayout(self.main_body)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.header_frame = QFrame(self.main_body)
        self.header_frame.setObjectName(u"header_frame")
        self.header_frame.setBaseSize(QSize(0, 0))
        font3 = QFont()
        font3.setPointSize(11)
        self.header_frame.setFont(font3)
        self.header_frame.setStyleSheet(u"background-color: rgb(0, 0, 0);")
        self.header_frame.setFrameShape(QFrame.StyledPanel)
        self.header_frame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.header_frame)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setSizeConstraint(QLayout.SetFixedSize)
        self.horizontalLayout_2.setContentsMargins(0, 4, 0, 0)
        self.frame_2 = QFrame(self.header_frame)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.verticalLayout_10 = QVBoxLayout(self.frame_2)
        self.verticalLayout_10.setSpacing(0)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.verticalLayout_10.setContentsMargins(10, 0, 500, 10)
        self.pushButton_16 = QPushButton(self.frame_2)
        self.pushButton_16.setObjectName(u"pushButton_16")
        self.pushButton_16.setIconSize(QSize(50, 16))

        self.verticalLayout_10.addWidget(self.pushButton_16)


        self.horizontalLayout_2.addWidget(self.frame_2)

        self.frame_4 = QFrame(self.header_frame)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setFrameShape(QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Raised)
        self.verticalLayout_9 = QVBoxLayout(self.frame_4)
        self.verticalLayout_9.setSpacing(0)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.pushButton_15 = QPushButton(self.frame_4)
        self.pushButton_15.setObjectName(u"pushButton_15")

        self.verticalLayout_9.addWidget(self.pushButton_15)


        self.horizontalLayout_2.addWidget(self.frame_4)

        self.frame = QFrame(self.header_frame)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_6 = QHBoxLayout(self.frame)
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.pushButton_13 = QPushButton(self.frame)
        self.pushButton_13.setObjectName(u"pushButton_13")
        icon13 = QIcon()
        icon13.addFile(u":/icons/icons/loader.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton_13.setIcon(icon13)
        self.pushButton_13.setIconSize(QSize(22, 22))

        self.horizontalLayout_6.addWidget(self.pushButton_13)

        self.pushButton_12 = QPushButton(self.frame)
        self.pushButton_12.setObjectName(u"pushButton_12")
        icon14 = QIcon()
        icon14.addFile(u":/icons/icons/maximize-2.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton_12.setIcon(icon14)
        self.pushButton_12.setIconSize(QSize(22, 22))

        self.horizontalLayout_6.addWidget(self.pushButton_12, 0, Qt.AlignHCenter)

        self.pushButton_11 = QPushButton(self.frame)
        self.pushButton_11.setObjectName(u"pushButton_11")
        icon15 = QIcon()
        icon15.addFile(u":/icons/icons/x.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton_11.setIcon(icon15)
        self.pushButton_11.setIconSize(QSize(22, 22))

        self.horizontalLayout_6.addWidget(self.pushButton_11, 0, Qt.AlignHCenter)


        self.horizontalLayout_2.addWidget(self.frame)


        self.verticalLayout.addWidget(self.header_frame, 0, Qt.AlignTop)

        self.main_body_conten = QFrame(self.main_body)
        self.main_body_conten.setObjectName(u"main_body_conten")
        self.main_body_conten.setFrameShape(QFrame.StyledPanel)
        self.main_body_conten.setFrameShadow(QFrame.Raised)
        self.verticalLayout_11 = QVBoxLayout(self.main_body_conten)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.label_3 = QLabel(self.main_body_conten)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setMinimumSize(QSize(480, 400))
        self.label_3.setScaledContents(True)
        self.label_3.setAlignment(Qt.AlignCenter)

        self.verticalLayout_11.addWidget(self.label_3)

        self.showtext = QTextBrowser(self.main_body_conten)
        self.showtext.setObjectName(u"showtext")
        font4 = QFont()
        font4.setFamily(u"TH Mali Grade 6")
        font4.setPointSize(45)
        font4.setBold(True)
        font4.setWeight(75)
        self.showtext.setFont(font4)
        self.showtext.setStyleSheet(u"border:3px solid rgb(230,5,64);\n"
"background-color: rgb(0, 0, 0);\n"
"")

        self.verticalLayout_11.addWidget(self.showtext)


        self.verticalLayout.addWidget(self.main_body_conten)

        self.frame_6 = QFrame(self.main_body)
        self.frame_6.setObjectName(u"frame_6")
        self.frame_6.setMaximumSize(QSize(16777215, 500))
        self.frame_6.setFrameShape(QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_7 = QHBoxLayout(self.frame_6)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.playSo = QPushButton(self.frame_6)
        self.playSo.setObjectName(u"playSo")
        self.playSo.setStyleSheet(u"border:3px solid rgb(230, 5, 64);\n"
"border-radius:20px\n"
"")
        icon16 = QIcon()
        icon16.addFile(u":/icons/icons/volume-2.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.playSo.setIcon(icon16)
        self.playSo.setIconSize(QSize(32, 32))

        self.horizontalLayout_7.addWidget(self.playSo)

        self.textBrowser = QTextBrowser(self.frame_6)
        self.textBrowser.setObjectName(u"textBrowser")
        self.textBrowser.setMaximumSize(QSize(16777215, 500))
        self.textBrowser.setFont(font4)
        self.textBrowser.setStyleSheet(u"border:3px solid rgb(230,5,64);\n"
"background-color: rgb(0, 0, 0);\n"
"")

        self.horizontalLayout_7.addWidget(self.textBrowser)


        self.verticalLayout.addWidget(self.frame_6)

        self.frame_3 = QFrame(self.main_body)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setFrameShape(QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.frame_3)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.frame_7 = QFrame(self.frame_3)
        self.frame_7.setObjectName(u"frame_7")
        self.frame_7.setFrameShape(QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_8 = QHBoxLayout(self.frame_7)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.mic = QPushButton(self.frame_7)
        self.mic.setObjectName(u"mic")
        self.mic.setStyleSheet(u"border:3px solid rgb(230, 5, 64);\n"
"border-radius:20px\n"
"")
        self.mic.setText(u"")
        icon17 = QIcon()
        icon17.addFile(u":/icons/icons/mic.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.mic.setIcon(icon17)
        self.mic.setIconSize(QSize(32, 32))

        self.horizontalLayout_8.addWidget(self.mic)

        self.putText = QLineEdit(self.frame_7)
        self.putText.setObjectName(u"putText")
        self.putText.setMaximumSize(QSize(600, 70))
        font5 = QFont()
        font5.setPointSize(22)
        self.putText.setFont(font5)
        self.putText.setStyleSheet(u"lineEdit,QLineEdit{\n"
"	background-color: rgb(0, 0, 0);\n"
"	border-bottom:3px solid rgb(230,5,64);\n"
"}")

        self.horizontalLayout_8.addWidget(self.putText)


        self.horizontalLayout_3.addWidget(self.frame_7)


        self.verticalLayout.addWidget(self.frame_3)


        self.horizontalLayout.addWidget(self.main_body)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.toolBox.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"SignLanguageTranslation", None))
        self.label_2.setText("")
        self.thai.setText(QCoreApplication.translate("MainWindow", u"\u0e2b\u0e21\u0e27\u0e14\u0e2d\u0e31\u0e01\u0e29\u0e23\u0e20\u0e32\u0e29\u0e32\u0e44\u0e17\u0e22", None))
        self.eng.setText(QCoreApplication.translate("MainWindow", u"\u0e2b\u0e21\u0e27\u0e14\u0e2d\u0e31\u0e01\u0e29\u0e23\u0e20\u0e32\u0e29\u0e32\u0e2d\u0e31\u0e07\u0e01\u0e24\u0e29", None))
        self.time.setText(QCoreApplication.translate("MainWindow", u"\u0e2b\u0e21\u0e27\u0e14\u0e40\u0e27\u0e25\u0e32", None))
        self.general.setText(QCoreApplication.translate("MainWindow", u"\u0e2b\u0e21\u0e27\u0e14\u0e04\u0e33\u0e17\u0e31\u0e48\u0e27\u0e44\u0e1b", None))
        self.day.setText(QCoreApplication.translate("MainWindow", u"\u0e2b\u0e21\u0e27\u0e14\u0e27\u0e31\u0e19\u0e40\u0e14\u0e37\u0e2d\u0e19\u0e1b\u0e35", None))
        self.num.setText(QCoreApplication.translate("MainWindow", u"\u0e2b\u0e21\u0e27\u0e14\u0e15\u0e31\u0e27\u0e40\u0e25\u0e02\u0e08\u0e33\u0e19\u0e27\u0e19\u0e19\u0e31\u0e1a", None))
        self.deny.setText(QCoreApplication.translate("MainWindow", u"\u0e2b\u0e21\u0e27\u0e14\u0e1b\u0e23\u0e30\u0e42\u0e22\u0e04\u0e1b\u0e0e\u0e34\u0e40\u0e2a\u0e18", None))
        self.question.setText(QCoreApplication.translate("MainWindow", u"\u0e2b\u0e21\u0e27\u0e14\u0e1b\u0e23\u0e30\u0e42\u0e22\u0e04\u0e04\u0e33\u0e16\u0e32\u0e21", None))
        self.sentence.setText(QCoreApplication.translate("MainWindow", u"\u0e2b\u0e21\u0e27\u0e14\u0e1b\u0e23\u0e30\u0e42\u0e22\u0e04\u0e17\u0e31\u0e48\u0e27\u0e44\u0e1b", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.menu1), QCoreApplication.translate("MainWindow", u"\u0e2b\u0e21\u0e27\u0e14\u0e04\u0e33-\u0e1b\u0e23\u0e30\u0e42\u0e22\u0e04", None))
        self.th_img.setText(QCoreApplication.translate("MainWindow", u"\u0e20\u0e32\u0e29\u0e32\u0e44\u0e17\u0e22", None))
        self.eng_img.setText(QCoreApplication.translate("MainWindow", u"\u0e15\u0e31\u0e27\u0e2d\u0e31\u0e01\u0e29\u0e23\u0e20\u0e32\u0e29\u0e32\u0e2d\u0e31\u0e07\u0e01\u0e24\u0e29 ", None))
        self.time_img.setText(QCoreApplication.translate("MainWindow", u"\u0e40\u0e27\u0e25\u0e32", None))
        self.num1.setText(QCoreApplication.translate("MainWindow", u"\u0e15\u0e31\u0e27\u0e40\u0e25\u0e02 1", None))
        self.num2.setText(QCoreApplication.translate("MainWindow", u"\u0e15\u0e31\u0e27\u0e40\u0e25\u0e02 2", None))
        self.day_img.setText(QCoreApplication.translate("MainWindow", u"\u0e27\u0e31\u0e19 \u0e40\u0e14\u0e37\u0e2d\u0e19 \u0e1b\u0e35", None))
        self.gen_img.setText(QCoreApplication.translate("MainWindow", u"\u0e04\u0e33\u0e17\u0e31\u0e48\u0e27\u0e44\u0e1b1", None))
        self.gen_img2.setText(QCoreApplication.translate("MainWindow", u"\u0e04\u0e33\u0e17\u0e31\u0e48\u0e27\u0e44\u0e1b2", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.menu2), QCoreApplication.translate("MainWindow", u"\u0e23\u0e39\u0e1b\u0e20\u0e32\u0e1e\u0e15\u0e31\u0e27\u0e2d\u0e22\u0e48\u0e32\u0e07", None))
        self.newMo.setText(QCoreApplication.translate("MainWindow", u"\u0e17\u0e48\u0e32\u0e43\u0e2b\u0e21\u0e48", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.pushButton_16.setText("")
        self.pushButton_15.setText("")
        self.pushButton_13.setText("")
        self.pushButton_12.setText("")
        self.pushButton_11.setText("")
        self.label_3.setText("")
        self.playSo.setText("")
        self.putText.setPlaceholderText(QCoreApplication.translate("MainWindow", u"       Text", None))
    # retranslateUi

