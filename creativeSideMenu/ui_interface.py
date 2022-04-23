# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'interfaceNRlEAw.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from Custom_Widgets.Widgets import QCustomSlideMenu

import resources_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(633, 351)
        MainWindow.setTabShape(QTabWidget.Rounded)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setStyleSheet(u"*{\n"
"	border: none;\n"
"	color: #fff;\n"
"}\n"
"#centralwidget{\n"
"	background-color: #09102a;\n"
"}\n"
"#sideMenuMainCont, #header, #mainBody, #widget, #notification > QFrame {\n"
"	background-color: #171d3c;\n"
"	border-radius: 20px;\n"
"}\n"
"QPushButton{\n"
"	text-align: left;\n"
"	background-color: #08112a;\n"
"	padding: 10px 2px;\n"
"	border-radius: 10px;\n"
"}\n"
"QLineEdit{\n"
"	padding: 5px;\n"
"	background: transparent;\n"
"}\n"
"#searchInput{\n"
"	border-radius: 20px;\n"
"	border: 2px solid #fd7012;\n"
"}\n"
"#pushButton_8, #pushButton{\n"
"	background: none;\n"
"	padding: 0;\n"
"}")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.sideMenuMainCont = QCustomSlideMenu(self.centralwidget)
        self.sideMenuMainCont.setObjectName(u"sideMenuMainCont")
        self.sideMenuMainCont.setMaximumSize(QSize(40, 16777215))
        self.verticalLayout = QVBoxLayout(self.sideMenuMainCont)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.widget = QWidget(self.sideMenuMainCont)
        self.widget.setObjectName(u"widget")
        self.verticalLayout_7 = QVBoxLayout(self.widget)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(6, 10, 10, 10)
        self.frame = QFrame(self.widget)
        self.frame.setObjectName(u"frame")
        self.frame.setMinimumSize(QSize(0, 30))
        self.frame.setMaximumSize(QSize(16777215, 30))
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_6 = QVBoxLayout(self.frame)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.pushButton = QPushButton(self.frame)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setMinimumSize(QSize(0, 30))
        icon = QIcon()
        icon.addFile(u":/icons/icons/align-justify.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton.setIcon(icon)
        self.pushButton.setIconSize(QSize(30, 30))

        self.verticalLayout_6.addWidget(self.pushButton)


        self.verticalLayout_7.addWidget(self.frame, 0, Qt.AlignTop)

        self.sideMenuSubCont = QWidget(self.widget)
        self.sideMenuSubCont.setObjectName(u"sideMenuSubCont")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sideMenuSubCont.sizePolicy().hasHeightForWidth())
        self.sideMenuSubCont.setSizePolicy(sizePolicy)
        self.sideMenuSubCont.setMinimumSize(QSize(200, 301))
        self.sideMenuSubCont.setMaximumSize(QSize(16777215, 16777215))
        self.verticalLayout_2 = QVBoxLayout(self.sideMenuSubCont)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.frame_2 = QFrame(self.sideMenuSubCont)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setMinimumSize(QSize(200, 301))
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.verticalLayout_8 = QVBoxLayout(self.frame_2)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.microphone = QPushButton(self.frame_2)
        self.microphone.setObjectName(u"microphone")
        icon1 = QIcon()
        icon1.addFile(u":/icons/icons/mic.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.microphone.setIcon(icon1)
        self.microphone.setIconSize(QSize(32, 32))

        self.verticalLayout_8.addWidget(self.microphone)

        self.opencamera = QPushButton(self.frame_2)
        self.opencamera.setObjectName(u"opencamera")
        icon2 = QIcon()
        icon2.addFile(u":/icons/icons/camera.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.opencamera.setIcon(icon2)
        self.opencamera.setIconSize(QSize(32, 32))

        self.verticalLayout_8.addWidget(self.opencamera)

        self.addSing = QPushButton(self.frame_2)
        self.addSing.setObjectName(u"addSing")
        icon3 = QIcon()
        icon3.addFile(u":/icons/icons/download.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.addSing.setIcon(icon3)
        self.addSing.setIconSize(QSize(32, 32))

        self.verticalLayout_8.addWidget(self.addSing)

        self.train = QPushButton(self.frame_2)
        self.train.setObjectName(u"train")
        icon4 = QIcon()
        icon4.addFile(u":/icons/icons/activity.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.train.setIcon(icon4)
        self.train.setIconSize(QSize(32, 32))

        self.verticalLayout_8.addWidget(self.train)

        self.Exit = QPushButton(self.frame_2)
        self.Exit.setObjectName(u"Exit")
        icon5 = QIcon()
        icon5.addFile(u":/icons/icons/log-out.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.Exit.setIcon(icon5)
        self.Exit.setIconSize(QSize(32, 32))

        self.verticalLayout_8.addWidget(self.Exit)


        self.verticalLayout_2.addWidget(self.frame_2, 0, Qt.AlignTop)


        self.verticalLayout_7.addWidget(self.sideMenuSubCont, 0, Qt.AlignTop)


        self.verticalLayout.addWidget(self.widget, 0, Qt.AlignTop)


        self.horizontalLayout.addWidget(self.sideMenuMainCont)

        self.widget_2 = QWidget(self.centralwidget)
        self.widget_2.setObjectName(u"widget_2")
        self.widget_2.setMaximumSize(QSize(16777215, 16777215))
        self.widget_2.setBaseSize(QSize(0, 0))
        self.verticalLayout_3 = QVBoxLayout(self.widget_2)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.notification = QCustomSlideMenu(self.widget_2)
        self.notification.setObjectName(u"notification")
        self.horizontalLayout_4 = QHBoxLayout(self.notification)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")

        self.verticalLayout_3.addWidget(self.notification, 0, Qt.AlignHCenter)

        self.header = QWidget(self.widget_2)
        self.header.setObjectName(u"header")
        self.header.setMaximumSize(QSize(600, 500))
        self.horizontalLayout_2 = QHBoxLayout(self.header)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.searchInput = QFrame(self.header)
        self.searchInput.setObjectName(u"searchInput")
        self.searchInput.setEnabled(True)
        self.searchInput.setMinimumSize(QSize(550, 0))
        self.searchInput.setMaximumSize(QSize(0, 100))
        self.searchInput.setFrameShape(QFrame.StyledPanel)
        self.searchInput.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.searchInput)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.textBrowser = QTextBrowser(self.searchInput)
        self.textBrowser.setObjectName(u"textBrowser")
        self.textBrowser.setMaximumSize(QSize(900, 50))
        palette = QPalette()
        brush = QBrush(QColor(255, 255, 255, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.WindowText, brush)
        palette.setBrush(QPalette.Active, QPalette.Text, brush)
        palette.setBrush(QPalette.Active, QPalette.ButtonText, brush)
        brush1 = QBrush(QColor(23, 29, 60, 255))
        brush1.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Base, brush1)
        brush2 = QBrush(QColor(255, 255, 255, 128))
        brush2.setStyle(Qt.NoBrush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Active, QPalette.PlaceholderText, brush2)
#endif
        palette.setBrush(QPalette.Inactive, QPalette.WindowText, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Text, brush)
        palette.setBrush(QPalette.Inactive, QPalette.ButtonText, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Base, brush1)
        brush3 = QBrush(QColor(255, 255, 255, 128))
        brush3.setStyle(Qt.NoBrush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush3)
#endif
        palette.setBrush(QPalette.Disabled, QPalette.WindowText, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Text, brush)
        palette.setBrush(QPalette.Disabled, QPalette.ButtonText, brush)
        brush4 = QBrush(QColor(240, 240, 240, 255))
        brush4.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Disabled, QPalette.Base, brush4)
        brush5 = QBrush(QColor(255, 255, 255, 128))
        brush5.setStyle(Qt.NoBrush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush5)
#endif
        self.textBrowser.setPalette(palette)
        font = QFont()
        font.setFamily(u"TH Krub")
        font.setPointSize(27)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.textBrowser.setFont(font)

        self.horizontalLayout_3.addWidget(self.textBrowser)


        self.horizontalLayout_2.addWidget(self.searchInput)


        self.verticalLayout_3.addWidget(self.header)

        self.widget_4 = QWidget(self.widget_2)
        self.widget_4.setObjectName(u"widget_4")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.widget_4.sizePolicy().hasHeightForWidth())
        self.widget_4.setSizePolicy(sizePolicy1)
        self.verticalLayout_4 = QVBoxLayout(self.widget_4)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.mainBody = QWidget(self.widget_4)
        self.mainBody.setObjectName(u"mainBody")
        self.verticalLayout_5 = QVBoxLayout(self.mainBody)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.label_3 = QLabel(self.mainBody)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.label_3)

        self.label_6 = QLabel(self.mainBody)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setMaximumSize(QSize(16777214, 16777215))
        self.label_6.setAlignment(Qt.AlignCenter)
        self.label_6.setWordWrap(True)

        self.verticalLayout_5.addWidget(self.label_6)


        self.verticalLayout_4.addWidget(self.mainBody)


        self.verticalLayout_3.addWidget(self.widget_4)


        self.horizontalLayout.addWidget(self.widget_2)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.pushButton.setText("")
        self.microphone.setText(QCoreApplication.translate("MainWindow", u"Investments", None))
        self.opencamera.setText(QCoreApplication.translate("MainWindow", u"Home", None))
        self.addSing.setText(QCoreApplication.translate("MainWindow", u"Messages", None))
        self.train.setText(QCoreApplication.translate("MainWindow", u"Reports", None))
        self.Exit.setText(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.textBrowser.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'TH Krub'; font-size:27pt; font-weight:600; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:'MS Shell Dlg 2'; font-size:26pt; font-weight:400;\"><br /></p></body></html>", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:28pt;\">The Sign Language</span></p></body></html>", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:28pt; font-weight:600; color:#ffffff;\">TRANSLATION</span></p></body></html>", None))
    # retranslateUi

