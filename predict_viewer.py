# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/predict.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

import sys
sys.path.insert(1, '/Users/zhenningyang/Documents/opencv_moving_box/src')
from darknet_func import *

class Ui_MainWindow(object):

    def __init__(self):
        # load model
        self.net = load_net(b"/Users/zhenningyang/Documents/darknet/cfg/yolov3_ssig.cfg",
                       b"/Users/zhenningyang/Documents/yolov3_weights/yolov3_ssig_final.weights", 0)
        self.meta = load_meta(b"/Users/zhenningyang/Documents/opencv_moving_box/cfg/ssig.data")

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.prediction_image = QtWidgets.QLabel(self.centralwidget)
        self.prediction_image.setGeometry(QtCore.QRect(0, 0, 801, 451))
        self.prediction_image.setText("")
        self.prediction_image.setPixmap(QtGui.QPixmap("data/car detection.png"))
        self.prediction_image.setScaledContents(True)
        self.prediction_image.setObjectName("prediction_image")
        self.predict_button = QtWidgets.QPushButton(self.centralwidget)
        self.predict_button.setGeometry(QtCore.QRect(340, 490, 113, 32))
        self.predict_button.setObjectName("predict_button")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.predict_button.clicked.connect(self.show_pred)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.predict_button.setText(_translate("MainWindow", "predict"))

    def show_pred(self):
        r = detect(self.net, self.meta, bytes("data/car detection.png", 'utf-8'))
        img = draw("{}".format("data/car detection.png"), r)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('prediction.png',img)
        self.prediction_image.setPixmap(QtGui.QPixmap('prediction.png'))

if __name__ == "__main__":
    #import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
