import sys
import tkinter
from tkinter import Frame, Tk, BOTH, Text, Menu, END
from tkinter.filedialog import Open, SaveAs

from model import create_model
import numpy as np
import os.path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib

from align import AlignDlib
from sklearn.svm import LinearSVC

import cv2

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),    
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

alignment = AlignDlib('./models/shape_predictor_68_face_landmarks.dat')

nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('./weights/nn4.small2.v1.h5')

svc = joblib.load('svc.pkl')

mydict = ['BanAnh','BanDat','BanDung','BanDuong','BanGiang','BanHuyen','BanLong','BanNhat','BanPhu','BanPhuoc','BanThanh','BanThy','BanVu','DinhTriem','MinhQuan','NgocTan', 'NhatLinh','PhuThinh','PhuongThao','ThanhCong','ThanhSang','ThayDuc']

class Main(Frame):
    
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()
  
    def initUI(self):
        self.parent.title("Nhan Dang Khuon Mat")
        self.pack(fill=BOTH, expand=1)
  
        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)
  
        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Open", command=self.onOpen)
        fileMenu.add_command(label="Recognition", command=self.onRecognition)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=fileMenu)
        self.txt = Text(self)
        self.txt.pack(fill=BOTH, expand=1)
  
    def onOpen(self):
        global ftypes
        ftypes = [('Images', '*.jpg *.tif *.bmp *.gif *.png')]
        dlg = Open(self, filetypes = ftypes)
        fl = dlg.show()
  
        if fl != '':
            global img
            global imgin
            imgin = cv2.imread(fl,cv2.IMREAD_COLOR)
            img = imgin[...,::-1]
            cv2.namedWindow("ImageIn", cv2.WINDOW_AUTOSIZE)
            #cv2.moveWindow("ImageIn", 200, 200)
            cv2.imshow("ImageIn", imgin)

    def onRecognition(self):
        
        img_test = align_image(img)
        
        # scale RGB values to interval [0,1]
        img_test = (img_test/255.).astype(np.float)
        # obtain embedding vector for image
        embedded_test = nn4_small2_pretrained.predict(np.expand_dims(img_test, axis=0))[0]
        test_prediction = svc.predict([embedded_test])
        result = mydict[test_prediction[0]]
        cv2.putText(imgin,result,(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
        cv2.namedWindow("ImageIn", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageIn", imgin)

root = Tk()
Main(root)
root.geometry("480x480+100+100")
root.mainloop()


