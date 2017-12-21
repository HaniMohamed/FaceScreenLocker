#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:27:56 2017

@author: hanihussein
"""

import cv2
import sqlite3
from threading import Timer
import os

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
faceObj = cv2.face.LBPHFaceRecognizer_create()
rec = faceObj
print(rec.getThreshold())


rec.read("recognizer/trainingData.yml")
font = cv2.FONT_HERSHEY_SIMPLEX


def lock_screen():
    os.system("gnome-screensaver-command -l")

def close():
    cam.release()
    cv2.destroyAllWindows()

def get_profile(id):
    conn = sqlite3.connect("FaceBase")
    cmd = "Select * FROM People WHERE ID = " + str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

t = Timer(20.0, lock_screen)
t.start()


while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 2, 5)
    Id = 0
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        Id = 0
        Id, conf = rec.predict(gray[y:y + h, x:x + w])
        profile = get_profile(Id)
        if(Id == int(profile[0])):
            print(profile[1])
            if (profile[1]=="Hani"):
                print("ٌ##-RESET")
                t.cancel()
                t.start()
                
                
       
    cv2.imshow("Face", img)
    if(cv2.waitKey(1) == ord('q')):
        close()


