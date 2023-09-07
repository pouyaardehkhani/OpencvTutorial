import numpy as np
import cv2 
import matplotlib.pyplot as plt

solvay = cv2.imread('solvay_conference.jpg',0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def adj_detect_face(img, scaleFactor=1.2, minNeighbors=5):
    
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img,
                                               scaleFactor=scaleFactor, 
                                               minNeighbors=minNeighbors) 
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 
        
    return face_img
    
result = adj_detect_face(solvay, scaleFactor=1.2, minNeighbors=5)
plt.imshow(result,cmap='gray')