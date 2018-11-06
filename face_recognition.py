import numpy as np
import cv2
import sys


face_cascade = cv2.CascadeClassifier( r"E:\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"E:\opencv-master\data\haarcascades\haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(r"E:\opencv-master\data\haarcascades\haarcascade_smile.xml")

##img = cv2.imread(r'C:\Users\SUPRATIK\Pictures\2017-05\two_friends.jpg')
##print (type(image))

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if(len(faces) != 0):
        for (x,y,w,h) in faces:
                 cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                 #print(img)
                 roi_gray = gray[y:y+h, x:x+w]
                 roi_color = frame[y:y+h, x:x+w]
                 eyes = eye_cascade.detectMultiScale(roi_gray)
                 smiles = smile_cascade.detectMultiScale(roi_gray,scaleFactor=1.3,
                                                     minNeighbors=22,
                                                     minSize=(25,25))
                 for (ex,ey,ew,eh) in eyes:
                     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

                 for (ex,ey,ew,eh) in smiles:
                     print("found smile ",end="")
                     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,250),2)
                 cv2.imshow('Test image',frame)
    else:
        cv2.imshow('Test image',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()
exit(1)
