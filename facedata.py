import numpy as np
import cv2

#coder:Hackno
#source algorithm: pythonprogramming.net
#v 1.0

yuz_kaynak = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

goz_kaynak = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yuzler = yuz_kaynak.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in yuzler:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        gozler = goz_kaynak.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in gozler:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    a = cv2.waitKey(30) & 0xff
    if a == 27:
        break

cap.release()
cv2.destroyAllWindows()
