import cv2
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#change face.png to the name of the desired png file. Should be in the same directory
img = cv2.imread('face.png')
grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),10)
cv2.imshow('Face detector',img)
cv2.waitKey()