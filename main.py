# Install modules
# pip install opencv-python-headless


# Import modules
import cv2
from random import randrange

# Load pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Image to detect faces
img = cv2.imread('people.jpg')

# Make grayscale images
grayscale_img =cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) 

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

# Draw rectangle around faces
for (x,y,w,h) in face_coordinates:
  cv2.rectangle(img , (x,y) , (x+w , y+h) , (randrange(256), randrange(256) , randrange(256)) , 2)

cv2.imshow('Face detection',img)
cv2.waitKey()

