import cv2
from random import randrange

# Load some pre-trained data on face frontalsfrom opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# # Choose an image to detect faces in
# img = cv2.imread('3.JPG')

# To capture video from webcam
webcam = cv2.VideoCapture(0)

while True:
    # Read the cureent frame
    succesful_frame_read, frame = webcam.read()


    # Must convert to color
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    print(face_coordinates)

    # Draw rectangle around the face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 2)

    cv2.imshow('Swapnil Sharma', frame)
    key = cv2.waitKey(1)

    # Stop if Q is pressed
    if key == 81 or key ==113:
        break

# Face Recognition Software using Machine Learning
