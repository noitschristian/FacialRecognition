#----------------------------------------------------------------#
#   Christian Pennington - R11445847
#   faceCapture.py for faceRecognition project
#   Description: Uses webcam to detect a face in the stream,
#       and then use 'c' to take a picture to be used for facial recognition
#
#   Date: 10/6/2020
#-----------------------------------------------------------------#
import os
import time
import cv2
# ---------------------------------------------------------------#
print("Who am I looking at? Please select user:\n")
print("1. Christian\n")
print("2. Kevin\n")
print("3. Jon\n")
path_var = input("Please enter option: ")

if path_var == "1":
    name = "Christian"
    path = 'Dataset/Christian'
elif path_var == "2":
    name = "Kevin"
    path = 'Dataset/Kevin'
elif path_var == "3":
    name = "Jon"
    path = 'Dataset/Jon'
else:
    name = "Unknown"
    path = 'C:/Users/chris/Documents/faceRecognition/Dataset/Unknown'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Initializing video stream")
cam = cv2.VideoCapture(0)
# 'http://129.118.19.74:8000/stream.mjpg'for streaming from Pi

time.sleep(1.0)  # Two second load tme
total = 0
#-----------------------------------------------------------------------#
while True:
    # Grab frame from video, make
    ret, frame = cam.read()
    orig = frame.copy()
    if not ret:
        print("Capture not opened")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray)

    # Detect faces in grayscale
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show output frames
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        print("Closing program")
        break

    elif key == ord("c"):
        img_name = "{}.png".format(total)
        cv2.imwrite(os.path.join(path, img_name), orig)
        print("image saved.")
        total += 1
#-------------------------------------------------------------------#
# print info
print("{} face images stored".format(total))
print("cleaning up space...")
cv2.destroyAllWindows()