#Example python script to test threading of video inputs
from threading import Thread
import cv2
import time
import numpy as np

# Initialize everything for object detection
proto = 'MobileNetSSD_deploy.prototxt'
model = 'MobileNetSSD_deploy.caffemodel'
confidenceTest = .2

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# initialize the consider set (class labels we care about and want
# to count), the object count dictionary, and the frame  dictionary
CONSIDER = set(["person", "car", "bicycle", "bus"])
objCount = {obj: 0 for obj in CONSIDER}

# load our serialized model from disk
print("> loading model...")
net = cv2.dnn.readNetFromCaffe(proto, model)

class ThreadCamera(object):
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # FPS = 1/X; X = desired FPS
        self.FPS = 1/30 #start with 10 fps
        self.FPS_MS = int(self.FPS * 1000)

        #Start frame retrieval thread
        self.thread = Thread(target = self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)

    def rover_guide(self):
        (w,h,c) = self.frame.shape
        blob = cv2.dnn.blobFromImage(cv2.resize(self.frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # reset the object count for each object in the CONSIDER set
        objCount = {obj: 0 for obj in CONSIDER}

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > confidenceTest:
                # extract the index of the class label from the
                # detections
                idx = int(detections[0, 0, i, 1])

                # check to see if the predicted class is in the set of
                # classes that need to be considered
                if CLASSES[idx] in CONSIDER:
                    # increment the count of the particular object
                    # detected in the frame
                    objCount[CLASSES[idx]] += 1

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # draw the bounding box around the detected object on
                    # the frame
                    cv2.rectangle(self.frame, (startX, startY), (endX, endY),
                                  (255, 0, 0), 2)
                    text = "{}".format(CLASSES[idx])
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(self.frame, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    print("Object detected: {}".format(CLASSES[idx]))

    def show_frame(self):
        cv2.rectangle(self.frame, (0, 0), (72, 240), (0, 0, 255), 2)
        cv2.rectangle(self.frame, (73, 0), (144, 240), (0, 0, 255), 2)
        cv2.rectangle(self.frame, (145, 0), (216, 240), (0, 0, 255), 2)
        cv2.rectangle(self.frame, (217, 0), (288, 240), (0, 0, 255), 2)
        cv2.rectangle(self.frame, (289, 0), (360, 240), (0, 0, 255), 2)
        cv2.imshow('frame', self.frame)
        cv2.waitKey(self.FPS_MS)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(">Starting video stream..")
    #src = 0
    src = 'http://bugge.ngrok.io/stream/video.mjpeg'

    threaded_cam = ThreadCamera(src)
    while True:
        try:
            threaded_cam.show_frame()
            threaded_cam.rover_guide()
        except AttributeError:
            pass
