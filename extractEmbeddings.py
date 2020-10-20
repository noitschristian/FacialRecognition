#------------------------------------------------------------------#
#   Christian Pennington - R11445847
#   extractEmbeddings.py for faceRecognition project
#   Description: Program to automatically extract the 128-D facial
#       embeddings of each image in the dataset and stores them
#       under the specified person.
#
#   Date: 10/6/2020
#
#   Command Line arguments:
#   -i, --dataset, = Dataset
#   -e, --embeddings, =output/embeddings.pickle
#   -d, --detector, =face_detection_model
#   -m, --embedding-model, =openface_nn4.small2.v1.t7
#-----------------------------------------------------------------#
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os
import argparse
#------------------------------------------------------------------#
# --Construct argument parser--#
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
                help="path to input directory of faces")
ap.add_argument("-e", "--embeddings", required=True,
                help="output path to serialized embeddings")
ap.add_argument("-d", "--detector", required=True,
                help="path to OpenCV face detector")
ap.add_argument("-m", "--embedding-model", required=True,
                help="Path to OpenCV face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
#------------------------------------------------------------------#
#--Load dataset directory--#
imagePaths = list(paths.list_images(args["dataset"]))
# --Load face detector--#
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# --load serialized faces into embedding model--#
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# --Initialize list of extracted facial embeddings and names--#
knownEmbeddings = []
knownNames = []
totalFaces = 0

for (i, imagePaths) in enumerate(imagePaths):

    name = imagePaths.split(os.path.sep)[-2]
    print("> Processing image {}/{} for {}".format(i + 1, len(imagePaths), name))

    # --load image, resize, and grab dimensions--#
    img = cv2.imread(imagePaths)
    img = imutils.resize(img, width=600)
    (h, w) = img.shape[:2]

    # --construct blob from image--#
    imgBlob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                    1.0, (300, 300), (104, 177, 123),
                                    swapRB=False, crop=False)

    # --Apply face detectors to models--#
    detector.setInput(imgBlob)
    detections = detector.forward()

    # --Process Detections in blob--#
    if len(detections) > 0:  # Ensure one face was found
        # --Assume every image has one face, find the largest box--#
        k = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, k, 2]

        # --Ensure detection is using largest probability and filters out weak detections
        if confidence > args["confidence"]:
            # Compute bounding box around face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract face AOI and AOI dimensions
            face = img[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            # --Construct blob from face AOI then pass blob through 128-d quantification
            faceBlob = cv2.dnn.blobFromImage(face, 1 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vac = embedder.forward()

            knownNames.append(name)
            knownEmbeddings.append(vac.flatten())
            totalFaces += 1
#-------------------------------------------------------------------------------#
# --dump facial recognitions and names to disk--#
print("> Serializing {} encodings..".format(totalFaces))
data = {"Embeddings": knownEmbeddings, "Names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()
print("> Embeddings extracted.")
