#------------------------------------------------------------------#
#   Christian Pennington - R11445847
#   trainModel.py for faceRecognition project
#   Description: Program to train the scikit-learn Support
#       Vector Machine (SVM), a machine learning model. Takes serialized
#       embeddings and labels them under pickle files. Both the SVM
#		and label encoding are saved for later.
#
#   Date: 10/6/2020
#   Command Line arguments:
#   -e, --embeddings, =output/embeddings.pickle
#   -r, --recognizer, =output/recognizer.pickle
#   -m, --le, =output/le.pickle
#-----------------------------------------------------------------#
# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

# load the face embeddings
print("> Loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())
print(type(data))

# encode the labels
print("> Encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["Names"])

# train the model used to accept the 128-d embeddings of the face
print("> Training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["Embeddings"], labels)

# write the actual face recognition model to disk
print("> Writing to disk..")
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
print("> Label encoder...")
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

print("> Training complete.")