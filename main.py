import cv2
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import pytesseract
from imutils import contours
import imutils
import re
import sys
import requests


eye_cascPath = 'haarcascade_eye_tree_eyeglasses.xml'  # eye detect model
face_cascPath = 'haarcascade_frontalface_alt.xml'  # face detect model
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)
# For testing
i = 0
j = 0
cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    if ret:
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        # print("Found {0} faces!".format(len(faces)))
        if len(faces) > 0:
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame_tmp = img[faces[0][1]:faces[0][1] + faces[0]
                            [3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
            frame = frame[faces[0][1]:faces[0][1] + faces[0]
                          [3], faces[0][0]:faces[0][0] + faces[0][2]:1]
            eyes = eyeCascade.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                # flags = cv2.CV_HAAR_SCALE_IMAGE
            )
            if len(eyes) == 0:
                print('no eyes!!!')
                i = i+1
            else:
                print('eyes!!!')
                j = j+1
                img_p = img
            frame_tmp = cv2.resize(frame_tmp, (400, 400),
                                   interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Face Recognition', frame_tmp)
        waitkey = cv2.waitKey(1)
        if i > 0 and j > 0:
            cv2.destroyAllWindows()
            break


def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

# extract faces and calculate face embeddings for a list of photo files


def get_embeddings(filenames):
    # extract faces
    faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False,
                    input_shape=(224, 224, 3), pooling='avg')
    # perform prediction
    yhat = model.predict(samples)
    return yhat

# determine if a candidate face is a match for a known face


def is_match(known_embedding, candidate_embedding, thresh=0.6):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))


# define filenames
cv2.imwrite("frame%d.jpg" % 0, img_p)
filenames = [sys.argv[1], 'frame0.jpg']
# get embeddings file filenames
embeddings = get_embeddings(filenames)
# define sharon stone
sharon_id = embeddings[0]
# verify known photos of sharon
is_match(embeddings[0], embeddings[1])


tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'

img = cv2.imread(sys.argv[1])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh1 = cv2.threshold(
    gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

refCnts = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_NONE)
refCnts = imutils.grab_contours(refCnts)
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
im2 = img.copy()
# print(refCnts)
file = open("recognized.txt", "w+")
file.write("")
file.close()

i = 0

for cnt in refCnts:
    x, y, w, h = cv2.boundingRect(cnt)

    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cropped = im2[y:y + h, x:x + w]

    file = open("recognized.txt", "a")

    text = pytesseract.image_to_string(cropped, config=tessdata_dir_config)
    split_text = text.split()
    symbols = '''{}()[],:;+-&|<>'=éè_çà=²&".^[^ !"`’'#%&,:;<>=@{}\$\(\)\\+\\\\?\[\]\^\|]+$'''

    res = [ele for ele in split_text if all(ch not in ele for ch in symbols)]
    res = [ele for ele in res if (len(ele) > 3 and ele.isupper())]
    removed = ['royaume', 'du', 'maroc', 'carte', 'jusqu’au',
               'nationale',  'né', 'le', 'valable', 'd’identite']
    res = " ".join(res)
    split_words = res.split()

    left_words = [word for word in split_words if word.lower() not in removed]
    text = ' '.join(left_words)
    ID = re.findall(r'\b[A-Z]\w+[0-9]\b', text)

    if len(text) != 0:
        if ID:
            file.write("ID: "+ID[0])
            file.write("\n")
        elif len(text) != 0:
            if(i == 0):
                file.write("prenom :")
                file.write(text)
                file.write("\n")
            if(i == 1):
                file.write("nom :")
                file.write(text)
                file.write("\n")
            if(i == 2):
                file.write("ville :")
                file.write(text)
                file.write("\n")
            i = i+1

    file.close
