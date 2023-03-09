import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fer import FER
from imutils import face_utils

import cv2
import dlib

'''
functions:
    determine_emotion(img) -> str
    change_emotion(img_non_masked, emotion:str) -> void
    face_swap(img_non-masked, img_masked) -> void
'''
predictor          = "..\model\shape_predictor_68_face_landmarks.dat"
dlib_face_detector = dlib.get_frontal_face_detector()
facial_landmarks   = dlib.shape_predictor(predictor)

fer_detector = FER()

def find_land_marks(face_img):  
    print("FIND_LAND_MARKS")  
    rects = dlib_face_detector(face_img, 0)
    print(rects)

    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = facial_landmarks(face_img, rect)
        shape = face_utils.shape_to_np(shape)
    
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(face_img, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow('GDJhbfjkdszfhkdzj', face_img)


def triangulation_face(img):
    pass

def face_swap(img_non_masked, img_masked):
    # find land marks and do the triangulation of both images
    pass

def change_emotion(img_non_masked, emotion):
    # Do transformations where the facial landmarks are    
    pass

def determine_emotion(img):
    fer_detector.detect_emotions(img)
    emotion, _ = fer_detector.top_emotion(img)
    return emotion

def input_images(img_1, img_2):
    #get the greyscale versions of our images
    maskless_image = cv2.imread(img_1, cv2.COLOR_BGR2GRAY)  
    masked_image = cv2.imread(img_2  , cv2.COLOR_BGR2GRAY)
    masked_image_resized = cv2.resize(masked_image, (maskless_image.shape[1], maskless_image.shape[0]))
    return maskless_image, masked_image_resized

maskless, masked = input_images("../images/maskless_dude.jpg", "../images/masked_dude.jpg")



cv2.imshow('Maskless', maskless)
cv2.imshow('Masked', masked)

find_land_marks(masked)

cv2.waitKey(0)
cv2.destroyAllWindows()