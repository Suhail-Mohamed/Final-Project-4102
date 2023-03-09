import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fer import FER
import cv2

files = ["../images/happy_lebron.jpg", "../images/another_happy_lebron.jpg", "../images/willy_wonka_smiling.jpg", "../images/sparta_angry.jpg"]


detector = FER()

for f in files:
    img = cv2.imread(f)
    detector.detect_emotions(img)
    emotion, score = detector.top_emotion(img)
    print(emotion)























