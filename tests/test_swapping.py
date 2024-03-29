import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from fer import FER
from imutils import face_utils

import face_alignment
from skimage import io

import cv2
import dlib

'''
Idea:
    - Align eyebrow/eyes/nose-bridge
    - Once these keypoints aligned alter non-masked face
    - face swap these 

functions:
    determine_emotion(img) -> str
    change_emotion(img_non_masked, emotion:str) -> void
    face_swap(img_non-masked, img_masked) -> void
'''
predictor          = "..\model\shape_predictor_68_face_landmarks.dat"
dlib_face_detector = dlib.get_frontal_face_detector()
facial_landmarks   = dlib.shape_predictor(predictor)

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
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

# cv2.imshow('Maskless', maskless)
# cv2.imshow('Masked', masked)

#find_land_marks(masked)
masked_input = io.imread('../images/lebron_masked.jpg')
maskless_input = io.imread('../images/lebron_maskless.jpg')

#masked_image = cv2.imread("../images/masked_dude.jpg"  , cv2.COLOR_BGR2GRAY)
masked_landmarks = fa.get_landmarks(masked_input)
maskless_landmarks = fa.get_landmarks(maskless_input)

print(len(masked_landmarks[0]))
count = 1

hull = []
face_landmark_mask_size = 69

mask_dst_pts = []
maskless_src_pts = []

maskless_nose_point = []
maskless_chin_point = []


for (x, y) in masked_landmarks[0]:
    if ((count >= 18 and count <= 27) or (count >= 37 and count <= 48)):
        mask_dst_pts.append([int(x), int(y)])
        cv2.circle(masked_input, (int(x), int(y)), 2, (255, 0, 0), -1)
    count = count + 1

count = 1
for (x, y) in maskless_landmarks[0]:
    if((count >= 18 and count <= 27) or (count >= 37 and count <= 48)):
        maskless_src_pts.append([int(x), int(y)])
        cv2.circle(maskless_input, (int(x), int(y)), 2, (255, 0, 0), -1)
    if count == 9:
        maskless_chin_point.append([int(x), int(y)])
        cv2.circle(maskless_input, (int(x), int(y)), 2, (255, 0, 0), -1)
    if count == 31:
        maskless_nose_point.append([int(x), int(y)])
        cv2.circle(maskless_input, (int(x), int(y)), 2, (255, 0, 0), -1)
    count = count + 1

M, mask_useless = cv2.findHomography(np.array(maskless_src_pts), np.array(mask_dst_pts), cv2.RANSAC, 5.0)
warped = cv2.warpPerspective(maskless_input, M, (masked_input.shape[1], masked_input.shape[0]))

chin_point_warped = cv2.perspectiveTransform(np.float32([maskless_chin_point]),M)
nose_point_warped = cv2.perspectiveTransform(np.float32([maskless_nose_point]),M)

print("PRINTING THE BOUNDRY OF THE POINTS")
print(chin_point_warped[0][0])
print(nose_point_warped[0][0])


mashed_image = np.copy(masked_input)

for i in range (mashed_image.shape[0]):
    for j in range (mashed_image.shape[1]):
        if i > 130 and i < 200 and j > 50 and j < 163:
            mashed_image[i][j] = warped[i][j]

#Mash that shit together j



# print(np.float32(maskless_landmarks[0]))

# maskless_transformed = cv2.perspectiveTransform(np.array([maskless_landmarks[0]]), M)
# print(maskless_transformed)

# count = 1
# for (x, y) in maskless_landmarks[0]:
#     if (count >= 1 and count <= 17):
#         pt_draw = (int(x), int(y))
        
#         cv2.circle(maskless_transformed, (int), 2, (255, 0, 0), -1)
#         print(type(maskless_transformed[0][count]))
#         hull.append(cv2.convexHull(np.array([warped[0][count]]), False))
#     count = count + 1

# hull.append(cv2.convexHull(np.array([maskless_transformed[0][29]]), False))
# print (hull)

# create an empty black image
# drawing = np.zeros((maskless_input.shape[0], maskless_input.shape[1], 3), np.uint8)
 
# # draw contours and hull points
# for i in range(len(hull)):
#     color_contours = (0, 255, 0) # green - color for contours
#     color = (255, 0, 0) # blue - color for convex hull
#     # 87draw ith contour
#     # cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
#     # draw ith convex hull object
#     # cv2.drawContours(drawing, hull, i, color, 1, 8)


print("Done ")
cv2.imshow('Normal', masked_input)
cv2.imshow('Rotated', maskless_input)
cv2.imshow('Warped', warped)
cv2.imshow('MASHED', mashed_image)



cv2.waitKey(0)
cv2.destroyAllWindows()
