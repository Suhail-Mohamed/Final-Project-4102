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
TODO:
    - Finding the chin (we can try)
    - Try to belend color, use mask color in the image and compare with the replaced face to ensure that there is no mask left
    - Clean up code
    - Smoothing at the seams of the mask
    - Write report

'''
#Initalizing the models that we are going to use

MASK_POINT_THRESHOLD = 5

predictor          = "..\model\shape_predictor_68_face_landmarks.dat"
dlib_face_detector = dlib.get_frontal_face_detector()
facial_landmarks   = dlib.shape_predictor(predictor)

#This is the model that allows us to get the facial landmark points of an image
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
fer_detector = FER()

def find_chin(maskless_transformed_landmarks):
    x_nose, y_nose = maskless_transformed_landmarks[27]
    x_chin, y_chin = maskless_transformed_landmarks[8]
    vector = [x_chin - x_nose, y_chin - y_nose]    
    
    return vector

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

#maskless, masked = input_images("../images/maskless_dude.jpg", "../images/masked_dude.jpg")

# cv2.imshow('Maskless', maskless)
# cv2.imshow('Masked', masked)
#find_land_marks(masked)

#reading in the images 
masked_input = io.imread('../images/masked_dude_rotated.jpg')
maskless_input = io.imread('../images/maskless_dude.jpg')

# masked_input = io.imread('../images/Biden_masked.jpg')
# maskless_input = io.imread('../images/Biden_maskless.jpg')

masked_output = np.copy(masked_input)
#masked_image = cv2.imread("../images/masked_dude.jpg"  , cv2.COLOR_BGR2GRAY)

masked_landmarks   = fa.get_landmarks(masked_output)
maskless_landmarks = fa.get_landmarks(maskless_input)

print(len(masked_landmarks[0]))

hull_list = []
face_landmark_mask_size = 69

mask_dst_pts = []
maskless_src_pts = []

count = 1
for (x, y) in masked_landmarks[0]:
    cv2.circle(masked_output, (int(x), int(y)), 2, (255, 0, 0), -1)
    if ((count >= 18 and count <= 27) or (count >= 37 and count <= 48)):
        mask_dst_pts.append([int(x), int(y)])
    if (count == 16  or count == 17 or count == 1 or count == 2):
        mask_dst_pts.append([int(x), int(y)])
    count = count + 1

count = 1
for (x, y) in maskless_landmarks[0]:
    cv2.circle(maskless_input, (int(x), int(y)), 2, (255, 0, 0), -1)
    if((count >= 18 and count <= 27) or (count >= 37 and count <= 48)):
        maskless_src_pts.append([int(x), int(y)])
    if (count == 16  or count == 17 or count == 1 or count == 2):
        maskless_src_pts.append([int(x), int(y)])
    count = count + 1

M, mask_useless = cv2.findHomography(np.array(maskless_src_pts), np.array(mask_dst_pts), cv2.RANSAC, 5.0)
warped          = cv2.warpPerspective(maskless_input, M, (masked_output.shape[1], masked_output.shape[0]))
maskless_transformed = cv2.perspectiveTransform(np.array([maskless_landmarks[0]]), M)

chin_vector  = find_chin(maskless_transformed[0])
nose_pt_mask = maskless_transformed[0][27]

print("OUTSIDE THE TRANSFORM LOOP", nose_pt_mask)
apprx_chin   = [int(nose_pt_mask[0] + chin_vector[0] + MASK_POINT_THRESHOLD), 
                int(nose_pt_mask[1] + chin_vector[1])]

print("STARTING POINT FOR VECTOR ADDITION:", nose_pt_mask)
cv2.circle(masked_output, (int(nose_pt_mask[0]), int(nose_pt_mask[1])), 2, (0, 255, 0), -1)

print("APPROXIMATE CHIN:", apprx_chin)
cv2.circle(masked_output, (int(apprx_chin[0]), int(apprx_chin[1])), 2, (0, 255, 0), -1)

# mask_dst_pts.append(apprx_chin)
# maskless_src_pts.append((int(maskless_landmarks[0][8][0]), int(maskless_landmarks[0][8][1])))

# M, mask_useless = cv2.findHomography(np.array(maskless_src_pts), np.array(mask_dst_pts), cv2.RANSAC, 5.0)
# warped          = cv2.warpPerspective(maskless_input, M, (masked_output.shape[1], masked_output.shape[0]))
# maskless_transformed = cv2.perspectiveTransform(np.array([maskless_landmarks[0]]), M)


count = 1
for (x, y) in maskless_transformed[0]:
    if count == 28:
        pt_draw = (int(x), int(y))
        print("INSIDE THE TRANSFORM LOOP: ", x, y)
        #cv2.circle(warped, pt_draw, 10, (255, 0, 255), -1)
    if (count >= 2 and count <= 16) or count == 28:
        pt_hull = np.float32([x, y]).reshape(-1, 2)
        print(pt_hull)
        hull_list.append([x, y])
    count = count + 1

'''
Creating mask from contour [[ 1626.   360.]
 [ 1776.  3108.]
 [  126.  3048.]
 [  330.   486.]]
 
def create_mask(img, cnt):
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    print("create_mask, cnt=%s" % cnt)
    cv2.drawContours(mask, [cnt.astype(int)], 0, (0, 255, 0), -1)
    return mask
'''

#hull = cv2.convexHull(np.array(hull_list, dtype='float32'))
print('PRINTING')
print(hull_list)

mask = np.zeros((warped.shape[0], warped.shape[1]), np.uint8)
hull_list = np.array(hull_list).reshape((-1,1,2)).astype(np.int32)

hull = cv2.drawContours(mask, [hull_list], -1, 255, cv2.FILLED)
idx  = (hull == 255)

for x in range(masked_output.shape[0]):
    for y in range(masked_output.shape[1]):
        if idx[x][y]:
            masked_output[x][y] = warped[x][y]


print(idx)
print("SIZES00")
print(masked_output.shape)
print(mask.shape)
print(warped.shape)

#result = cv2.bitwise_and(img2_resized, mask)
poly_image = cv2.polylines(warped, [hull_list], True,(0, 255, 0), 2)

print("Code is done Running ")

#Displaying the orgingal image and the results
cv2.imshow('Orginal Masked Image', masked_input)
cv2.imshow('Maskless Image', maskless_input)
cv2.imshow('Warped Image' , warped)
cv2.imshow('Masked Output - Removal' , masked_output)

#Determining the emotions 
print("Emotions")
print(determine_emotion(masked_output))
print(determine_emotion(maskless_input))


cv2.waitKey(0)
cv2.destroyAllWindows()
