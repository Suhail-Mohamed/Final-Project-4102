import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from fer import FER
from imutils import face_utils

import face_alignment
from skimage import io

import cv2
import dlib

import matplotlib.pyplot as plt

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
MASK_BOUNDING_BOX_HEIGHT_THRESHOLD = 10

predictor          = "..\model\shape_predictor_68_face_landmarks.dat"
dlib_face_detector = dlib.get_frontal_face_detector()
facial_landmarks   = dlib.shape_predictor(predictor)

#This is the model that allows us to get the facial landmark points of an image
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
fer_detector = FER()

def find_chin_using_canny(canny_masked_input, chin_src_pts):
    print("CHINNNNNN POINTS: ", chin_src_pts)
    num_white_pts_seen = 0

    dst_pts = []
    good_chin_src_pts = []
    print("SHAPE_Rows: ", canny_masked_input.shape[0])
    print("SHAPE_Columns: ", canny_masked_input.shape[1])

    
    for point in chin_src_pts:
        print("Point: ", point)
        
        row = int(point[1])
        col  = int(point[0])
        r_copy = row

        while (r_copy < canny_masked_input.shape[0] and canny_masked_input[r_copy][col] == 0):
            r_copy = r_copy + 1
        
        if (r_copy != canny_masked_input.shape[0]):
            print("APPENDING A POINT")
            dst_pts.append((r_copy, col))
            good_chin_src_pts.append((row, col))
    
    print(good_chin_src_pts)
    print(dst_pts)

    return np.array(dst_pts, dtype=np.float32), np.array(good_chin_src_pts, dtype=np.float32)

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
masked_input = io.imread('../images/masked_dude.jpg')
maskless_input = io.imread('../images/maskless_dude.jpg')

# masked_input   = cv2.cvtColor(masked_input, cv2.COLOR_BGR2RGB)
# maskless_input = cv2.cvtColor(maskless_input, cv2.COLOR_BGR2RGB)

# masked_input = io.imread('../images/Biden_masked.jpg')
# maskless_input = io.imread('../images/Biden_maskless.jpg')

masked_output = np.copy(masked_input)
#masked_image = cv2.imread("../images/masked_dude.jpg"  , cv2.COLOR_BGR2GRAY)

masked_landmarks   = fa.get_landmarks(masked_output)
maskless_landmarks = fa.get_landmarks(maskless_input)

canny_masked_output = cv2.Canny(masked_input,100,200)

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

mask_dst_pts.append(apprx_chin)
maskless_src_pts.append((int(maskless_landmarks[0][8][0]), int(maskless_landmarks[0][8][1])))

M, mask_useless = cv2.findHomography(np.array(maskless_src_pts), np.array(mask_dst_pts), cv2.RANSAC, 5.0)
warped          = cv2.warpPerspective(maskless_input, M, (masked_output.shape[1], masked_output.shape[0]))
maskless_transformed = cv2.perspectiveTransform(np.array([maskless_landmarks[0]]), M)

count = 1
for (x, y) in maskless_transformed[0]:
    pt_draw = (int(x), int(y))
    # cv2.circle(canny_masked_output, pt_draw, 2, (255, 0, 255), -1)
    if count == 28:
        print("INSIDE THE TRANSFORM LOOP: ", x, y)
        # #cv2.circle(canny_masked_output, pt_draw, 10, (255, 0, 255), -1)
    if (count >= 2 and count <= 16) or count == 28:
        pt_hull = np.float32([x, y]).reshape(-1, 2)
        print(pt_hull)
        hull_list.append([x, y])
    count = count + 1

#hull = cv2.convexHull(np.array(hull_list, dtype='float32'))
print('PRINTING')
print(hull_list)

mask = np.zeros((warped.shape[0], warped.shape[1]), np.uint8)
hull_list = np.array(hull_list).reshape((-1,1,2)).astype(np.int32)

hull = cv2.drawContours(mask, [hull_list], -1, 255, cv2.FILLED)
idx  = (hull == 255)

new_warped = np.copy(warped)

#Cutting out the maskless portion
for h in range(masked_output.shape[0]):
    for w in range(masked_output.shape[1]):
        if not(idx[h][w]):
            new_warped[h][w] = 0

cv2.imshow('MASKED OUTPUT BEFORE DOING THE SWAP', masked_output)

chin_src_pts = maskless_transformed[0][6:12]
new_returned_dst_pts, good_chin_src_pts = find_chin_using_canny(canny_masked_output,chin_src_pts)

for (x, y) in new_returned_dst_pts:
    print("Drawing points: ", x , " : ", y)
    pt_draw = (int(y), int(x))
    cv2.circle(canny_masked_output, pt_draw, 2, (255, 255, 255), -1)

# M_chin, mask_useless = cv2.findHomography(good_chin_src_pts, new_returned_dst_pts, cv2.RANSAC, 5.0)
# stretched_output     = cv2.warpPerspective(new_warped, M_chin, (masked_output.shape[0], masked_output.shape[1]))

# M = cv2.getAffineTransform(good_chin_src_pts[1:4], new_returned_dst_pts[1:4])
# stretched_output = cv2.warpAffine(new_warped, M, (masked_output.shape[0], masked_output.shape[1]))

# M_chin = cv2.getPerspectiveTransform(good_chin_src_pts[1:5], new_returned_dst_pts[1:5])
# stretched_output = cv2.warpPerspective(new_warped, M_chin, (masked_output.shape[0], masked_output.shape[1]))

# new_warped2 = cv2.resize(new_warped, (masked_output.shape[0] * 1.2, masked_output.shape[1] * 2), interpolation=cv2.INTER_LINEAR)

# rect = cv2.boundingRect(hull_list)
# subdiv = cv2.Subdiv2D(rect)
# subdiv.insert(masked_landmarks)
# triangles = subdiv.getTriangleList()
# triangles = np.array(triangles, dtype=np.int32)

cv2.circle(canny_masked_output, (5, 200), 10, (255, 255, 255), -1)

# for x in range(masked_output.shape[0]):
#     for y in range(masked_output.shape[1]):
#         if stretched_output[x][y][0]:
#             #print("MASKED OUTPUT:", masked_output[x][y])
#             masked_output[x][y] = stretched_output[x][y]

# filtering_output = np.copy(masked_output)

# x_bound,y_bound,w_bound,h_bound = cv2.boundingRect(hull)
# cv2.rectangle(masked_output, (x_bound,y_bound), (x_bound+w_bound,y_bound+h_bound + MASK_BOUNDING_BOX_HEIGHT_THRESHOLD), (0,255,255), 2)

# # print("Color on mask: ", masked_output[x+ int(w_bound/2)][y+h])

# dark_blue_mask  = (215, 168, 97)
# light_blue_mask = (246, 221, 201)

# masked_output   = cv2.cvtColor(masked_output, cv2.COLOR_BGR2RGB)
# for y in range(y_bound, y_bound + h_bound + MASK_BOUNDING_BOX_HEIGHT_THRESHOLD):
#     for x in range(x_bound, x_bound + w_bound):
#         #print("INSIDE LOOP: ", x, " ", y)
#         as_tuple = tuple(masked_output[y][x])
#         if as_tuple > dark_blue_mask and as_tuple < light_blue_mask:
#             masked_output[y][x] = (124,177,227)
            
# print(idx)
# print("SIZES00")
# print(masked_output.shape)
# print(mask.shape)
# print(warped.shape)

#result = cv2.bitwise_and(img2_resized, mask)
#poly_image = cv2.polylines(warped, [hull_list], True,(0, 255, 0), 2)

# M_chin = cv2.getPerspectiveTransform(good_chin_src_pts, chin_dst_pts)
# stretched_output = cv2.warpPerspective(masked_output, M_chin, (masked_output.shape[0], masked_output.shape[1]))

# M_chin, mask_useless = cv2.findHomography(good_chin_src_pts, new_returned_dst_pts, cv2.RANSAC, 5.0)
# stretched_output     = cv2.warpPerspective(warped, M_chin, (masked_output.shape[0], masked_output.shape[1]))

# M = cv2.getAffineTransform(good_chin_src_pts[1:4], new_returned_dst_pts[1:4])
# output_test = cv2.warpAffine(masked_output, M, (masked_output.shape[0], masked_output.shape[1]))

                                    
print("Code is done Running ")
#Displaying the orgingal image and the results
cv2.imshow('Orginal Masked Image', masked_input)
cv2.imshow('Maskless Image', maskless_input)
cv2.imshow('Warped Image' , warped)
cv2.imshow('Masked Output - Removal' , masked_output)
# cv2.imshow('FILTERING OUTPUT' , filtering_output)
cv2.imshow('Canny' , canny_masked_output)
cv2.imshow('Cut out of maskless' , new_warped)
#cv2.imshow('Stretching - output' , new_warped2)




#cv2.imshow('MASK_FILTERING' , mask_for_filtering_mask)

#Determining the emotions 
print("Emotions")
print(determine_emotion(masked_output))
print(determine_emotion(maskless_input))


cv2.waitKey(0)
cv2.destroyAllWindows()
