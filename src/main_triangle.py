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

maskless_hull_list      = []
masked_hull_list        = []
face_landmark_mask_size = 68

mask_dst_pts           = []
maskless_src_pts       = []
maskless_subdiv_points = []

count = 1
for (x, y) in masked_landmarks[0]:
    #cv2.circle(masked_output, (int(x), int(y)), 2, (255, 0, 0), -1)
    if ((count >= 18 and count <= 27) or (count >= 37 and count <= 48)):
        mask_dst_pts.append([int(x), int(y)])
    if (count == 16  or count == 17 or count == 1 or count == 2):
        mask_dst_pts.append([int(x), int(y)])
    count = count + 1

count = 1
for (x, y) in maskless_landmarks[0]:
    #cv2.circle(maskless_input, (int(x), int(y)), 2, (255, 0, 0), -1)
    if((count >= 18 and count <= 27) or (count >= 37 and count <= 48)):
        maskless_src_pts.append([int(x), int(y)])
    if (count == 16  or count == 17 or count == 1 or count == 2):
        maskless_src_pts.append([int(x), int(y)])
    count = count + 1

M, mask_useless      = cv2.findHomography(np.array(maskless_src_pts), np.array(mask_dst_pts), cv2.RANSAC, 5.0)
warped               = cv2.warpPerspective(maskless_input, M, (masked_output.shape[1], masked_output.shape[0]))
maskless_transformed = cv2.perspectiveTransform(np.array([maskless_landmarks[0]]), M)

chin_vector  = find_chin(maskless_transformed[0])
nose_pt_mask = maskless_transformed[0][27]

print("OUTSIDE THE TRANSFORM LOOP", nose_pt_mask)
apprx_chin   = [int(nose_pt_mask[0] + chin_vector[0] + MASK_POINT_THRESHOLD), 
                int(nose_pt_mask[1] + chin_vector[1])]

mask_dst_pts.append(apprx_chin)
maskless_src_pts.append((int(maskless_landmarks[0][8][0]), int(maskless_landmarks[0][8][1])))

M, mask_useless      = cv2.findHomography(np.array(maskless_src_pts), np.array(mask_dst_pts), cv2.RANSAC, 5.0)
warped               = cv2.warpPerspective(maskless_input, M, (masked_output.shape[1], masked_output.shape[0]))
maskless_transformed = cv2.perspectiveTransform(np.array([maskless_landmarks[0]]), M)

count = 1
for (x, y) in maskless_transformed[0]:
    pt_draw = (int(x), int(y))
    if (count >= 2 and count <= 16) or count == 28:
        cv2.circle(warped, (int(x), int(y)), 1, (0, 0, 255), -1)
        maskless_hull_list.append([x, y])
        maskless_subdiv_points.append((x, y))
    if (count >= 29 and count <= 36) or count >= 49:
        cv2.circle(warped, (int(x), int(y)), 1, (0, 0, 255), -1)
        maskless_subdiv_points.append((x, y))
    count = count + 1

#hull = cv2.convexHull(np.array(hull_list, dtype='float32'))
# print('PRINTING')
# print(maskless_hull_list)

mask = np.zeros((warped.shape[0], warped.shape[1]), np.uint8)
maskless_hull_list = np.array(maskless_hull_list).reshape((-1,1,2)).astype(np.int32)

hull = cv2.drawContours(mask, [maskless_hull_list], -1, 255, cv2.FILLED)
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

# Calculate the points on the masked image for triangulation
masked_cp        = masked_input.copy()

masked_subdiv_points = []
count = 1
chin_point_counter = 0
print("Chin Src pts: ", new_returned_dst_pts)
for (x, y) in maskless_transformed[0]:
    pt_draw = (int(x), int(y))
    
    if (count >= 2 and count <= 16) or count == 28:
        if count >= 7 and count <= 12:
            # add the custom found chin point
            pt_draw = (int(new_returned_dst_pts[chin_point_counter][1]), 
                       int(new_returned_dst_pts[chin_point_counter][0]))
            
            masked_subdiv_points.append((new_returned_dst_pts[chin_point_counter][1], 
                                         new_returned_dst_pts[chin_point_counter][0]))
        
            masked_hull_list.append([new_returned_dst_pts[chin_point_counter][1], 
                                     new_returned_dst_pts[chin_point_counter][0]])
            
            chin_point_counter = chin_point_counter + 1
        else:
            masked_subdiv_points.append((x, y))
            masked_hull_list.append([x, y])
        
    if (count >= 29 and count <= 36) or count >= 49:
        cv2.circle(masked_cp, pt_draw, 1, (0, 0, count * 8), -1)
        masked_subdiv_points.append((x, y))
    
    count = count + 1

for (x, y) in masked_subdiv_points:
    cv2.circle(masked_cp, (int(x), int(y)), 1, (0, 0, 255), -1)

print("MASKLESS HULL LIST", len(maskless_hull_list))
print("MASKED HULL LIST"  , len(masked_hull_list))

print("MASKLESS SUBDIV POINTS:", len(maskless_subdiv_points))
print("MASKED SUBDIV POINTS:", len(masked_subdiv_points))

for (x, y) in new_returned_dst_pts:
    print("Drawing points: ", x , " : ", y)
    pt_draw = (int(y), int(x))
    cv2.circle(canny_masked_output, pt_draw, 2, (255, 255, 255), -1)
    # cv2.circle(warped, pt_draw, 2, (255, 0, 255), -1)

# For maskless image triangulation
maskless_rect = cv2.boundingRect(maskless_hull_list)
print("RECTANGLE:", maskless_rect)
# cv2.rectangle(warped, (maskless_rect[0], maskless_rect[1]), (maskless_rect[0] + maskless_rect[2], maskless_rect[1] + maskless_rect[3]), (255, 0, 0), 5)

maskless_subdiv = cv2.Subdiv2D(maskless_rect)
print("MAsked Transformed[0]", maskless_transformed[0])
print("Subdiv points", maskless_subdiv_points)
print("NUM SUBDIV POINTS: ", len(maskless_subdiv_points))

maskless_subdiv.insert(maskless_subdiv_points)
maskless_triangles = maskless_subdiv.getTriangleList()
print("GET TRIANGLES LEN Maskless: ", len(maskless_triangles))

maskless_triangles = np.array(maskless_triangles, dtype=np.int32)

maskless_indexes_triangles = []
warped_cp                  = warped.copy()

for triangle in maskless_triangles :
    # Gets the vertex of the triangle
    pt1 = (triangle[0], triangle[1])
    pt2 = (triangle[2], triangle[3])
    pt3 = (triangle[4], triangle[5])

    # Draws a line for each side of the triangle
    # cv2.line(warped_cp, pt1, pt2, (255, 255, 255), 1,  0)
    # cv2.line(warped_cp, pt2, pt3, (255, 255, 255), 1,  0)
    # cv2.line(warped_cp, pt3, pt1, (255, 255, 255), 1,  0)

    maskless_indexes_triangles.append((pt1, pt2, pt3))

masked_hull_list = np.array(masked_hull_list).reshape((-1,1,2)).astype(np.int32)

print("MASKLESS HULL LIST: ", maskless_hull_list)
print("MASKED HULL LIST", masked_hull_list)

# For masked image trangulation
masked_rect = cv2.boundingRect(masked_hull_list)
masked_subdiv = cv2.Subdiv2D(masked_rect)
masked_subdiv.insert(masked_subdiv_points)
masked_triangles = masked_subdiv.getTriangleList()
print("GET TRIANGLES LEN Masked: ", len(masked_triangles))

masked_triangles = np.array(masked_triangles, dtype=np.int32)
masked_indexes_triangles = []

cv2.rectangle(masked_cp, (masked_rect[0], masked_rect[1]), (masked_rect[0] + masked_rect[2], masked_rect[1] + masked_rect[3]), (255, 0, 0), 1)

for triangle in masked_triangles :
    # Gets the vertex of the triangle
    pt1 = (triangle[0], triangle[1])
    pt2 = (triangle[2], triangle[3])
    pt3 = (triangle[4], triangle[5])

    # Draws a line for each side of the triangle
    # cv2.line(masked_cp, pt1, pt2, (255, 255, 255), 1,  0)
    # cv2.line(masked_cp, pt2, pt3, (255, 255, 255), 1,  0)
    # cv2.line(masked_cp, pt3, pt1, (255, 255, 255), 1,  0)

    masked_indexes_triangles.append((pt1, pt2, pt3))

print("MASKED triangles SIZE", len(masked_indexes_triangles))
print("Maskless triagles size", len(maskless_indexes_triangles)) 

good_masked_output = np.zeros_like(masked_input)
for idx in range (len(masked_indexes_triangles) - 1):
    # Coordinates of the first person's delaunay triangles
    #print("TRIANGLE:", maskless_indexes_triangles[idx])

    pt1_maskless = maskless_indexes_triangles[idx][0]
    pt2_maskless = maskless_indexes_triangles[idx][1]
    pt3_maskless = maskless_indexes_triangles[idx][2]
    
    # Gets the delaunay triangles
    (x, y, width, height) = cv2.boundingRect(np.array([pt1_maskless, pt2_maskless, pt3_maskless], np.int32))
    maskless_cropped_triangle = warped[y: y+height, x: x+width]
    maskless_cropped_mask     = np.zeros((height, width), np.uint8)

    # Fills triangle to generate the mask
    maskless_pts = np.array([[pt1_maskless[0]-x, pt1_maskless[1]-y], 
                             [pt2_maskless[0]-x, pt2_maskless[1]-y], 
                             [pt3_maskless[0]-x, pt3_maskless[1]-y]], np.int32)
    
    cv2.fillConvexPoly(maskless_cropped_mask, maskless_pts, 255)
    
    pt1_masked = masked_indexes_triangles[idx][0]
    pt2_masked = masked_indexes_triangles[idx][1]
    pt3_masked = masked_indexes_triangles[idx][2]

    # Gets the delaunay triangles
    (x, y, width, height) = cv2.boundingRect(np.array([pt1_masked, pt2_masked, pt3_masked], np.int32))
    masked_cropped_mask     = np.zeros((height, width), np.uint8)

    # Fills triangle to generate the mask
    masked_pts = np.array([[pt1_masked[0]-x, pt1_masked[1]-y], 
                           [pt2_masked[0]-x, pt2_masked[1]-y], 
                           [pt3_masked[0]-x, pt3_masked[1]-y]], np.int32)
    
    cv2.fillConvexPoly(masked_cropped_mask, masked_pts, 255)
    
    maskless_pts = np.float32(maskless_pts)
    masked_pts   = np.float32(masked_pts)
    
    M             = cv2.getAffineTransform(maskless_pts, masked_pts)  # Warps the content of the first triangle to fit in the second one
    dist_triangle = cv2.warpAffine(maskless_cropped_triangle, M, (width, height))
    dist_triangle = cv2.bitwise_and(dist_triangle, dist_triangle, mask=masked_cropped_mask)

    # Joins all the distorted triangles to make the face mask to fit in the second person's features
    body_new_face_rect_area = good_masked_output[y: y+height, x: x+width]
    body_new_face_rect_area_gray = cv2.cvtColor(body_new_face_rect_area, cv2.COLOR_BGR2GRAY)

    # Creates a mask
    masked_triangle = cv2.threshold(body_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    dist_triangle = cv2.bitwise_and(dist_triangle, dist_triangle, mask=masked_triangle[1])

    # Adds the piece to the face mask
    body_new_face_rect_area = cv2.add(body_new_face_rect_area, dist_triangle)
    good_masked_output[y: y+height, x: x+width] = body_new_face_rect_area

# body_face_mask = np.zeros_like(masked_input)
# body_head_mask = cv2.fillConvexPoly(body_face_mask, masked_hull_list, 255)
# body_face_mask = cv2.bitwise_not(body_head_mask)

# body_maskless = cv2.bitwise_and(masked_output, masked_output, mask=body_face_mask)
# result = cv2.add(body_maskless, good_masked_output)

print("Code is done Running ")
#Displaying the orgingal image and the results
cv2.imshow('Orginal Masked Image', masked_input)
cv2.imshow('Maskless Image', maskless_input)
cv2.imshow('Warped Image' , warped)
cv2.imshow('Masked Output - Removal' , masked_output)
# cv2.imshow('FILTERING OUTPUT' , filtering_output)
cv2.imshow('Canny' , canny_masked_output)
cv2.imshow('Cut out of maskless' , new_warped)
cv2.imshow('Warped Copy - tri' , warped_cp)
cv2.imshow('Masked Copy - tri' , masked_cp)
cv2.imshow('GOOD MASKED OUTPUT' , good_masked_output)
# cv2.imshow('Result' , result)

cv2.waitKey(0)
cv2.destroyAllWindows()
