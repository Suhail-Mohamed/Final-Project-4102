import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import face_alignment
from skimage import io

import cv2

# defining constants
MASK_POINT_THRESHOLD               = 5
MASK_BOUNDING_BOX_HEIGHT_THRESHOLD = 10
CHIN_THRESHOLD                     = 20
img_number                         = 1

#This is the model that allows us to get the facial landmark points of an image
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

# determines how to cover the mask in the maskless transformed image, uses canny
def find_chin_using_canny(canny_masked_input, chin_src_pts):
    dst_pts           = []
    good_chin_src_pts = []
    
    for point in chin_src_pts:
        row = int(point[1])
        col = int(point[0])
        r_copy = row

        while (r_copy < canny_masked_input.shape[0] and canny_masked_input[r_copy][col] == 0 and r_copy - row < CHIN_THRESHOLD):
            r_copy = r_copy + 1
        
        if (r_copy != canny_masked_input.shape[0]):
            dst_pts.append((r_copy, col))
            good_chin_src_pts.append((row, col))
    
    return np.array(dst_pts, dtype=np.float32)

#returns a vector that is the difference is distance between facial landmark point between the eyes and the chin of a person
def find_chin(maskless_transformed_landmarks):
    x_nose, y_nose = maskless_transformed_landmarks[27]
    x_chin, y_chin = maskless_transformed_landmarks[8]
    vector = [x_chin - x_nose, y_chin - y_nose]    
    return vector

def find_in_subdiv(pt, subdiv_points):
    for x in range(len(subdiv_points)):
        #print(pt, " - ", subdiv_points[x])
        if int(subdiv_points[x][0]) == pt[0] and int(subdiv_points[x][1]) == pt[1]:
            return x
    return -1


#getting the command line arg
try:
    img_number = int(sys.argv[1])
except:
    print("Invalid image number param defaulting to Masked man")

try:
    CHIN_THRESHOLD = int(sys.argv[2])
except:
    print("Invalid Chin threshold param defaulting to 20")

#reading in the images 
if img_number == 1:
    masked_input = io.imread('../images/masked_dude.jpg')
    maskless_input = io.imread('../images/maskless_dude.jpg')

elif img_number == 2:
    masked_input = io.imread('../images/masked_dude_rotated.jpg')
    maskless_input = io.imread('../images/maskless_dude.jpg')

elif img_number == 3:
    masked_input = io.imread('../images/Biden_masked.jpg')
    maskless_input = io.imread('../images/Biden_maskless.jpg')

elif img_number == 4:
    masked_input = io.imread('../images/lebron_masked.jpg')
    maskless_input = io.imread('../images/lebron_maskless.jpg')

else: 
    masked_input = io.imread('../images/masked_dude.jpg')
    maskless_input = io.imread('../images/maskless_dude.jpg')

masked_output = np.copy(masked_input)

masked_landmarks   = fa.get_landmarks(masked_output)
maskless_landmarks = fa.get_landmarks(maskless_input)

canny_masked_output = cv2.Canny(masked_input,100,200)

#Defining some variables to store our key facial landmark points
maskless_hull_list      = []
masked_hull_list        = []
face_landmark_mask_size = 68

mask_dst_pts           = []
maskless_src_pts       = []
maskless_subdiv_points = []

# collects facial landmarks for eyes and eyebrows for masked image
count = 1
for (x, y) in masked_landmarks[0]:
    #cv2.circle(masked_output, (int(x), int(y)), 2, (255, 0, 0), -1)
    if ((count >= 18 and count <= 27) or (count >= 37 and count <= 48)):
        mask_dst_pts.append([int(x), int(y)])
    if (count == 16  or count == 17 or count == 1 or count == 2):
        mask_dst_pts.append([int(x), int(y)])
    count = count + 1

# collects the facial landmark points for eyes and eye brows of the maskless image
count = 1
for (x, y) in maskless_landmarks[0]:
    #cv2.circle(maskless_input, (int(x), int(y)), 2, (255, 0, 0), -1)
    if((count >= 18 and count <= 27) or (count >= 37 and count <= 48)):
        maskless_src_pts.append([int(x), int(y)])
    if (count == 16  or count == 17 or count == 1 or count == 2):
        maskless_src_pts.append([int(x), int(y)])
    count = count + 1

# aliging the masked and maskless images so that they have the same perspective
M, mask_useless      = cv2.findHomography(np.array(maskless_src_pts), np.array(mask_dst_pts), cv2.RANSAC, 5.0)
warped               = cv2.warpPerspective(maskless_input, M, (masked_output.shape[1], masked_output.shape[0]))
maskless_transformed = cv2.perspectiveTransform(np.array([maskless_landmarks[0]]), M)

# correcting the homography to include an artificial/estimated chin point
# this improves our aligning substantially
chin_vector  = find_chin(maskless_transformed[0])
nose_pt_mask = maskless_transformed[0][27]
apprx_chin   = [int(nose_pt_mask[0] + chin_vector[0] + MASK_POINT_THRESHOLD), 
                int(nose_pt_mask[1] + chin_vector[1])]

mask_dst_pts.append(apprx_chin)
maskless_src_pts.append((int(maskless_landmarks[0][8][0]), int(maskless_landmarks[0][8][1])))

# re-computing homography with estimated chin point, improves the alignment
M, mask_useless      = cv2.findHomography(np.array(maskless_src_pts), np.array(mask_dst_pts), cv2.RANSAC, 5.0)
warped               = cv2.warpPerspective(maskless_input, M, (masked_output.shape[1], masked_output.shape[0]))
maskless_transformed = cv2.perspectiveTransform(np.array([maskless_landmarks[0]]), M)
masked_cp = np.zeros_like(masked_input)
warped_cp = np.zeros_like(warped)

# determining the hull list for the convex hull sorrounding our aligned maskless image
# this hull list is later used to cut our the lower half of the maskless transformed image 
count = 1
for (x, y) in maskless_transformed[0]:
    pt_draw = (int(x), int(y))
    if (count >= 2 and count <= 16) or count == 28:
        cv2.circle(warped_cp, (int(x), int(y)), 1, (0, 0, 255), -1)
        maskless_hull_list.append([x, y])
        maskless_subdiv_points.append((x, y))
    if (count >= 29 and count <= 36) or count >= 49:
        cv2.circle(warped_cp, (int(x), int(y)), 1, (0, 0, 255), -1)
        maskless_subdiv_points.append((x, y))
    count = count + 1

mask = np.zeros((warped.shape[0], warped.shape[1]), np.uint8)
maskless_hull_list = np.array(maskless_hull_list).reshape((-1,1,2)).astype(np.int32)

hull       = cv2.drawContours(mask, [maskless_hull_list], -1, 255, cv2.FILLED)
idx        = (hull == 255)
new_warped = np.copy(warped)

#Cutting out the maskless portion
for h in range(masked_output.shape[0]):
    for w in range(masked_output.shape[1]):
        if not(idx[h][w]):
            new_warped[h][w] = 0

chin_src_pts         = maskless_transformed[0][6:12]
new_returned_dst_pts = find_chin_using_canny(canny_masked_output, chin_src_pts)

# Calculate the points on the masked image for triangulation
masked_subdiv_points = list(maskless_subdiv_points)
count                = 1
chin_point_counter   = 0

for (x, y) in maskless_transformed[0]:
    if (count >= 2 and count <= 16) or count == 28:
        if count >= 7 and count <= 12:
            new_chin_pt = (new_returned_dst_pts[chin_point_counter][1], 
                           new_returned_dst_pts[chin_point_counter][0])
            
            masked_hull_list.append(new_chin_pt)
            masked_subdiv_points[count - 2] = new_chin_pt
            chin_point_counter = chin_point_counter + 1
        else:
            masked_hull_list.append([x, y])
    count = count + 1

print("LENGTH OF SUBDIV MASKED  : ", len(masked_subdiv_points))
print("LENGTH OF SUBDIV MASKLESS: ", len(maskless_subdiv_points))

for (x, y) in new_returned_dst_pts:
    pt_draw = (int(y), int(x))
    cv2.circle(canny_masked_output, pt_draw, 2, (255, 255, 255), -1)
    cv2.circle(masked_cp, pt_draw, 5, (0, 255, 0), -1)

# Creating subdiv lists and triangles that are used for applying Delaunay triangulation
maskless_rect = cv2.boundingRect(maskless_hull_list)
cv2.rectangle(warped, (maskless_rect[0], maskless_rect[1]), (maskless_rect[0] + maskless_rect[2], maskless_rect[1] + maskless_rect[3]), (255, 0, 0), 1)
maskless_subdiv = cv2.Subdiv2D(maskless_rect)
maskless_subdiv.insert(maskless_subdiv_points)
maskless_triangles = maskless_subdiv.getTriangleList()
maskless_triangles = np.array(maskless_triangles, dtype=np.int32)

maskless_indexes_triangles = []
masked_indexes_triangles   = []

#we needed to manually ensure that the trainglulation on the masked and maskless images matched up. 
for triangle in maskless_triangles :
    # Gets the vertex of the triangle
    pt1 = (triangle[0], triangle[1])
    pt2 = (triangle[2], triangle[3])
    pt3 = (triangle[4], triangle[5])

    idx1 = find_in_subdiv(pt1, maskless_subdiv_points)
    if idx1 == -1: 
        break
    
    idx2 = find_in_subdiv(pt2, maskless_subdiv_points)
    if idx2 == -1:
        break

    idx3 = find_in_subdiv(pt3, maskless_subdiv_points)
    if idx3 == -1:
        break

    pt1_masked = masked_subdiv_points[idx1]
    pt2_masked = masked_subdiv_points[idx2]
    pt3_masked = masked_subdiv_points[idx3]

    # Draws a line for each side of the triangle
    cv2.line(warped_cp, pt1, pt2, (255, 255, 255), 1,  0)
    cv2.line(warped_cp, pt2, pt3, (255, 255, 255), 1,  0)
    cv2.line(warped_cp, pt3, pt1, (255, 255, 255), 1,  0)

    masked_indexes_triangles.append((pt1_masked, pt2_masked, pt3_masked))
    maskless_indexes_triangles.append((pt1, pt2, pt3))

masked_hull_list = np.array(masked_hull_list).reshape((-1,1,2)).astype(np.int32)

print("MASKLESS HULL LIST: ", len(maskless_hull_list))
print("MASKED HULL LIST  : ", len(masked_hull_list))

for t in masked_indexes_triangles:
    # Draws a line for each side of the triangle
    cv2.line(masked_cp, (int(t[0][0]), int(t[0][1])), (int(t[1][0]), int(t[1][1])), (255, 255, 255), 1,  0)
    cv2.line(masked_cp, (int(t[1][0]), int(t[1][1])), (int(t[2][0]), int(t[2][1])), (255, 255, 255), 1,  0)
    cv2.line(masked_cp, (int(t[0][0]), int(t[0][1])), (int(t[2][0]), int(t[2][1])), (255, 255, 255), 1,  0)

for (x, y) in masked_subdiv_points:
    cv2.circle(masked_cp, (int(x), int(y)), 2, (0, 0, 255), -1)

print("MASKED   triangles size: ", len(masked_indexes_triangles))
print("MASKLESS triangles size: ", len(maskless_indexes_triangles)) 

if len(masked_indexes_triangles) != len(maskless_indexes_triangles):
    print("Error Different sized triangles")

# https://pysource.com/2019/05/28/face-swapping-explained-in-8-steps-opencv-with-python/ 
good_masked_output = np.zeros_like(masked_input)
for idx in range (len(masked_indexes_triangles) - 1):
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

#replacing the face of the maskless input image onto the masked image
result = masked_input.copy()
for h in range(good_masked_output.shape[0]):
    for w in range(good_masked_output.shape[1]):
        if (good_masked_output[h][w][0] != 0):
            result[h][w] = good_masked_output[h][w]

result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

print("Code is done Running ")
#Displaying the orgingal image and the results
cv2.imshow('Orginal Masked Image', masked_input)
cv2.imshow('Maskless Image', maskless_input)
cv2.imshow('Warped Image' , warped)
#cv2.imshow('Masked Output - Removal' , masked_output)
# cv2.imshow('FILTERING OUTPUT' , filtering_output)
cv2.imshow('Canny' , canny_masked_output)
cv2.imshow('Cut out of maskless' , new_warped)
cv2.imshow('Warped Copy - tri' , warped_cp)
cv2.imshow('Masked Copy - tri' , masked_cp)
cv2.imshow('GOOD MASKED OUTPUT' , good_masked_output)
cv2.imshow('Result' , result)

cv2.waitKey(0)
cv2.destroyAllWindows()