import cv2 
import numpy as np  

import matplotlib.pyplot as plt

flat_chess = cv2.imread('flat_chessboard.png')
gray_flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2GRAY)

real_chess = cv2.imread('real_chessboard.jpg')
gray_real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2GRAY)

# --------------------------Corner Harris Detection------------------------------
'''
cornerHarris Function

    src Input single-channel 8-bit or floating-point image.
    dst Image to store the Harris detector responses. It has the type CV_32FC1 and the same size as src .
    blockSize Neighborhood size (see the details on #cornerEigenValsAndVecs ).
    ksize Aperture parameter for the Sobel operator.
    k Harris detector free parameter. See the formula in DocString
    borderType Pixel extrapolation method. See #BorderTypes.
'''

# Convert Gray Scale Image to Float Values
gray = np.float32(gray_flat_chess)

# Corner Harris Detection
dst = cv2.cornerHarris(src=gray,blockSize=2,ksize=3,k=0.04)

# result is dilated for marking the corners, not important to actual corner detection
# this is just so we can plot out the points on the image shown
dst = cv2.dilate(dst,None)

flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2RGB)

# Threshold for an optimal value, it may vary depending on the image.
flat_chess[dst>0.01*dst.max()]=[255,0,0]

plt.imshow(flat_chess)


# Convert Gray Scale Image to Float Values
gray = np.float32(gray_real_chess)

# Corner Harris Detection
dst = cv2.cornerHarris(src=gray,blockSize=2,ksize=3,k=0.04)

# result is dilated for marking the corners, not important to actual corner detection
# this is just so we can plot out the points on the image shown
dst = cv2.dilate(dst,None)

real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2RGB)

# Threshold for an optimal value, it may vary depending on the image.
real_chess[dst>0.01*dst.max()]=[255,0,0]

plt.imshow(real_chess)

# --------------------------Shi-Tomasi Corner Detector--------------------------
'''
goodFeatureToTrack Function Parameters
    image Input 8-bit or floating-point 32-bit, single-channel image.
    corners Output vector of detected corners.
    maxCorners Maximum number of corners to return. If there are more corners than are found,the strongest of them is returned. maxCorners <= 0 implies that no limit on the maximum is set and all detected corners are returned.
    qualityLevel Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see #cornerMinEigenVal ) or the Harris function response (see #cornerHarris ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.
'''
# Need to reset the images since we drew on them
flat_chess = cv2.imread('flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2RGB)
gray_flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray_flat_chess,64,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(flat_chess,(x,y),3,255,-1)

plt.imshow(flat_chess)

# Need to reset the images since we drew on them
real_chess = cv2.imread('real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2RGB)
gray_real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray_real_chess,80,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(real_chess,(x,y),3,255,-1)

plt.imshow(real_chess)
