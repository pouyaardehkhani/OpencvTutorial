import numpy as np
import cv2
import matplotlib.pyplot as plt

def display(img,cmap=None):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap=cmap)

# --------------------------------Watershed Algorithm--------------------------

# Let's now try the watershed algorithm apporach to draw contours around the pennies. Also make sure to watch the video to understand what the function calls are doing here.

# Using the WaterShed Algorithm

#  Step 1: Read Image
img = cv2.imread('pennies.jpg')
sep_coins = cv2.imread('pennies.jpg')

# Step 2: Apply Blur
img = cv2.medianBlur(img,35)

# Step 3: Convert to Grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Step 4: Apply Threshold (Inverse Binary with OTSU as well)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
display(thresh,cmap='gray')

# Optional Step 5: Noise Removal
# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
display(opening,cmap='gray')

# Step 6: Grab Background that you are sure of
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
display(sure_bg,cmap='gray')

# Step 7: Find Sure Foreground
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
display(dist_transform,cmap='gray')
display(sure_fg,cmap='gray')

# Step 8: Find Unknown Region
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
display(unknown,cmap='gray')

# Step 9: Label Markers of Sure Foreground
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
display(markers,cmap='gray')

# Step 10: Apply Watershed Algorithm to find Markers
markers = cv2.watershed(img,markers)
display(markers)

# Step 11: Find Contours on Markers
contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# For every entry in contours
for i in range(len(contours)):
    
    # last column in the array is -1 if an external contour (no contours inside of it)
    if hierarchy[0][i][3] == -1:
        
        # We can now draw the external contours from the list of contours
        cv2.drawContours(sep_coins, contours, i, (255, 0, 0), 10)
display(sep_coins)