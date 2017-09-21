# -*- coding: utf-8 -*-
"""
*******************************************************************************
Author:    LUIZ RICARDO TAKESHI HORITA
Email:     lrthorita@gmail.com
*******************************************************************************
Date: September 19-th, 2017
*******************************************************************************
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

## ========================   CIRCLES IMAGE   =============================
circles_img = cv2.imread('imagens/circles.png',1)

# Converting to grayscale
circles_gray = cv2.cvtColor(circles_img, cv2.COLOR_BGR2GRAY)
circles_gray = cv2.equalizeHist(circles_gray)

# Segment circles
ret, circles_gray = cv2.threshold(circles_gray,127,255,cv2.THRESH_BINARY_INV)

# Detect circles
circles_circles = cv2.HoughCircles(circles_gray, cv2.HOUGH_GRADIENT, dp=7,
                                   minDist=30, param1=50, param2=80, 
                                   minRadius=15, maxRadius=100)
circles_circles = np.round(circles_circles[0,:]).astype('int')

circles_circles_small = cv2.HoughCircles(circles_gray, cv2.HOUGH_GRADIENT, dp=2,
                                   minDist=50, param1=100, param2=15, 
                                   minRadius=5, maxRadius=15)
circles_circles_small = np.round(circles_circles_small[0,:]).astype('int')

circles_circles = np.vstack((circles_circles, circles_circles_small))

# Draw circles on image
for i in xrange(np.size(circles_circles,0)):
    cv2.circle(circles_img, (circles_circles[i,0],circles_circles[i,1]),
               circles_circles[i,2], (50,180,50), 3)


## ========================   SHAPES_LEO IMAGE   =============================
shapes_img = cv2.imread('imagens/shapes_leo.jpg',1)

# Converting to grayscale
shapes_gray = cv2.cvtColor(shapes_img, cv2.COLOR_BGR2GRAY)
shapes_gray = cv2.equalizeHist(shapes_gray)

# Segment circles
ret, shapes_gray = cv2.threshold(shapes_gray,127,255,cv2.THRESH_BINARY)

# Opening
morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,ksize=(7,7))
shapes_gray = cv2.morphologyEx(shapes_gray, cv2.MORPH_OPEN, morphKernel)

# Closing
shapes_gray = cv2.morphologyEx(shapes_gray, cv2.MORPH_CLOSE, morphKernel)

# Detect circles
shapes_circles = cv2.HoughCircles(shapes_gray, cv2.HOUGH_GRADIENT, dp=4,
                                   minDist=30, param1=50, param2=180, 
                                   minRadius=45, maxRadius=60)
shapes_circles = np.round(shapes_circles[0,:]).astype('int')

# Draw circles on image
for i in xrange(np.size(shapes_circles,0)):
    cv2.circle(shapes_img, (shapes_circles[i,0],shapes_circles[i,1]),
               shapes_circles[i,2], (0,255,0), 3)


print("The detected circles with diameter grater than 10 pixels (or radius \
grater than 5) are highlighted with green edge on the images, and listed below.\
 On the lists, the first and second columns are the coordinates x and y of each \
 circle's center, and the third column is the radius.\n")

print('* For "circles.png"'+'\n    x    y     r')
print(circles_circles.astype(str))

print('\n* For "shapes_leo.jpg"'+'\n    x    y    r')
print(shapes_circles.astype(str))

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.title.set_text('circles.png')
plt.imshow(circles_img)
ax2 = fig.add_subplot(122)
ax2.title.set_text('shape_leo.jpg')
plt.imshow(shapes_img)
plt.show()
