import cv2
import numpy as np
import math

import random
def drawText(image, txt):
	# font
	font = cv2.FONT_HERSHEY_SIMPLEX
	# org
	org = (50, 50)
	# fontScale
	fontScale = 1
	# Blue color in BGR
	color = (255, 0, 0)
	# Line thickness of 2 px
	thickness = 2
	# Using cv2.putText() method
	image = cv2.putText(image, txt, org, font,
						fontScale, color, thickness, cv2.LINE_AA)
	return

# Ham tinh khoang cach giua hai deim
def distance(p1,p2):
    my_dist = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    return  my_dist

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def order_points(pts):

	rect = np.zeros((4, 2), dtype="float32")

	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):


	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def find_corner_by_rotated_rect(box,approx):
    corner = []
    for p_box in box:
        min_dist = 999999999
        min_p = None
        for p in approx:
            dist = distance(p_box, p[0])
            if dist < min_dist:
                min_dist = dist
                min_p = p[0]
        corner.append(min_p)

    corner = np.array(corner)
    return corner
right_anser = [{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1,13:2,14:3},{0: 2, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1}]
right_anser2 = [{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1,13:2,14:3},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1}]
right_anser3 = [{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1,13:2,14:3},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1}]
right_anser4 = [{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1,13:2,14:3},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1}]
right_anser5 = [{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1,13:2,14:3},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1}]
right_anser6 = [{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1,13:2,14:3},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1}]
right_anser7 = [{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1,13:2,14:3},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1}]
right_anser8 = [{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1,13:2,14:3},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1},{0: 1, 1: 2, 2: 0, 3: 3, 4: 1,5: 1,6: 0,7: 2,8:1,9:0,10:1,11:2,12:1}]

image = cv2.imread('C:/Users/True/Downloads/Q.PNG')
cv2.imshow ('image',image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(gray,200,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,3)
cv2.imshow ('thre',thresh)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True)
approx = cv2.approxPolyDP(contours[1], 0.01 * cv2.arcLength(contours[1], True), True)
rect = cv2.minAreaRect(contours[1])
box = cv2.boxPoints(rect);
corner = find_corner_by_rotated_rect(box,approx)
image = four_point_transform(image,corner)
wrap = four_point_transform(thresh,corner)
cv2.imshow('canganh',image)
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sobelx=cv2.Sobel(grey,cv2.CV_32F,1,0,-1)
sobely=cv2.Sobel(grey,cv2.CV_32F,0,1,-1)
gra1 = cv2.subtract(sobelx, sobely)
gra1 = cv2.convertScaleAbs(gra1)
cv2.imshow ('gra',gra1)
ret, th = cv2.threshold(gra1,60, 255, cv2.THRESH_BINARY)
cv2.imshow ('gra1',th)
kernel=np.ones((3,3),np.uint8)
dilated = cv2.dilate(th,kernel, iterations=3)
#cv2.imshow ('a',dilated)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (22, 11))
closed1 = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
cv2.imshow ('th',closed1)
contours, h = cv2.findContours(closed1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:5]
t=0
image1=image
X=[0,0,0,0,0]
Y=[0,0,0,0,0]
W=[0,0,0,0,0]
H=[0,0,0,0,0]
for c in contours:
    t=t+1
    x, y, w, h = cv2.boundingRect(c)
    if (w / (h) > 2):
        continue

    ROI = image[y:y + h + 25, x:x + w]
    X[t] = x
    print (x)
    Y[t] = y
    W[t] = w
    H[t] = h
    cv2.imwrite('opencv' + str(t) + '.png', ROI)
    cv2.imshow('t' + str(t), ROI)
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if (w / (h) <0.3):
        continue
    ROI1 = image[y:y + h-30, x+80:x + w]
    ROI2= image[y:y + h, x:x + w]

    cv2.imwrite('opencv.png', ROI1)
    cv2.imwrite('opencvR.png', ROI2)
    cv2.imshow ('MADE',ROI2)
image = cv2.imread('G:/PycharmProjects/cam/cam1/opencv.png')
#cv2.imshow ('Ima',image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
#cv2.imshow ('Thresg',thresh)
    #closed2 = cv2.erode(thresh, None, iterations=1)
    #thresh = cv2.dilate(closed2, None, iterations=1)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print (len(contours))
tickcontours = []

for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    ar = w / float(h)
    if (w >= 13 and h >= 13 and 0.8 <= ar <= 1.2):
        tickcontours.append(c)
        continue
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
#cv2.imshow ('A2',image)
#tickcontours = sort_contours(tickcontours, method="top-to-bottom")[0]

for (q, i) in enumerate(np.arange(0, len(tickcontours), 8)):
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cnts = sort_contours(tickcontours[i:i + 8])[0]

        #cv2.drawContours(image, cnts, -1, color, 3)
    choice = (0, 0)
    total = 0
    for (j, c) in enumerate(cnts):
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
        if total > choice[0]:
            choice = (total, j)
    D= (choice[1]+1)


correct = 0
for t in range(1,5):
    image = cv2.imread('G:/PycharmProjects/cam/cam1/opencv'+str(t)+'.png')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
    #cv2.imshow ('Ga',thresh)
    #closed2 = cv2.erode(thresh, None, iterations=1)
    #thresh = cv2.dilate(closed2, None, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tickcontours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        if (w >= 13 and h >= 13 and 0.8 <= ar <= 1.2):
            tickcontours.append(c)
    tickcontours = sort_contours(tickcontours, method="top-to-bottom")[0]

    for (q, i) in enumerate(np.arange(0, len(tickcontours), 4)):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cnts = sort_contours(tickcontours[i:i + 4])[0]
        #cv2.drawContours(image, cnts, -1, color, 3)
        choice = (0, 0)
        total = 0
        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            if total > choice[0]:
                choice = (total, j)

        current_right = right_anser[t-1][q]

        if current_right == choice[1]:
            color = (0, 255, 0)
            correct += 1
        else:
            color = (0, 0, 255)
        cv2.drawContours(image, [cnts[current_right]], -1, color, 3)
        #cv2.imshow ('TN',image)
    #cv2.imshow('t',image1[(Y[1]):(Y[1]) + H[1] + 25,X[1]:X[1] +W[1]])
    image1[Y[t]:Y[t] + H[t]+25 , X[t]:X[t] + W[t]]=image
correct=correct/50*10
drawText(image1 , str(round(correct,2))+" Diem"+"     "+"De:  " + str(round(D,20)))

cv2.imshow("AA",image1)

cv2.waitKey()
