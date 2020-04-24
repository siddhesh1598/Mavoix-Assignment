from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils
from imutils.object_detection import non_max_suppression
import pytesseract
import math

def ROI(imagePath):
	# import image and resize it
	image = cv2.imread(imagePath)
	ratio = image.shape[0] / 800.0
	orig = image.copy()
	image = imutils.resize(image, height=800)

	# convert the image to grayscale, blur it and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 0, 255)

	# find contours in the image and 
	# keeping only the largest one
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
							cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, 
				reverse=True)[:10]

	# loop over contours
	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)
		
		if w > (1.25 * h):
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)	
		

			# if len(approx) == 4:
			screenCnt = approx
			break

	# show contour outline on the image
	cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

	# apply a four point transform
	warped = four_point_transform(orig, 
					screenCnt.reshape(4, 2) * ratio)

	cv2.imwrite("image_new_1.jpg", imutils.resize(warped, height = 800))
	img = imutils.resize(warped, height = 800)
	# img = cv2.imread('image_new_1.jpg')
	resize = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)
	h, w = resize.shape[:2]
	gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
	_,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
	contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


	images = []
	for cnt in contours:
	    size = cv2.contourArea(cnt)
	    if 5000 < size < ((h-100)*(w-100)): 
	        image = resize.copy()
	        # cv2.drawContours(image, [cnt], 0, (0,255,0), 2)
	        # cv2.imshow('img', image)
	        # cv2.waitKey(0)
	        
	        images.append(cv2.boundingRect(cnt))
	        # crop_img = resize[y:y+h, x:x+w]
	        # images.append(crop_img)
	        # cv2.imshow('img', crop_img)
	        # cv2.waitKey(0)
	        
	images = sorted(images, key=lambda r:r[0])[-3]
	# print(images)
	# for (x, y, w, h) in images:
	(x, y, w, h) = images
	crop_img = resize[y:y+h, x:x+w]
	image = imutils.resize(crop_img, width=64)
	cv2.imwrite("sub_marks.jpg", imutils.resize(crop_img, width=64))

	return imutils.resize(crop_img, width=64)