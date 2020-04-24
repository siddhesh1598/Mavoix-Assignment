# import 
import numpy as np
import argparse
import cv2
import imutils
from imutils.object_detection import non_max_suppression
import pytesseract
import math
import requests

from getMarks import Marks
from extractROI import ROI
'''
# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
				help="path to the input image")
args = vars(ap.parse_args())

image = ROI(args["image"])
'''
def getAnswer(url):

	imagePath = "marksheet.jpg"

	r = requests.get(url, timeout=60)
	print(type(r.content))
	f = open(imagePath, "wb")
	f.write(r.content)
	f.close()

	image = ROI(imagePath)

	def convertMarks(str):
		convert = {ord('0'): 0, ord('1'): 1, ord('2'): 2,
				ord('3'): 3, ord('4'): 4, ord('5'): 5,
				ord('6'): 6,ord('7'): 7,ord('8'): 8, 
				ord('9'): 9,}

		num = 0
		for i in str:
			x = convert[ord(i)]
			num = num*10 + x

		return num


	marks = Marks(image)
	M = convertMarks(marks[1]) 
	P = convertMarks(marks[2])
	C = convertMarks(marks[3])
	avg = ((P+C+M)/300) * 100

	ans = "We recommend you: "

	if avg >= 90:
		ans += "Campus Alpha"
	elif 80 <= avg < 90:
		ans += "Campus Beta"
	elif 70 <= avg < 80:
		ans += "Campus Gamma"
	else:
		ans = "Sorry, we cannot recommend you any campus."


	#print("Your marks: ")
	#print("Physics: {} / 100".format(P))
	#print("Chemistry: {} / 100".format(C))
	#print("Mathematics: {} / 100".format(M))
	#print("Total: {} / 300".format(P+C+M))

	# print("Average: {:.2f}".format((P+C+M)/300))


	return ans


