import cv2 as cv
import numpy as np

img = cv.imread('IMG_6796.JPG')
c = 11
img = cv.resize(img, (int(600/img.shape[0]*img.shape[1]), 600))
blurred = cv.GaussianBlur(img, (c, c), 0)
imghsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
print(img[0][0])
print(img[0][0][0])
print(blurred[0,0,:])
aim = np.uint8([[img[0, 0, :]]])
cv.imshow('', aim)
cv.waitKey()
aimhsv = cv.cvtColor(aim, cv.COLOR_BGR2HSV)
cv.imshow('', aimhsv)
cv.waitKey()
mask = cv.inRange(imghsv, np.array([aimhsv[0, 0, 0] - 60, 0, 0]),np.array([aimhsv[0, 0, 0] + 60, 255, 255]))
mask = cv.erode(mask, None, iterations=2)
mask = cv.dilate(mask, None, iterations=2)
dismask = cv.bitwise_not(mask)
img1 = cv.bitwise_and(img, img, mask=dismask)
bg = img.copy()
rows,cols,channels = img.shape
bg[:rows,:cols,:] = [0, 0, 255]
img2 = cv.bitwise_and(bg, bg, mask=mask)
result = cv.add(img1, img2)
cv.imshow('', mask)
cv.waitKey()
cv.imshow('', result)
cv.waitKey()

