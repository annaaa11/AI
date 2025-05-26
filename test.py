# Для наступних зображень зображень:
#  data/lesson4/apple.png
#  data/lesson4/apple_noised.png
#  data/lesson4/apple_salt_pepper.png
# використовуючи гаусове розмиття, виявлення країв та
# морфологічні оператори, отримайте краї яблука.

import numpy as np
import utils
import cv2

######data/lesson4/apple.png
img1 = cv2.imread("data/lesson4/apple.png", cv2.IMREAD_GRAYSCALE)

img1 = cv2.GaussianBlur(img1, (5,5), 2 )
eng = cv2.Canny(img1, 80, 200)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilate = cv2.dilate(eng, kernel, iterations=3)
erode = cv2.erode(dilate, kernel, iterations=2)
dilate = cv2.dilate(erode, kernel, iterations=1)


cv2.imshow("original", img1)
#cv2.imshow("eng", eng)
cv2.imshow(" edges of the apple dilate", dilate)
cv2.imshow("edges of the apple erode", erode)
cv2.waitKey(0)

######data/lesson4/apple_noised.png

img2 = cv2.imread("data/lesson4/apple_noised.png", cv2.IMREAD_GRAYSCALE)

img2 = cv2.GaussianBlur(img2, (7,7), 7 )
eng = cv2.Canny(img2, 90, 120)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilate = cv2.dilate(eng, kernel, iterations=2)
erode = cv2.erode(dilate, kernel, iterations=1)
dilate = cv2.dilate(erode, kernel, iterations=1)

cv2.imshow("original", img2)
cv2.imshow("dilate edges of the apple ", dilate)
cv2.imshow("erode edges of the apple ", erode)
cv2.waitKey(0)

##### img3 = cv2.imread("data/lesson4/apple_salt_pepper.png", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("data/lesson4/apple_salt_pepper.png", cv2.IMREAD_GRAYSCALE)
img3 = cv2.GaussianBlur(img3, (9,9), 10 )
eng = cv2.Canny(img3, 82, 98)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
dilate = cv2.dilate(eng, kernel, iterations=2)
erode = cv2.erode(dilate, kernel, iterations=1)


cv2.imshow("original", img3)
cv2.imshow("dilate edges of the apple ", dilate)
cv2.imshow("erode edges of the apple ", erode)
cv2.waitKey(0)

#подбор low, upper
# @utils.trackbar_decorator(low = (0,255), upper = (0,255))
# def GaussCanny(img3, low, upper):
#     img3 = cv2.GaussianBlur(img3, (7,7), 9 )
#     eng = cv2.Canny(img3, low, upper)
#     return eng
#
# GaussCanny(img3)


