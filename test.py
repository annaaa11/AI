# # Завдання 1
# # Відкрийте зображення data/lesson3/sonet.png. Проведіть
# # бінарізацію.
# # Обов’язково використайте:
# #  розмиття або наведення різкості
# #  адаптивну бінарізацію
# #  очищеня шумів
#
# import numpy as np
# import utils
# import cv2
#
# #
# # img = cv2.imread("data/lesson3/sonet.png")
# #
# # #розмиття або наведення різкості
# @utils.trackbar_decorator(alpha=(0,10))
# def task3(img, alpha =3.0):
#     alpha = float(alpha)
#     img1 = img.copy()
#     blurrred = cv2.GaussianBlur(img1, (5, 5), 7)
#
#     image = (1 + alpha) * img1 - alpha * blurrred
#     image = np.clip(image, 0, 255).astype(np.uint8)
#
#     #cv2.imshow("orig", img1)
#     #cv2.imshow("sharp", image)
#     #cv2.waitKey(0)
#
#
#     return image
# #
# #
# #
# #
# # #cv2.waitKey(0)
# #
# #
# # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #
# # #очищеня шумів
# # img_gray = cv2.bilateralFilter(img_gray,  # оригільне зображення
# #                               d=3,  # розмір ядра\фільра\рамки
# #                               sigmaColor=75,  # впливає на коефіцієнт за кольором
# #                               sigmaSpace=75,  # вплива на коефіцієнти як в гауса
# #                               )
# #
# # #адаптивну бінарізацію
# # thresh = cv2.adaptiveThreshold(img_gray,  # оригільне зображення(чорнобіле)
# #                                 255, # інтенсивність пікселів білого кольору
# #                                 cv2.ADAPTIVE_THRESH_MEAN_C,  # алгоритм як рахувати threshold
# #                                 cv2.THRESH_BINARY,  # тип бінарізації
# #                                 3,  # розмір ядра\фільра\рамки
# #                                 1.1,  # наскільки сильною є бінарізацію
# #                                 )
# #
# #
# # cv2.imshow('Original', img)
# # cv2.imshow('GRAY', img_gray)
# # cv2.imshow('THRESH', thresh)
# #
# # cv2.waitKey(0)
# #
# # task3(img)
#
# # Відкрийте зображення data/lesson3/sonnet_noised.png.
# # Проведіть бінарізацію. Застосуйте код з завдання 1 та
# # спробуйте покращити результат
#
# img = cv2.imread("data/lesson3/sonet_noised.png")
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#
# #img_gray = task3(img_gray)
#
# #очищеня шумів
# img_gray = cv2.bilateralFilter(img_gray,  # оригільне зображення
#                               d=3,  # розмір ядра\фільра\рамки
#                               sigmaColor=70,  # впливає на коефіцієнт за кольором
#                               sigmaSpace=70,  # вплива на коефіцієнти як в гауса
#                               )
#
#
# #адаптивну бінарізацію
# thresh = cv2.adaptiveThreshold(img_gray,  # оригільне зображення(чорнобіле)
#                                 255, # інтенсивність пікселів білого кольору
#                                 cv2.ADAPTIVE_THRESH_MEAN_C,  # алгоритм як рахувати threshold
#                                 cv2.THRESH_BINARY,  # тип бінарізації
#                                 5,  # розмір ядра\фільра\рамки
#                             2.2,  # наскільки сильною є бінарізацію
#                                 )
#
#
#
# cv2.imshow('Original', img)
# cv2.imshow('GRAY', img_gray)
# cv2.imshow('THRESH', thresh)
#
# cv2.waitKey(0)

import cv2
import numpy as np
import utils

# в opencv кольорове зображення у форматі BGR
img = cv2.imread("data/lesson4/castello.png")

# межі шукають на чорнобілому зображені
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# kernel = np.array([[-1, 0, 1],
#                    [-2, 0, 2],
#                    [-1, 0, 1]])
#
# vert = cv2.filter2D(gray, -1, kernel)
#
# kernel = np.array([[-1, -2, -1],
#                    [0, 0, 0],
#                    [1, 2, 1]])
#
# horiz = cv2.filter2D(gray, -1, kernel)
#
# cv2.imshow("original", img)
# cv2.imshow("vertical", vert)
# cv2.imshow("horizontal", horiz)
# cv2.waitKey(0)

# пошук меж

# edged = cv2.Canny(gray,  # зображення де шукаємо межі
#                   100,  # нижня межі інтенсивності межі
#                   150   # верхня межі інтенсивності межі
#                   )
#
# cv2.imshow("original", img)
# cv2.imshow("edged", edged)
# cv2.waitKey(0)

# функція для меж

# @utils.trackbar_decorator(lower=(0, 255), upper=(0, 255))
# def func(img, lower, upper):
#     # перетворення в чорнобіле зображення
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # розмити зображення
#     gray = cv2.GaussianBlur(gray,
#                             (5, 5),
#                             sigmaX=2)
#
#     # алгоритм Canny(пошук меж)
#     edged = cv2.Canny(gray, lower, upper)
#
#     return edged
#
# func(img)

#
# img = cv2.imread("data/lesson4/j.png", cv2.IMREAD_GRAYSCALE)
#
# # ерозія
# # якщо навколо пікселя є хоча б один чорний -- то піксель стає чорним
#
# # піксель по сусідству -- в сежах квадрату 3х3
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# eroded = cv2.erode(img, kernel)
#
# # dilate(розширення?)
# # якщо навколо пікселя є хоча б один білий -- то піксель стає білим
# dilated = cv2.dilate(img, kernel)
#
#
# both = cv2.erode(img, kernel)
# both = cv2.dilate(both, kernel, iterations=2)
#
# cv2.imshow("original", img)
# cv2.imshow("eroded", eroded)
# cv2.imshow("dilate", dilated)
# cv2.imshow("both", both)
# cv2.waitKey(0)

# Відкрийте зображення
# data/lesson4/lego.jpg. Отримайте краї фігур,
# використайте морфологічні оператори для
# усунення неточностей та щоб зробити межі
# жирнішими

# img = cv2.imread("data/lesson4/lego.jpg", cv2.IMREAD_GRAYSCALE)
#
# img = cv2.GaussianBlur(img, (5,5), 2 )
# eng = cv2.Canny(img, 62, 75)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# dilate = cv2.dilate(eng, kernel, iterations=3)
# erode = cv2.erode(dilate, kernel, iterations=3)
#
#
#
# #cv2.imshow("original", img)
# cv2.imshow("eng", eng)
# cv2.imshow("dilate", dilate)
# cv2.imshow("erode", erode)
# cv2.waitKey(0)

# @utils.trackbar_decorator(low = (0,255), upper = (0,255))
# def eng_lego(img, low, upper):
#     img = cv2.GaussianBlur(img, (5,5), 2 )
#     eng = cv2.Canny(img, low, upper)
#     return eng
#
# eng_lego(img)

###2####

img = cv2.imread("data/lesson4/lego.jpg")
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = (45,50,20)
upper = (75,255,255)

mask_green = cv2.inRange(img_hsv, lower, upper)

cv2.imshow("mask_green", mask_green)
cv2.waitKey(0)

# img = cv2.GaussianBlur(img, (5,5), 2 )
# eng = cv2.Canny(img, 62, 75)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# dilate = cv2.dilate(eng, kernel, iterations=3)
# erode = cv2.erode(dilate, kernel, iterations=3)
#
#
#
# #cv2.imshow("original", img)
# cv2.imshow("eng", eng)
# cv2.imshow("dilate", dilate)
# cv2.imshow("erode", erode)
# cv2.waitKey(0)

# @utils.trackbar_decorator(low = (0,255), upper = (0,255))
# def eng_lego(img, low, upper):
#     img = cv2.GaussianBlur(img, (3,3), 4 )
#     eng = cv2.Canny(img, low, upper)
#     return eng
#
# eng_lego(img)