# # Завдання 1
# # Відкрийте зображення data/lesson3/sonet.png. Проведіть
# # бінарізацію.
# # Обов’язково використайте:
# #  розмиття або наведення різкості
# #  адаптивну бінарізацію
# #  очищеня шумів
#
import numpy as np
import utils
import cv2
#
# #
img = cv2.imread("data/lesson3/sonet.png")
#
# #розмиття або наведення різкості
@utils.trackbar_decorator(alpha=(0,10))
def task3(img, alpha =3.0):
    alpha = float(alpha)
    img1 = img.copy()
    blurrred = cv2.GaussianBlur(img1, (5, 5), 7)

    image = (1 + alpha) * img1 - alpha * blurrred
    image = np.clip(image, 0, 255).astype(np.uint8)

    cv2.imshow("orig", img1)
    cv2.imshow("sharp", image)

    return image

#
#
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# #очищеня шумів
img_gray = cv2.bilateralFilter(img_gray,  # оригільне зображення
                              d=3,  # розмір ядра\фільра\рамки
                              sigmaColor=75,  # впливає на коефіцієнт за кольором
                              sigmaSpace=75,  # вплива на коефіцієнти як в гауса
                              )

#адаптивну бінарізацію
thresh = cv2.adaptiveThreshold(img_gray,  # оригільне зображення(чорнобіле)
                                255, # інтенсивність пікселів білого кольору
                                cv2.ADAPTIVE_THRESH_MEAN_C,  # алгоритм як рахувати threshold
                                cv2.THRESH_BINARY,  # тип бінарізації
                                3,  # розмір ядра\фільра\рамки
                                1.1,  # наскільки сильною є бінарізацію
                                )


cv2.imshow('Original', img)
cv2.imshow('GRAY', img_gray)
cv2.imshow('THRESH', thresh)

cv2.waitKey(0)

task3(img) # #розмиття або наведення різкості


# Відкрийте зображення data/lesson3/sonnet_noised.png.
# Проведіть бінарізацію. Застосуйте код з завдання 1 та
# спробуйте покращити результат

img = cv2.imread("data/lesson3/sonet_noised.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#img_gray = task3(img_gray)

#очищеня шумів
img_gray = cv2.bilateralFilter(img_gray,  # оригільне зображення
                              d=7,  # розмір ядра\фільра\рамки
                              sigmaColor=50,  # впливає на коефіцієнт за кольором
                              sigmaSpace=50,  # вплива на коефіцієнти як в гауса
                              )


#адаптивну бінарізацію
thresh = cv2.adaptiveThreshold(img_gray,  # оригільне зображення(чорнобіле)
                                255, # інтенсивність пікселів білого кольору
                                cv2.ADAPTIVE_THRESH_MEAN_C,  # алгоритм як рахувати threshold
                                cv2.THRESH_BINARY,  # тип бінарізації
                                5,  # розмір ядра\фільра\рамки
                            3,  # наскільки сильною є бінарізацію
                                )



cv2.imshow('Original', img)
cv2.imshow('GRAY', img_gray)
cv2.imshow('THRESH', thresh)

cv2.waitKey(0)
