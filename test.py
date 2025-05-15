import numpy as np
import cv2


# Завдання 1
# Відкрийте зображення
# data/lesson1/Lenna.png
# Створіть маску для пікселів з
# інтенсивністю більше 128.
# Усі пікселі які відповідають цій масці
# замінити на 255.
# Усі пікселі які не відповідають цій
# масці замінити на 0.

img = cv2.imread("data/lesson1/Lenna.png", # шлях до файлу
                 cv2.IMREAD_GRAYSCALE          # зображення чорнобіле
                 )
cv2.imshow(' ', img)
cv2.waitKey(0)

# умови з масивами
# маска для пікселів які більше 128
mask = img > 128

print(mask.shape)
print(mask.dtype)

# дісати пікселі, які відповідають масці

# print(img[mask])

img[mask] = 255  # всі пікселі що відповідають масці
img[~mask] = 0   # всі пікселі що не відповідають масці

cv2.imshow('', img)
cv2.waitKey(0)

# Завдання 2
# Відкрийте зображення data/lesson1/baboo.jpg
# Виведіть таке зображення

img = cv2.imread("data/lesson1/baboo.jpg", # шлях до файлу
                 cv2.IMREAD_GRAYSCALE          # зображення чорнобіле
                 )
cv2.imshow(' ', img)
cv2.waitKey(0)

# glaza
segment = img[10:50, 65:-65]
cv2.imshow('glaza', segment)
cv2.waitKey(0)