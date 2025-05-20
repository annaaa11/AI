
# Відкрийте зображення data/lesson2/darken.png
# Переведіть його в формат HSV
# Далі для каналу value зробіть одну з двох обробок
# 1. Застосуйте вирівнювання гістограм
# 2. Збільшіть значення десь на 20-50%, для цього
# o Помножте усі значення value на відповідне
# число
# o Оскільки ви вийдете за межі діапазону 0-255
# застосуйте
# np.clip(value, 0, 255)
# o Оскільки результат не ціле число
# value.astype(np.unit8)
# o Напишіть для цієї частини функцію з
# utils.trackbar_decorator
# Переведіть результат назад у формат BGR

import cv2
import numpy as np
import utils

#Функція збільшує яскравість (канал Value) у форматі HS
@utils.trackbar_decorator(brightness=(20, 100))  # Відсоток збільшення яскравості (0-100%)
def adjust_brightness(image_hsv, brightness):

    # канал Value
    value = image_hsv[:, :, 2]
    # Збільшуємо яскравість
    value = value * (1 + brightness / 100.0)
    value = np.clip(value, 0, 255)
    image_hsv[:, :, 2] = value.astype(np.uint8)

    result_brightness = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    return result_brightness


image = cv2.imread('data/lesson2/darken.png')

# Перетворення в HSV
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Вирівнювання гістограми
hsv_equalized = image_hsv.copy()  # Створюємо копію для вирівнювання, щоб не змінити image_hsv
value = hsv_equalized[:, :, 2]  # канал Value
equalized_value = cv2.equalizeHist(value)
hsv_equalized[:, :, 2] = equalized_value
result_equalized = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)  # Перетворюємо в BGR





# Відображення результатів
cv2.imshow("Original", image)  # Показуємо оригінальне зображення
cv2.imshow("Histogram Equalized", result_equalized) # Показуємо зображення з вирівняною гістограмою


# збільшення яскравості
result_brightness = adjust_brightness(image_hsv)
cv2.imshow("Brightness Adjusted", result_brightness)  # Показуємо зображення зі збільшеною яскравістю

cv2.waitKey(0)
