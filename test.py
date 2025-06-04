
# Відкрийте зображення data/lesson_seg/tumor1.jpg
# Проведіть сегментацію зображення використовуючи
# модель data/lesson_seg/brain-tumor-seg.jpg
# Визначте площу пухлини в пікселях.
# Визначте площу в
# (1 піксель – 0,0025
# )
# В залежності від площі присвойте пухлині певний тип
#  <10 – small
#  10-25 – middle
#  >25 – large
# Покажіть пухлину – за допомогою маски усі лишні
# пікселі зробіть 0, а як назву зображення використайте її тип

import cv2
import ultralytics
import numpy as np

# Завантаження зображення з пухлиною
img = cv2.imread("data/lesson_seg/tumor1.jpg")

# Завантаження моделі сегментації пухлин
model = ultralytics.YOLO("data/lesson_seg/brain-tumor-seg.pt")

# Сегментація
results = model.predict(img)
result = results[0]

# Отримання маски
if result.masks is None:
    print("пухлина не знайдена.")
    exit()

masks = result.masks.data.numpy().astype(bool)
mask = masks[0]

# Обчислення площі пухлини в пікселях
area_pixels = mask.sum()
pixel_to_meter = 0.0025
area_meters = area_pixels * pixel_to_meter

# Класифікація пухлини за площею
if area_meters < 10:
    tumor_type = "small"
elif area_meters <= 25:
    tumor_type = "middle"
else:
    tumor_type = "large"

print(f"Площа пухлини в пікселях: {area_pixels}")
print(f"Площа пухлини в см2: {area_meters:.2f}")
print(f"Тип пухлини: {tumor_type}")

# Виділення пухлини: інші пікселі зануляємо

tumor_only = img.copy()
tumor_only[~mask] = 0
tumor_only[mask] = img[mask]

# Показ результату
cv2.imshow(tumor_type, tumor_only)
cv2.waitKey(0)
cv2.destroyAllWindows()