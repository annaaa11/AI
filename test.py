
# Відкрийте відео з файлу data\lesson7\meter.mp4.
# Проведіть бінарізацію кадрів та збережіть в новий файл.
# Можливо очистіть від шуму або наведіть різкість через
# bilateralFilter

# import cv2
# import numpy as np
#
#
# cap = cv2.VideoCapture(r'data/lesson7/meter.mp4')
#
#
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))# Встановлення FPS
#
# resized_width = int(width * 0.3)
# resized_height = int(height * 0.3)
#
# writer = cv2.VideoWriter('data/lesson7/binar_pw.mp4',
#                          cv2.VideoWriter_fourcc(*'mp4v'),
#                          cap.get(cv2.CAP_PROP_FPS),
#                          (resized_width, resized_height),
#                          isColor=False)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # # Зміна розміру кадру
#     frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
#
#
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     bilateralFilter = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
#
#
#     binar = cv2.adaptiveThreshold(bilateralFilter,  # оригільне зображення(чорнобіле)
#                                   255,  # інтенсивність пікселів білого кольору
#                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # алгоритм як рахувати threshold
#                                   cv2.THRESH_BINARY,  # тип бінарізації
#                                   9,  # розмір ядра\фільра\рамки
#                                   2,  # наскільки сильною є бінарізацію
#                                   )
#
#     # Відображення кадруq
#     cv2.imshow("bilateralFilter", bilateralFilter)
#     cv2.imshow("Binar+bilateralFilter", binar)
#     cv2.waitKey(1)
#
#     writer.write(binar)  # Запис кадру у відео
#
#     # Вихід з циклу при натисканні клавіші 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# writer.release()

import ultralytics
import torch
import langchain
model = ultralytics.YOLO("yolov8n.pt")

import ultralytics
import cv2

# # дістати перший кадр з відео
# cap = cv2.VideoCapture("data/lesson8/cars.mp4")
#
# ret, img = cap.read()
# # =========
#
# # використання моделі
# model = ultralytics.YOLO("yolov8n.pt")
#
# results = model.predict(
#     img,
#     conf=0.25,  # мінімальний відсоток з яким можна визначити об'єкт
#     iou=0.75,    # якщо для двох рамок iuo менший за це число, то вважаємо що це 2 різних об'єкта
#     #classes=[0, 2]  # індекси класів(тип об'єктів) які будуть детектитись
# )
# # results -- список з результами на кожне зображення в predict
#
# # результати для першого зображення
# result = results[0]
#
# #print(result)
#
# # отримати зображення з результатами детекції
# result_img = result.plot()
#
# # словник з назвами класів
# names = result.names
#
# # рамки
# boxes = result.boxes
#
# #print(boxes)
#
# # візуалізація
# cv2.imshow('original', img)
# cv2.imshow('result', result_img)
# cv2.waitKey(0)
#
#
# # отримання рамок для об'єктів
#
# for cls, xyxy in zip(boxes.cls, boxes.xyxy):
#     # print(cls, xyxy)
#
#     # перевести все в int
#     cls = int(cls)
#     x1, y1, x2, y2 = map(int, xyxy)
#
#     #print(cls, x1, y1, x2, y2)
#
#     # отримати назву об'єкти
#     cls_name = names[cls]
#
#     # вирізати рамку з зображення
#     # Region of Interest -- область яка нас цікавить
#     roi = img[y1:y2, x1:x2]
#
#     # візуалізація
#     cv2.imshow(cls_name, roi)
#
# cv2.waitKey(0)

# Отримайте перший кадр з файлу data\lesson8\animals.mp4
# та виведіть його на екран.

# Проведіть детекцію об’єктів зо допомогою YOLO та
# виведіть результати.
# Змініть параметри моделі conf та iou і подивіться як це
# впливає на результат.
# Отримайте рамки для кожного об’єкта, виріжіть їх та
# виведіть як окремі зображення

cap = cv2.VideoCapture(r"data\lesson8\animals.mp4")

#ret, img = cap.read()

#frame = cv2.resize(img, None, fx=0.3, fy=0.3)
#cv2.imshow("img 30%", frame)

#cv2.waitKey(0)

model = ultralytics.YOLO("yolo11s.pt")
#


while True:
     ret, frame = cap.read()
     frame = cv2.resize(frame, None, fx=0.3, fy=0.3)

     if not ret:
         break
     results = model.track(
         frame,
         conf=0.15,  # мінімальний відсоток з яким можна визначити об'єкт
         iou=0.25,  # якщо для двох рамок iuo менший за це число, то вважаємо що це 2 різних об'єкта
         # classes=[14]  # індекси класів(тип об'єктів) які будуть детектитись
     )

     result = results[0]

     #print(result)

     img2 = result.plot()
     cv2.imshow("img plot", img2)
     #cv2.waitKey(0)

     #print(result.boxes)
     boxes = result.boxes
     names = result.names

     for id, cls, xyxy in zip(boxes.id, boxes.cls, boxes.xyxy):
         #     # print(cls, xyxy)
         #
         # перевести все в int
         cls = int(cls)
         id = int(id)

         x1, y1, x2, y2 = map(int, xyxy)

         # print(cls, x1, y1, x2, y2)

         # отримати назву об'єкти
         cls_name = names[cls]

         # вирізати рамку з зображення
         # Region of Interest -- область яка нас цікавить
         roi = frame[y1:y2, x1:x2]

         # візуалізація
         name = f"{cls_name}_{id}"

         #cv2.imshow(name, roi)

     #cv2.waitKey(1)

     if cv2.waitKey(1) & 0xFF == ord('q'):
         break