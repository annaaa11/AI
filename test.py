
# 1. Відкрийте відео з файлу data\lesson8\meetings.mp4
# Застосуйте детекцію та виведіть результат, підберіть
# параметри
# Можете змінити розмір кадру для кращої візуалізації
# cv2.resize()

import ultralytics
import torch
import langchain
import cv2

model = ultralytics.YOLO("yolo11s.pt")
cap = cv2.VideoCapture(r"data\lesson8\meetings.mp4")

while True:
     ret, frame = cap.read()
     frame = cv2.resize(frame, None, fx=0.3, fy=0.3)

     if not ret:
         break
     results = model.track(
         frame,
         conf=0.35,  # мінімальний відсоток з яким можна визначити об'єкт
         iou=0.55,  # якщо для двох рамок iuo менший за це число, то вважаємо що це 2 різних об'єкта
         # classes=[14]  # індекси класів(тип об'єктів) які будуть детектитись
     )

     result = results[0]

     #print(result)

     img2 = result.plot()
     cv2.imshow("img plot", img2)
     #cv2.waitKey(0)

     # print(result.boxes)
     # boxes = result.boxes
     # names = result.names


     if cv2.waitKey(1) & 0xFF == ord('q'):
         break


# 2. Відкрийте відео з файлу data\lesson8\meetings.mp4
# Застосуйте детекцію та почніть показувати відео з
# моменту, коли людей стало 5


import ultralytics
import torch
import langchain
import cv2

model = ultralytics.YOLO("yolo11s.pt")

cap = cv2.VideoCapture(r"data\lesson8\meetings.mp4")

# ret, frame = cap.read()
# frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
# results = model.track(
#          frame,
#          conf=0.35,  # мінімальний відсоток з яким можна визначити об'єкт
#          iou=0.55,  # якщо для двох рамок iuo менший за це число, то вважаємо що це 2 різних об'єкта
#          # classes=[14]  # індекси класів(тип об'єктів) які будуть детектитись
#      )
#
# result = results[0]
# print(result) ##0: 'person',
#
#
while True:
     ret, frame = cap.read()
     frame = cv2.resize(frame, None, fx=0.3, fy=0.3)

     if not ret:
         break
     results = model.track(
         frame,
         conf=0.35,  # мінімальний відсоток з яким можна визначити об'єкт
         iou=0.55,  # якщо для двох рамок iuo менший за це число, то вважаємо що це 2 різних об'єкта
         classes=[0]  # 0: 'person'
     )

     result = results[0]


     # Подсчёт количества людей
     if result and result.boxes is not None:
         num_people = len(result.boxes)
     else:
         num_people = 0

     # Показываем изображение только если 5 или больше человек
     if num_people >= 5:
         img2 = result.plot()
         cv2.imshow("img plot", img2)

         if cv2.waitKey(1) & 0xFF == ord('q'):
             break

cap.release()
cv2.destroyAllWindows()
     #cv2.waitKey(1)

