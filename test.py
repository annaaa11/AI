
# Відкрийте відео з файлу data\lesson7\meter.mp4.
# Проведіть бінарізацію кадрів та збережіть в новий файл.
# Можливо очистіть від шуму або наведіть різкість через
# bilateralFilter

import cv2
import numpy as np


cap = cv2.VideoCapture(r'data/lesson7/meter.mp4')


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))# Встановлення FPS

resized_width = int(width * 0.3)
resized_height = int(height * 0.3)

writer = cv2.VideoWriter('data/lesson7/binar_pw.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         cap.get(cv2.CAP_PROP_FPS),
                         (resized_width, resized_height),
                         isColor=False)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # # Зміна розміру кадру
    frame = cv2.resize(frame, None, fx=0.3, fy=0.3)


    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bilateralFilter = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)


    binar = cv2.adaptiveThreshold(bilateralFilter,  # оригільне зображення(чорнобіле)
                                  255,  # інтенсивність пікселів білого кольору
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # алгоритм як рахувати threshold
                                  cv2.THRESH_BINARY,  # тип бінарізації
                                  9,  # розмір ядра\фільра\рамки
                                  2,  # наскільки сильною є бінарізацію
                                  )

    # Відображення кадруq
    cv2.imshow("bilateralFilter", bilateralFilter)
    cv2.imshow("Binar+bilateralFilter", binar)
    cv2.waitKey(1)

    writer.write(binar)  # Запис кадру у відео

    # Вихід з циклу при натисканні клавіші 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

writer.release()