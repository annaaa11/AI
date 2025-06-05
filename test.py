# #
# # # Відкрийте зображення data/lesson_seg/tumor1.jpg
# # # Проведіть сегментацію зображення використовуючи
# # # модель data/lesson_seg/brain-tumor-seg.jpg
# # # Визначте площу пухлини в пікселях.
# # # Визначте площу в
# # # (1 піксель – 0,0025
# # # )
# # # В залежності від площі присвойте пухлині певний тип
# # #  <10 – small
# # #  10-25 – middle
# # #  >25 – large
# # # Покажіть пухлину – за допомогою маски усі лишні
# # # пікселі зробіть 0, а як назву зображення використайте її тип
# #
# # import cv2
# # import ultralytics
# # import numpy as np
# #
# # # Завантаження зображення з пухлиною
# # img = cv2.imread("data/lesson_seg/tumor1.jpg")
# #
# # # Завантаження моделі сегментації пухлин
# # model = ultralytics.YOLO("data/lesson_seg/brain-tumor-seg.pt")
# #
# # # Сегментація
# # results = model.predict(img)
# # result = results[0]
# #
# # # Отримання маски
# # if result.masks is None:
# #     print("пухлина не знайдена.")
# #     exit()
# #
# # masks = result.masks.data.numpy().astype(bool)
# # mask = masks[0]
# #
# # # Обчислення площі пухлини в пікселях
# # area_pixels = mask.sum()
# # pixel_to_meter = 0.0025
# # area_meters = area_pixels * pixel_to_meter
# #
# # # Класифікація пухлини за площею
# # if area_meters < 10:
# #     tumor_type = "small"
# # elif area_meters <= 25:
# #     tumor_type = "middle"
# # else:
# #     tumor_type = "large"
# #
# # print(f"Площа пухлини в пікселях: {area_pixels}")
# # print(f"Площа пухлини в см2: {area_meters:.2f}")
# # print(f"Тип пухлини: {tumor_type}")
# #
# # # Виділення пухлини: інші пікселі зануляємо
# #
# # tumor_only = img.copy()
# # tumor_only[~mask] = 0
# # tumor_only[mask] = img[mask]
# #
# # # Показ результату
# # cv2.imshow(tumor_type, tumor_only)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# import cv2
# import ultralytics
#
# model = ultralytics.YOLO('yolo11n-pose.pt')
#
# img = cv2.imread('data/lesson_pose/human.jpg')
#
# results = model.predict(img)
#
# res = results[0]
#
# # result_img = res.plot()
# #
# # # вивід результатів
# # print(res)
# # print(res.keypoints)  # ключові точки
# #
# # cv2.imshow("result", result_img)
# # cv2.waitKey(0)
#
# # координати точок
# xy_coords = res.keypoints.xy
#
# # дістати координати точок для першого об'єктів
# xy_coords = xy_coords[0]
#
# # змінити масив на numpy
# xy_coords = xy_coords.numpy()
#
# # змінити тип даних на int
# xy_coords = xy_coords.astype(int)
#
# # координати правої долоні
# x, y = xy_coords[10]
#
# # намалювати круг в даній точці
# # res_img = cv2.circle(
# #     img,  # зображення на якому намалювати коло
# #     (x, y),   # координати центру кола
# #     5,       # радіус у пікселях
# #     (0, 255, 0),  # колір у bgr форматі(тут -- зелений)
# #     -1      # товщина лінії(-1 -- запоанити коло повністю)
# # )
# #
# # cv2.imshow("right arm", res_img)
# # cv2.waitKey(0)
#
#
# # # перевиріти що права долоня вища за праве коліно
# #
# # # коорлинати долоні
# # x_arm, y_arm = xy_coords[10]
# #
# # # коордлинати коліна
# # x_knee, y_knee = xy_coords[14]
# #
# # # вивід коорлинати
# # print(f"Координати долоні -- {x_arm}, {y_arm}")
# # print(f"Координати коліна -- {x_knee}, {y_knee}")
# #
# # # перевірка
# # if y_arm < y_knee:
# #     print("Долоня вище зо коліно")
# # else:
# #     print("Долоня нижче зо коліно")
#
# # перевірити чи правий плече знаходиться правіше за ліву плече
#
# x_right, y_right = xy_coords[6]
# x_left, y_left = xy_coords[5]
#
# # вивід коорлинати
# print(f"Координати лівого плеча  -- {x_left}, {y_left}")
# print(f"Координати правого плеча -- {x_right}, {y_right}")
#
# # перевірка
# if x_right > x_left:
#     text = "human back"  # людина повернута спиною
# else:
#     text = "human face"  # людина повернута до вас
#
# # нанести текст на зображення
#
# img = cv2.putText(
#     img,  # зображення на яке насти текст
#     text,   # сам текст
#     (50, 350),   # координати тексту
#     cv2.FONT_HERSHEY_SIMPLEX,   # шрифт
#     1.5,   # розмір шрифту
#     (255, 255, 255),   # колір у форматі bgr(тут -- білий)
#     2   # товщина лінії
# )
#
# cv2.imshow("", img)
# cv2.waitKey(0)
#


# Завдання 1
# Відкрийте відео data/lesson_pose/sitting.mp4
# Ваша задача рахувати кількість присідань.
# Отримайте перший кадр та виділіть основні точки.
# Отримайте координати однієї з долонь та лівого коліна.
# Вважайте що людина присіла, коли її рука опустилась
# нижче коліна, і піднялась коли її рука опинилась вище коліна.
# Оскільки на відео є декілька людей то обирайте ту, яка
# знаходиться найближче, тобто в якої найбільша площа
# рамки(можете потренуватись на 200-му кадрі)

import cv2
import ultralytics

model = ultralytics.YOLO('yolo11n-pose.pt')

cap = cv2.VideoCapture('data/lesson_pose/sitting.mp4')

squat_count = 0
is_down = False

count = 0

# for i in range(0,200):
#     red, imag = cap.read()
#
# imag = cv2.resize(imag, None, fx=0.1, fy=0.1)
# results = model.predict(imag)
# res1 = results[0]
# cv2.imshow("", res1.plot())
# #print(res1.boxes)
#
#
# xywh = res1.boxes.xywh.numpy()
# w_xywh = xywh[:,2]
# h_xywh = xywh[:,3]
#
# area = w_xywh * h_xywh
#
# index = area.argmax()
#
# res1 = results[0]
# xy = res1.keypoints.xy.numpy()[index]
# xy = xy.astype(int)
# xknee, yknee = xy[14]
#
# res_img = cv2.circle(
#         imag,  # зображення на якому намалювати коло
#         (xknee, yknee),  # координати центру кола
#         5,  # радіус у пікселях
#         (0, 255, 0),  # колір у bgr форматі(тут -- зелений)
#         -1  # товщина лінії(-1 -- запоанити коло повністю)
# )
# cv2.imshow("", res_img)
# print(w_xywh)
# print(h_xywh)
# print(area)

#cv2.waitKey(0)


while True:
    red, imag = cap.read()
    if not red:
        break

    imag = cv2.resize(imag, None, fx=0.1, fy=0.1)
    results = model.predict(imag)
    res1 = results[0]

    xywh = res1.boxes.xywh.numpy()
    w_xywh = xywh[:,2]
    h_xywh = xywh[:,3]

    area = w_xywh * h_xywh

    index = area.argmax()

    xy = res1.keypoints.xy.numpy()[index]
    xy = xy.astype(int)
    xknee, yknee = xy[14]

    res_img = cv2.circle(
        imag,  # зображення на якому намалювати коло
        (xknee, yknee),  # координати центру кола
        5,  # радіус у пікселях
        (0, 255, 0),  # колір у bgr форматі(тут -- зелений)
        -1  # товщина лінії(-1 -- запоанити коло повністю)
    )

    xarm, yarm = xy[9]

    res_img = cv2.circle(
        imag,  # зображення на якому намалювати коло
        (xarm, yarm),  # координати центру кола
        5,  # радіус у пікселях
        (0, 255, 0),  # колір у bgr форматі(тут -- зелений)
        -1  # товщина лінії(-1 -- запоанити коло повністю)
    )

    if yarm > yknee and not is_down: #рука ниже колена
        squat_count += 1
        is_down = True
    elif yarm < yknee and is_down:
        is_down = False


    res_img = cv2.putText(
        res_img,  # зображення на яке насти текст
        f"count: {squat_count}",   # сам текст
        (50, 50),   # координати тексту
        cv2.FONT_HERSHEY_SIMPLEX,   # шрифт
        1.5,   # розмір шрифту
        (255, 255, 255),   # колір у форматі bgr(тут -- білий)
        2   # товщина лінії
    )



    cv2.imshow("", res_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#cv2.waitKey(0)

