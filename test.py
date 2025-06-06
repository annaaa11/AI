#  Завдання 1
# Відкрийте відео data/lesson_pose/squat.mp4
# Ваша задача рахувати кількість присідань.
# Отримайте перший кадр та виділіть основні точки.
# Отримайте координати 3-ох точок ноги
# Визначте кут між цими трьома точками. Скористайтесь
# функцією utils.get_angle(x1, y1, x2, y2, x3, y3) де x2, y2 –
# координати коліна(центральна точка)
# Запустіть відео та добавте на сам кадр кут згинання ніг.
# Визначіть нижню межу кута(якщо людина опустилась
# нижче  вважаємо що вона достатньо опустилась) та верхню
# межу кута(якщо людина піднялась вище вважаємо що вона
# достатньо піднялась)
# Добавте кількість присідань та
# кут на кожен кадр.

import cv2
import numpy as np
import ultralytics

#Визначте кут між цими трьома точками
def get_angle(x1, y1, x2, y2, x3, y3):
    a = np.array([x1, y1])
    b = np.array([x2, y2])
    c = np.array([x3, y3])

    ab = a - b
    cb = c - b

    dot = ab @ cb
    norm_ab = (ab @ ab) ** 0.5
    norm_cb = (cb @ cb) ** 0.5
    angle = np.arccos(dot / norm_ab / norm_cb)
    angle = angle / np.pi * 180

    return angle

model = ultralytics.YOLO('yolo11n-pose.pt')
cap = cv2.VideoCapture('data/lesson_pose/squat.mp4')

squat_count = 0
is_down = False

LOWER_ANGLE = 75   # присідання виконано
UPPER_ANGLE = 155  # піднявся

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
    results = model.predict(frame)
    res1 = results[0]

    xy = res1.keypoints.xy.numpy()[0]
    xy = xy.astype(int)

    # Точки: таз (hip) = 12, коліно = 14, щиколотка = 16 (для правої ноги)
    x1, y1 = xy[12]
    x2, y2 = xy[14]
    x3, y3 = xy[16]

    angle = int(get_angle(x1, y1, x2, y2, x3, y3))

    # Малюємо ключові точки
    for (x, y) in [ (x1, y1), (x2, y2), (x3, y3) ]:
        frame = cv2.circle(
            frame,  # зображення на якому намалювати коло
            (x, y),  # координати центру кола
            5,  # радіус у пікселях
            (0, 255, 0),  # колір у bgr форматі(тут -- зелений)
            -1  # товщина лінії(-1 -- запоанити коло повністю)
        )

    # Підрахунок присідань
    if angle < LOWER_ANGLE and not is_down:
        is_down = True
    elif angle > UPPER_ANGLE and is_down:
        squat_count += 1
        is_down = False

    # Виводимо кут та кількість присідань

    frame = cv2.putText(
        frame,  # зображення на яке насти текст
        f'Angle: {angle}',  # сам текст
        (50, 100),  # координати тексту
        cv2.FONT_HERSHEY_SIMPLEX,  # шрифт
        1.5,  # розмір шрифту
        (255, 255, 255),  # колір у форматі bgr(тут -- білий)
        2  # товщина лінії
    )

    frame = cv2.putText(
        frame,  # зображення на яке насти текст
        f'Squats: {squat_count}',  # сам текст
        (50, 50),  # координати тексту
        cv2.FONT_HERSHEY_SIMPLEX,  # шрифт
        1.5,  # розмір шрифту
        (255, 255, 255),  # колір у форматі bgr(тут -- білий)
        2  # товщина лінії
    )


    cv2.imshow('Squat Counter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

