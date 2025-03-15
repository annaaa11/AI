import cv2
import numpy as np
import ultralytics
import matplotlib.pyplot as plt

# def main():
#     cap = cv2.VideoCapture(0)  # Захоплення відео з вебкамери
#
#     ret, prev_frame = cap.read()
#     prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         #gray = cv2.bilateralFilter(gray, 15, 75, 75)
#
#         # Обчислення оптичного потоку
#         flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#
#         # Обчислення магнітуди та напрямку
#         magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#
#         # Нормалізація магнітуди для візуалізації
#         #mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
#         #mag_norm = np.uint8(mag_norm)
#
#         angle = 255 * angle / np.pi / 2
#
#         hsv = np.zeros_like(frame)
#
#         hsv[:, :, 0] = np.uint8(angle)
#         hsv[:, :, 1] = 255
#         hsv[:, :, 2] = np.minimum(magnitude*4, 255)
#
#         res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#         cv2.imshow('', gray)
#         cv2.imshow('Magnitude of Optical Flow', res)
#
#         prev_gray = gray.copy()
#
#         if cv2.waitKey(1) & 0xFF == 27:
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()



def main():
    cap = cv2.VideoCapture('data/lesson7/coconut.mp4')  # Захоплення відео з вебкамери
    # model = ultralytics.YOLO('yolov8n.pt')
    #
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    #
    # out = cv2.VideoWriter('data/lesson7/clean_coconut.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    line_y = 185
    offset = 10

    # Лічильник кокосів
    coconut_count = 0
    crossed_coconuts = set()

    # Налаштування SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 500  # Мінімальний розмір кокоса
    params.maxArea = 10000

    params.filterByCircularity = True
    params.minCircularity = 0.1  # Кокоси можуть бути неідеально круглі

    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        # mask2 = cv2.inRange(hsv, (250, 100, 100), (255, 255, 255))
        # frame = cv2.bitwise_or(mask1, mask2)
        # sums = frame.sum(axis=1)
        # print(np.where(sums > 20000))
        # plt.plot(sums)
        # plt.show()

        hsv[178] = cv2.addWeighted(hsv[177], 0.8, hsv[182], 0.2, 0)
        hsv[179] = cv2.addWeighted(hsv[177], 0.6, hsv[182], 0.4, 0)
        hsv[180] = cv2.addWeighted(hsv[177], 0.4, hsv[182], 0.6, 0)
        hsv[181] = cv2.addWeighted(hsv[177], 0.2, hsv[182], 0.8, 0)

        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.bilateralFilter(frame, 15, 50, 50)
        # frame = cv2.Canny(frame, 100, 150)
        # frame[170:190] = cv2.fastNlMeansDenoisingColored(frame[170:190])

        #results = model.predict(frame)

        #res = results[0]
        #frame = res.plot()
        # out.write(frame)
        # cv2.imshow('', frame)
        # print(fps)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #blurred = cv2.GaussianBlur(gray, (3, 3), 2)
        blurred = cv2.bilateralFilter(gray, 15, 75, 75)
        # Визначаємо ключові точки (центри кокосів)
        keypoints = detector.detect(blurred)

        # Малюємо лінію відстеження
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)

        for kp in keypoints:
            cx, cy = int(kp.pt[0]), int(kp.pt[1])

            # Малюємо знайдені кокоси
            cv2.circle(frame, (cx, cy), int(kp.size / 2), (0, 255, 0), 2)

            # Перевіряємо, чи центр кокоса перетнув лінію
            if line_y - offset < cy < line_y + offset:
                if (cx, cy) not in crossed_coconuts:
                    crossed_coconuts.add((cx, cy))
                    coconut_count += 1

                    # Відображаємо кількість кокосів
        cv2.putText(frame, f'Coconuts Passed: {coconut_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Показуємо кадр
        cv2.imshow('Coconut Counter', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()