import cv2
import numpy as np
import os
import utils

# Налаштування
canvas_size = (2000, 2000)
frame_size = (800, 800)
duration = 10  # тривалість відео в секундах
fps = 30
total_frames = duration * fps

# Завантаження зображень
images_folder = 'data/lesson many/cells'  # папка з клітинами
image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder)]
images = [cv2.imread(img) for img in image_files]

# Створення полотна
canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 0  # білий фон

# Розміщення клітин
positions = [(200, 200), (1000, 300), (500, 1200), (1500, 1500)]  # координати розміщення
positions = np.random.randint(200, canvas_size[0]-200,
                              size=(len(images), 2))

for img, pos in zip(images, positions):
    resized_img = cv2.resize(img, (200, 200))  # змінюємо розмір клітини
    mask = resized_img.sum(axis=-1) != 0
    x, y = pos
    roi = canvas[y:y+resized_img.shape[0], x:x+resized_img.shape[1]]# = img
    roi[mask] = resized_img[mask]

canvas = utils.add_gaussian_noise(canvas)
# Підготовка відеозапису
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('cells_video_opencv.mp4', fourcc, fps, frame_size)

# Створення кадрів
for frame_idx in range(total_frames):
    t = frame_idx / total_frames * duration

    zoom = 1 + 0.5 * t / duration  # від 1x до 1.5x збільшення
    center_x = int(canvas_size[0] / 2 + 300 * np.sin(2 * np.pi * t / duration))
    center_y = int(canvas_size[1] / 2 + 300 * np.cos(2 * np.pi * t / duration))

    width = int(frame_size[0] / zoom)
    height = int(frame_size[1] / zoom)

    x1 = np.clip(center_x - width // 2, 0, canvas_size[0] - width)
    y1 = np.clip(center_y - height // 2, 0, canvas_size[1] - height)

    cropped = canvas[y1:y1+height, x1:x1+width]
    frame = cv2.resize(cropped, frame_size)

    out.write(frame)

out.release()
print("Відео збережено як 'cells_video_opencv.mp4'")