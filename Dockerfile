FROM python:3.9-slim

# Устанавливаем зависимости для Chrome и chromedriver
RUN apt-get update && apt-get install -y \
chromium \
chromium-driver \
libglib2.0-0 \
libnss3 \
libgconf-2-4 \
libfontconfig1 \
&& rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY . .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r req.txt

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Запускаем приложение с gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--timeout", "120", "frax_land:app"]