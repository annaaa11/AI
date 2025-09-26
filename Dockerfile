FROM python:3.9

# Устанавливаем Chromium и зависимости
RUN apt-get update && apt-get install -y \
    chromium \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python-зависимости
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY render_fluid.py .

# Запускаем приложение
CMD ["python", "render_fluid.py"]