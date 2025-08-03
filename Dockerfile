FROM python:3.9-slim

# Устанавливаем Chrome и ChromeDriver
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python-зависимости
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY render_fluid.py .

# Запускаем приложение
CMD ["python", "render_fluid.py"]