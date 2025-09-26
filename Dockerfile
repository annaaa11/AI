FROM python:3.9

# Устанавливаем Chromium и зависимости
RUN apt-get update && apt-get install -y \
    chromium \
    libxss1 \
    fonts-liberation \
    libgbm-dev \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем ChromeDriver версии 140.0.7339.207
RUN wget -q https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/140.0.7339.207/linux64/chromedriver-linux64.zip \
    && unzip chromedriver-linux64.zip \
    && mv chromedriver-linux64/chromedriver /usr/bin/chromedriver \
    && chmod +x /usr/bin/chromedriver \
    && rm chromedriver-linux64.zip

# Устанавливаем Python-зависимости
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY render_fluid.py .

# Запускаем приложение
CMD ["python", "render_fluid.py"]