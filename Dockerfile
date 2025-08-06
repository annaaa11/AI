FROM python:3.11-slim

# Устанавливаем зависимости и Chromium с драйвером
RUN apt-get update && apt-get install -y \
    chromium chromium-driver wget unzip curl \
    libglib2.0-0 libnss3 libgconf-2-4 libfontconfig1 libx11-6 libx11-xcb1 libxi6 \
    libxcomposite1 libxdamage1 libxrandr2 libxtst6 libxss1 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python зависимости
WORKDIR /app
COPY req.txt .
RUN pip install --no-cache-dir -r req.txt
COPY . .

ENV PATH="/usr/lib/chromium/:$PATH"
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER=/usr/bin/chromedriver

CMD ["python", "test2.py"]
