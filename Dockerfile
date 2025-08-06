FROM python:3.9-slim
RUN apt-get update && apt-get install -y \
    chromium \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    libx11-6 \
    libx11-xcb1 \
    libxi6 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libxtst6 \
    libxss1 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/* \
    && which chromium && echo "Chromium found at $(which chromium)" || echo "Chromium not found"
RUN wget -q https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/138.0.7204.183/linux64/chromedriver-linux64.zip \
    && unzip chromedriver-linux64.zip \
    && mv chromedriver-linux64/chromedriver /usr/bin/chromedriver \
    && chmod +x /usr/bin/chromedriver \
    && rm chromedriver-linux64.zip
WORKDIR /app
COPY req.txt .
COPY frax_last.py .
RUN pip install --no-cache-dir -r req.txt
ENV PYTHONUNBUFFERED=1
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver
RUN curl -I https://facts.frax.finance/fraxlend/pairs || echo "Не удалось получить доступ к fraxlend"
CMD ["python", "frax_last.py"]