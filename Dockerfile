FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r req.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "frax_land.py"]