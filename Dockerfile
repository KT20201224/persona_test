
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (for matplotlib/seaborn fonts if needed)
RUN apt-get update && apt-get install -y \
    fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Reports directory to be mounted
RUN mkdir -p reports

CMD ["python", "main.py"]
