FROM python:3.10

# Обновление и установка системных зависимостей
RUN apt-get update && \
    apt-get install -y gcc g++ libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/

RUN chmod +x /app/udpipe

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
