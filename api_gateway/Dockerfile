FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY app/ /app/

# Запускаем FastAPI
CMD ["uvicorn", "app.main", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
