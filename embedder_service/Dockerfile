FROM python:3.10

# Установка системных зависимостей (только необходимые)
RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копируем только зависимости, чтобы использовать кэш Docker
COPY requirements.txt .

# Обновляем pip и устанавливаем зависимости без кэша
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Копируем остальной проект
COPY . .

# Открываем порт и запускаем сервис
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
