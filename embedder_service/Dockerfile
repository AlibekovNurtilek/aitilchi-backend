FROM python:3.10-slim

# Установим системные зависимости
RUN apt-get update && apt-get install -y git && apt-get clean

# Установим рабочую директорию
WORKDIR /app

# Скопируем весь проект
COPY . .

# Установим Python зависимости
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Открываем порт и запускаем API
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
