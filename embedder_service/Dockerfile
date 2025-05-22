FROM python:3.9

# Установка минимальных системных зависимостей
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем только requirements.txt для кэширования зависимостей
COPY requirements.txt .

# Установка зависимостей
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем остальной проект
COPY . .

# Открываем порт приложения
EXPOSE 8000

# Запускаем FastAPI через uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
