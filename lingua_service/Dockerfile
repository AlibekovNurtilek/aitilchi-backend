# ✅ Базовый образ с Python
FROM python:3.10-slim

# 🔧 Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 📂 Создаём рабочую директорию
WORKDIR /app

# 📥 Копируем зависимости
COPY requirements.txt .

# ⬇️ Устанавливаем Python-библиотеки
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 📥 Копируем всё приложение
COPY . .

# 🚀 Запуск FastAPI-приложения
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
