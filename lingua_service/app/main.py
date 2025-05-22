from fastapi import FastAPI
from app.api import morphology, chat
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AITilchi API", version="1.0")

# Подключаем маршруты
app.include_router(morphology.router, prefix="/predict", tags=["Morphology"])
app.include_router(chat.router, tags=["WebSocket Chat"])

