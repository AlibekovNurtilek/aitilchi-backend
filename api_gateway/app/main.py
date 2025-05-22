from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Доступ всем
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

# 🔹 Вызов токенизатора
async def get_conllu(text: str) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post("http://tokenizer_service:8000/tokenize", json={"text": text})
        resp.raise_for_status()
        return resp.json()["conllu"]

# 🔹 Вызов эмбеддера
async def get_embeddings(conllu_text: str, model: str = "bert-base-multilingual-uncased-last4") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".conllu") as tmp:
        tmp.write(conllu_text.encode("utf-8"))
        conllu_path = tmp.name

    with open(conllu_path, "rb") as f:
        files = {"file": ("input.conllu", f, "text/plain")}
        data = {"model": model}
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post("http://embedder_service:8000/embed", data=data, files=files)
            response.raise_for_status()
            npz_path = tempfile.NamedTemporaryFile(delete=False, suffix=".npz").name
            with open(npz_path, "wb") as out_f:
                out_f.write(response.content)

    os.remove(conllu_path)  # Удалим .conllu файл после использования
    return npz_path

# 🔹 Вызов грамматического сервиса — теггера
async def call_tagger(conllu_text: str, npz_path: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".conllu") as f:
        f.write(conllu_text.encode("utf-8"))
        conllu_path = f.name

    with open(conllu_path, "rb") as f1, open(npz_path, "rb") as f2:
        files = {
            "conllu_file": ("input.conllu", f1, "text/plain"),
            "npz_file": ("input.npz", f2, "application/octet-stream"),
        }
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post("http://grammar_service:8000/tagger", files=files)
            response.raise_for_status()
            return response.json()["conllu"]

# 🔹 Вызов грамматического сервиса — парсера (если потребуется)
async def call_parser(conllu_text: str, npz_path: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".conllu") as f:
        f.write(conllu_text.encode("utf-8"))
        conllu_path = f.name

    with open(conllu_path, "rb") as f1, open(npz_path, "rb") as f2:
        files = {
            "conllu_file": ("input.conllu", f1, "text/plain"),
            "npz_file": ("input.npz", f2, "application/octet-stream"),
        }
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post("http://grammar_service:8000/parser", files=files)
            response.raise_for_status()
            return response.json()["conllu"]



# ✅ 🔹 Новый: Вызов lingua_service (morphology API)
async def call_lingua_service(word: str) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post("http://lingua_service:8000/predict/", json={"word": word})
        response.raise_for_status()
        return response.json()
    

# 🔹 Главный endpoint для теггинга
@app.post("/tagging")
async def tagging_endpoint(input: TextInput):
    conllu = await get_conllu(input.text)
    npz_path = await get_embeddings(conllu, "xlm-roberta-base-last4")
    tagged_conllu = await call_tagger(conllu, npz_path)

    return {
        "message": "Теггинг выполнен",
        "conllu": tagged_conllu
    }

# 🔹 Промежуточный endpoint (для отладки или ручной проверки)
@app.post("/parsing")
async def parsing_endpoint(input: TextInput):
    conllu = await get_conllu(input.text)
    npz_path = await get_embeddings(conllu, "bert-base-multilingual-uncased-last4")
    parsed_conllu = await call_parser(conllu, npz_path)

    return {
        "message": "Парсинг выполнен",
        "conllu": parsed_conllu
    }


# ✅ 🔹 Новый endpoint: морфология (проксирует как есть)
@app.post("/morphology")
async def morphology_endpoint(input: TextInput):
    result = await call_lingua_service(input.text)
    return result  # ⚠️ Возвращаем точно как есть (для фронта)