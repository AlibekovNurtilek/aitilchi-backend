from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from app.embedder import process_conllu_file
import tempfile
import sys, os

# Добавляем путь к wembeddings.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.wembeddings import WEmbeddings

app = FastAPI()

# Глобальный кэш инициализированных WEmbeddings
WEMBEDDER = WEmbeddings()

@app.post("/embed")
async def embed_file(
    file: UploadFile = File(...),
    model: str = Form("bert-base-multilingual-uncased-last4")
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".conllu") as tmp_in:
        tmp_in.write(await file.read())
        input_path = tmp_in.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp_out:
        output_path = tmp_out.name

    # Используем заранее загруженный embedder
    process_conllu_file(input_path, output_path, model_name=model, embedder=WEMBEDDER)

    return FileResponse(output_path, filename="embeddings.npz", media_type="application/octet-stream")
