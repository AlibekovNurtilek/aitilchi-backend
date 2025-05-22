# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import shutil
import os

from tagger.tagger_model import TaggerModelWrapper
from parser_module.parser_model import ParserModelWrapper

app = FastAPI(title="AITilchi Tagger Service")

# === Инициализация модели при запуске ===
TAGGER_MODEL_PATH = "models/kyrgyz-ud-model"
PARSER_MODEL_PATH = "models/kyrgyz-parser-model"
tagger = TaggerModelWrapper(TAGGER_MODEL_PATH)
parser = ParserModelWrapper(PARSER_MODEL_PATH)

@app.post("/tagger")
async def tagger_endpoint(
    conllu_file: UploadFile = File(...),
    npz_file: UploadFile = File(...)
):
    try:
        # Сохраняем файлы во временную директорию
        with tempfile.TemporaryDirectory() as tmpdir:
            conllu_path = os.path.join(tmpdir, "input.conllu")
            npz_path = os.path.join(tmpdir, "input.npz")

            with open(conllu_path, "wb") as f:
                shutil.copyfileobj(conllu_file.file, f)
            with open(npz_path, "wb") as f:
                shutil.copyfileobj(npz_file.file, f)

            # Предсказание
            conllu_result = tagger.predict(conllu_path, npz_path)

        return JSONResponse(content={"conllu": conllu_result})

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("❌ Ошибка в /tagger:")
        print(tb)
        return JSONResponse(status_code=500, content={"error": str(e), "trace": tb})


    


@app.post("/parser")
async def parser_endpoint(
    conllu_file: UploadFile = File(...),
    npz_file: UploadFile = File(...)
):
    try:
        # Сохраняем файлы во временную директорию
        with tempfile.TemporaryDirectory() as tmpdir:
            conllu_path = os.path.join(tmpdir, "input.conllu")
            npz_path = os.path.join(tmpdir, "input.npz")

            with open(conllu_path, "wb") as f:
                shutil.copyfileobj(conllu_file.file, f)
            with open(npz_path, "wb") as f:
                shutil.copyfileobj(npz_file.file, f)

            # Предсказание
            conllu_result = parser.predict(conllu_path, npz_path)

        return JSONResponse(content={"conllu": conllu_result})

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("❌ Ошибка в /tagger:")
        print(tb)
        return JSONResponse(status_code=500, content={"error": str(e), "trace": tb})
