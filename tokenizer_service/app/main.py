from fastapi import FastAPI
from pydantic import BaseModel
from tokenizer import run_udpipe_tokenizer

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/tokenize")
def tokenize_text(input: TextInput):

    input_text = split_punctuations(input.text)
    result = run_udpipe_tokenizer(input_text)
    return {"conllu": result}

import re

def split_punctuations(text: str) -> str:
    """
    Разделяет знаки препинания пробелами (кроме - _ @), чтобы они не слипались со словами.
    Пример: "Менин атым Нуртилек." → "Менин атым Нуртилек ."
    """
    # Разделяем все знаки препинания, кроме - _ @
    pattern = r"([^\w\s@_-])"  # всё, что не буква, не пробел, не -, _, @
    text = re.sub(pattern, r" \1 ", text)

    # Удаляем лишние пробелы
    return re.sub(r"\s+", " ", text).strip()
