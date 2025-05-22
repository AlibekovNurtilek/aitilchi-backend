from fastapi import FastAPI
from pydantic import BaseModel
from tokenizer import run_udpipe_tokenizer

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/tokenize")
def tokenize_text(input: TextInput):
    result = run_udpipe_tokenizer(input.text)
    return {"conllu": result}
