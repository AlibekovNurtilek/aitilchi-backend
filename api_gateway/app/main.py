from fastapi import FastAPI
from pydantic import BaseModel
import httpx

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
async def analyze(input: TextInput):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://tokenizer_service:8000/tokenize",
            json={"text": input.text}
        )
        tokenizer_output = response.json()
    
    return {
        "tokens_conllu": tokenizer_output["conllu"]
    }
