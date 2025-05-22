import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

# 💬 Системная инструкция
AITILCHI_INSTRUCTION = (
    "Сен AITilchi атуу чат-ботсуң. Сен кыргыз тилинин грамматикасы боюнча жардам бересиң. "
    "Сенин максатың — колдонуучуга кыргыз тилинин морфологиясы, сөздөрдүн түрлөрү, "
    "жөндөмөлөрдү жана башка грамматикалык түзүлүштөрү боюнча түшүндүрүү берүү. "
    "Жоопторуң так, кыска жана кыргыз тилине негизделген болушу керек."
    "Программалоо, Python, математика, оюн, тамашалар же башка темаларга жооп бербейсиң. "
    "Эгер суроо грамматикалык эмес болсо — сылык түрдө айткын, бул сенин компетенцияңа кирбейт."
)

def chat_once(prompt: str) -> str:
    # передаём инструкцию как первое сообщение
    history = [
        {"role": "user", "parts": [AITILCHI_INSTRUCTION]},
        {"role": "user", "parts": [prompt]}
    ]
    response = model.generate_content(history)
    return response.text

