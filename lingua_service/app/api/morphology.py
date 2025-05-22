from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import json
import os

from app.api.split_syllable import split_last_syllable
from app.api.fetch_tamga_data import fetch_tamga_data

router = APIRouter()

# === Загрузка модели и словаря ===
model_path = os.path.join("model", "atooch_to_cases_model.keras")
dict_path = os.path.join("model", "atooch_char2idx.json")

model = tf.keras.models.load_model(model_path)
with open(dict_path, "r", encoding="utf-8") as f:
    char2idx = json.load(f)
idx2char = {i: ch for ch, i in char2idx.items()}
EOS = char2idx["<EOS>"]
PAD = char2idx["<PAD>"]

max_len_input = model.input_shape[1]
max_len_output = model.output_shape[0][1]
cases = ["Илик", "Барыш", "Табыш", "Жатыш", "Чыгыш"]

# === Вспомогательные функции ===
def encode(word: str):
    seq = [char2idx.get(ch, PAD) for ch in word] + [EOS]
    return np.array([seq + [PAD] * (max_len_input - len(seq))])

def decode(indices):
    return ''.join(idx2char.get(int(i), '') for i in indices if int(i) not in [EOS, PAD])

# === Запросный формат ===
class WordRequest(BaseModel):
    word: str

# === Эндпоинт инференса ===
@router.post("/")
def predict_cases(req: WordRequest):
    original_input = req.word

    # Шаг 0. Удаляем "ь" или "ъ" в конце слова
    if original_input and original_input[-1] in ('ь', 'ъ'):
        original_input = original_input[:-1]

    # Шаг 1. Разделение слова
    rest_part, last_syllable = split_last_syllable(original_input)

    # Шаг 2. Запоминаем регистр букв в last_syllable
    uppercase_indices = [i for i, c in enumerate(last_syllable) if c.isupper()]

    # Шаг 3. Переводим last_syllable в нижний регистр
    lower_syllable = last_syllable.lower()

    # Шаг 4. Кодируем и предсказываем
    encoded_input = encode(lower_syllable)
    preds = model.predict(encoded_input, verbose=0)
    results = {}

    # Шаг 5. Восстановление предсказаний
    for i, p in enumerate(preds):
        pred_seq = np.argmax(p[0], axis=-1)
        predicted_ending = decode(pred_seq)

        # Восстановим регистр
        predicted_chars = list(predicted_ending)
        for idx in uppercase_indices:
            if idx < len(predicted_chars):
                predicted_chars[idx] = predicted_chars[idx].upper()
        restored_ending = ''.join(predicted_chars)

        # Шаг 6. Корректируем только окончание
        corrected_ending = correct_stem_error(last_syllable, restored_ending)

        # Шаг 7. Собираем полное слово
        full_word = rest_part + corrected_ending
        results[cases[i]] = full_word
        suggestions = []
    # Шаг 8. Получаем внешнюю морфологическую информацию с сайта tamgasoft.kg
    try:
        suggestions, external_info = fetch_tamga_data(original_input)
    except Exception as e:
        external_info = {"error": str(e)}

    return {
        "word": req.word,
        "cases": results,
        "suggestions": suggestions,
        "external_info": external_info
    }




def correct_stem_error(original_last: str, predicted: str) -> str:
    """
    Исправляет ошибку, когда модель случайно подменила основу.
    Сравнивается оригинальный последний слог с началом предсказания.
    """
    original_lower = original_last.lower()
    predicted_lower = predicted.lower()

    # Если предсказание начинается с другого, чем original_last — исправляем
    if not predicted_lower.startswith(original_lower):
        # Отрезаем "лишнюю" подменённую часть и сохраняем только суффикс
        # Пример: predicted="баңдын", original="бад" → суффикс="дын"
        suffix = predicted[len(original_last):] if len(predicted) >= len(original_last) else ''
        return original_last + suffix

    # Если всё ок — ничего не делаем
    return predicted
