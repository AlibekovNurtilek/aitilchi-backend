import subprocess
import tempfile
import os

UDPIPE_PATH = "./udpipe"
MODEL_PATH = "./kyrgyz_tokenizer"

def run_udpipe_tokenizer(text: str) -> str:
    # Создаём временный файл с текстом
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as input_file:
        input_file.write(text)
        input_file_path = input_file.name

    try:
        # Запускаем udpipe
        result = subprocess.run(
            [UDPIPE_PATH, "--tokenize", MODEL_PATH, input_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"UDPipe error: {result.stderr}")

        return result.stdout

    finally:
        os.remove(input_file_path)
