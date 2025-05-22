import numpy as np
from conllu import parse_incr

def process_conllu_file(input_path: str, output_path: str, model_name: str, embedder):
    sentences = []

    with open(input_path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            tokens = [token["form"] for token in tokenlist if isinstance(token["id"], int)]
            if tokens:
                sentences.append(tokens)

    if not sentences:
        raise ValueError("Файл пуст или содержит только служебные строки.")

    # Генерация эмбеддингов
    embs = embedder.compute_embeddings(model=model_name, sentences=sentences)

    import zipfile

    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for i, emb in enumerate(embs):
            with archive.open(f"arr_{i}", mode="w") as f:
                np.save(f, emb.astype(np.float32))  # или args.dtype, если параметризовано

