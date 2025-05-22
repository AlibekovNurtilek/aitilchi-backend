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

    words = [word for sentence in sentences for word in sentence]
    word_ids = [i for i, sentence in enumerate(sentences) for _ in sentence]
    embeddings = np.vstack(embs)

    np.savez_compressed(output_path,
                        words=np.array(words),
                        word_ids=np.array(word_ids),
                        embeddings=embeddings)
