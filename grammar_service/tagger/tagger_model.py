# tagger/tagger_model.py

import os
import json
import argparse
import sys
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(__file__))
from .aitilchi import AITilchi
from . import dataset

class TaggerModelWrapper:
    def __init__(self, model_path: str):
        # === Load args from options.json ===
        options_path = os.path.join(model_path, "options.json")
        with open(options_path, "r", encoding="utf-8") as f:
            args_dict = json.load(f)

        # Convert to Namespace (like argparse)
        self.args = argparse.Namespace(**args_dict)

        # Post-process args (e.g. tags: str -> list[str], epochs: str -> list[tuple])
        AITilchi.postprocess_arguments(self.args)

        # === Load mappings from model ===
        self.train = dataset.Dataset.load_mappings(os.path.join(model_path, "mappings.pickle"))

        # === Initialize model and build TF graph ===
        self.model = AITilchi(threads=self.args.threads, seed=self.args.seed)
        self.model.construct(self.args, self.train, devs=[], tests=[], predict_only=True)
        self.model.load(model_path, morphodita_dictionary=None)

    def predict(self, conllu_path: str, npz_path: str) -> str:
        """
        Возвращает строку CoNLL-U с результатами анализа.
        """
        test_dataset = dataset.Dataset(
            path=conllu_path,
            train=self.train,
            shuffle_batches=False,
            embeddings=[npz_path]
        )

        # Выполняем предикт
        conllu_str = self.model.predict(test_dataset, evaluating=False, args=self.args)
        return conllu_str
