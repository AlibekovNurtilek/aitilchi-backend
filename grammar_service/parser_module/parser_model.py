# parser/parser_model.py

import os
import json
import argparse
import sys
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(__file__))
from .udpipe2 import UDPipe2
from . import udpipe2_dataset

class ParserModelWrapper:
    def __init__(self, model_path: str):
        # === Load args from options.json ===
        options_path = os.path.join(model_path, "options.json")
        with open(options_path, "r", encoding="utf-8") as f:
            args_dict = json.load(f)

        # Convert to Namespace (like argparse)
        self.args = argparse.Namespace(**args_dict)

        # Post-process args (e.g. tags: str -> list[str], epochs: str -> list[tuple])
        UDPipe2.postprocess_arguments(self.args)

        # === Load mappings from model ===
        self.train = udpipe2_dataset.UDPipe2Dataset.load_mappings(os.path.join(model_path, "mappings.pickle"))

        # === Initialize model and build TF graph ===
        self.model = UDPipe2(threads=self.args.threads, seed=self.args.seed)
        self.model.construct(self.args, self.train, devs=[], tests=[], predict_only=True)
        self.model.load(model_path, morphodita_dictionary=None)

    def predict(self, conllu_path: str, npz_path: str) -> str:
        """
        Возвращает строку CoNLL-U с результатами анализа.
        """
        test_dataset = udpipe2_dataset.UDPipe2Dataset(
            path=conllu_path,
            train=self.train,
            shuffle_batches=False,
            embeddings=[npz_path]
        )

        # Выполняем предикт
        conllu_str = self.model.predict(test_dataset, evaluating=False, args=self.args)
        return conllu_str
