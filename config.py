# coding: utf-8

'''
Read configuration files for both model initialization and
training
'''

from typing import Tuple, Dict, NamedTuple

import pickle
import os
import json
from enum import Enum

class Config:
    def __init__(self, cfg_path: str) -> None:
        with open(os.path.join(cfg_path, "model_cfg.json"), "r") as model_f:
            self.model = json.load(model_f)
        with open(os.path.join(cfg_path, "train_cfg.json"), "r") as train_f:
            self.train = json.load(train_f)

        # Get vocabulary size to initialize decoder embeddings
        vocab = pickle.load(open(self.train["data"]["vocab_path"], "rb"))
        vocab_size = len(vocab[self.train["data"]['dec_key']]['w2i'])
        self.model["rnn_config"]["dec_vocab_size"] = vocab_size
        print("vocab size {0:s} = {1:d}".format(self.train["data"]['dec_key'], 
                                                vocab_size))

        self.model["model_dir"] = cfg_path

    # end __init__

