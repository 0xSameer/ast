# coding: utf-8

"""
Compute metrics - BLEU, precision/recall
"""

import nltk.translate.bleu_score
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, modified_precision

import os

class Eval:
    def __init__(self, path: str, n_evals: int) -> None:
        self.ids = []

        with open(os.path.join(path, "eval.ids"), 
                  "r", encoding="utf-8") as id_f:
            self.ids = [line.strip() for line in id_f]

        self.refs = []
        for i in range(n_evals):
            self.refs.append([])
            with open(os.path.join(path, "ref.en{0:d}".format(i)), 
                      "r", encoding="utf-8") as ref_f:
                self.refs[i] = [line.strip().split() for line in ref_f]
        self.refs = list(zip(*self.refs))
    # end init

    def calc_bleu(self, hyps):
        en_hyp = [hyps[u] for u in self.ids]
        smooth_fun = nltk.translate.bleu_score.SmoothingFunction()

        b_score_value = corpus_bleu(self.refs,
                              en_hyp,
                              weights=(0.25, 0.25, 0.25, 0.25),
                              smoothing_function=smooth_fun.method2)

        return b_score_value

