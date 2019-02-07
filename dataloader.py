# coding: utf-8

'''
Dataloader

Loads Fisher data, and creates batches, and evaluation files

Author: Sameer Bansal
'''

from preprocessing import prep_buckets
from eval import Eval
import cupy
from chainer import cuda, Variable
import chainer.functions as F
import numpy as np
import pickle
import os

import random

import chainer.functions as F

# Special vocabulary symbols - we always put them at the start.

class SYMBOLS:
    PAD = b"_PAD"
    GO = b"_GO"
    EOS = b"_EOS"
    UNK = b"_UNK"
    START_VOCAB = [PAD, GO, EOS, UNK]

    PAD_ID = 0
    GO_ID = 1
    EOS_ID = 2
    UNK_ID = 3


class DataLoader:
    def __init__(self):
        self.map = {}
        self.vocab = {}
        self.info = {}

    def get_batch(self, batch_size, train=True, labels=False):
        raise NotImplementedError


class FisherDataLoader(DataLoader):
    def __init__(self, data_cfg, model_dir, gpuid):
        super().__init__()
        self.gpuid = gpuid
        self.data_cfg = data_cfg
        self.model_dir = model_dir
        print("Loading data dictionaries")
        self.map = pickle.load(open(data_cfg["map_path"], "rb"))
        self.vocab = pickle.load(open(data_cfg["vocab_path"], "rb"))
        self.info = pickle.load(open(data_cfg["info_path"], "rb"))

        print("Organising data into buckets")
        self.buckets = prep_buckets.buckets_main(self.model_dir,
                                            data_cfg['buckets_num'],
                                            data_cfg['buckets_width'],
                                            key="sp",
                                            scale=data_cfg["train_scale"],
                                            seed='haha',
                                            info_path=data_cfg['info_path'])

        # Get total number of utterances
        self.n_utts = {}
        for key in self.buckets:
            self.n_utts[key] = len([u for bucket in self.buckets[key]["buckets"] for u in bucket])

        # print("Loading references for evaluation")
        # self.refs = {}
        # for key in self.buckets:
        #     evals_path = os.path.join(data_cfg['refs_path'], key)
        #     if os.path.exists(evals_path):
        #         print("loading refs for: {0:s}".format(key))
        #     self.refs[key] = Eval(evals_path, data_cfg['n_evals'])


    def _drop_frames(self, x_data, drop_rate):
        xp = cuda.cupy if self.gpuid >= 0 else np
        sp_mask = xp.ones(len(x_data), dtype=xp.float32)
        num_drop_frame = int(drop_rate * len(x_data))
        if num_drop_frame > 0:
            inds=np.random.choice(np.arange(len(x_data)),size=num_drop_frame)
            sp_mask[inds] = 0
            masked_x = x_data * sp_mask[:,xp.newaxis]
            return masked_x
        else:
            return x_data

    def _load_speech(self, utt, set_key, max_sp):
        xp = cuda.cupy if self.gpuid >= 0 else np
        # Path for speech files
        SP_PATH = os.path.join(self.data_cfg["speech_path"], set_key)
        utt_path = os.path.join(SP_PATH, "{0:s}.npy".format(utt))
        if not os.path.exists(utt_path):
            utt_path = os.path.join(SP_PATH, utt.split('_',1)[0],
                                    "{0:s}.npy".format(utt))
        x_data = xp.load(utt_path)[:max_sp]
        # Drop frames if training
        if "train" in set_key and self.data_cfg["zero_input"] > 0:
            x_data = self._drop_frames(x_data, self.data_cfg["zero_input"])

        return x_data


    def get_batch(self, batch_size, set_key, train, labels=False):
        xp = cuda.cupy if self.gpuid >= 0 else np

        batches = []

        num_b = self.buckets[set_key]["num_b"]
        width_b = self.buckets[set_key]["width_b"]
        max_sp = (num_b+1)*width_b


        if labels:
            dec_key = self.data_cfg["dec_key"]
            max_pred = self.data_cfg["max_pred"]

        for b, bucket in enumerate(self.buckets[set_key]["buckets"]):
            # Shuffle utterances in a bucket
            random.shuffle(bucket)
            for i in range(0, len(bucket), batch_size):
                # append utterances, and the width of the current batch
                # width of 10, implies 10 speech frames = 10 * 10ms = 100ms
                batches.append((bucket[i:i+batch_size], (b+1)*width_b))
        # end for

        # Shuffle all the batches
        random.shuffle(batches)

        # Generator for batches
        for (utts, b) in batches:
            batch_data = {"X": [], "utts": []}

            if labels:
                batch_data["y"] = []

            for u in utts:
                batch_data["X"].append(self._load_speech(u, set_key, max_sp))
                if labels:
                    en_ids = [self.vocab[dec_key]['w2i'].get(w, SYMBOLS.UNK_ID)
                              for w in self.map[set_key][u][dec_key]]

                    y_ids = [SYMBOLS.GO_ID] + en_ids[:max_pred-2] + [SYMBOLS.EOS_ID]
                    batch_data["y"].append(xp.asarray(y_ids, dtype=xp.int32))

            # end for utts
            # include the utt ids
            batch_data['utts'].extend(utts)
            batch_data['X'] = F.pad_sequence(batch_data['X'],
                                             padding=SYMBOLS.PAD_ID)
            batch_data['X'].to_gpu(self.gpuid)
            if labels:
                batch_data['y'] = F.pad_sequence(batch_data['y'],
                                             padding=SYMBOLS.PAD_ID)
                batch_data['y'].to_gpu(self.gpuid)

            yield batch_data


    def get_hyps(self, preds):
        dec_key = self.data_cfg["dec_key"]
        join_str = ' ' if dec_key.endswith('_w') else ''
        en_hyps = {}
        for utt, p in preds:
            en_hyps[utt] = []
            if type(p) == list:
                t_str = join_str.join([self.vocab[dec_key]['i2w'][i].decode()
                                       for i in p if i >= len(SYMBOLS.START_VOCAB)])
                if "bpe_w" in dec_key:
                    t_str = t_str.replace("@@ ", "")
                # if t_str.find(SYMBOLS.EOS.decode()) >= 0:
                #     t_str = t_str[:t_str.find(SYMBOLS.EOS.decode())]
                en_hyps[utt].extend(t_str.strip().split())
            # end if prediction contains text
        # end for all utts
        return en_hyps

class GlobalPhoneDataLoader(DataLoader):
    def __init__(self, data_cfg, model_dir, gpuid):
        super().__init__()
        self.gpuid = gpuid
        self.data_cfg = data_cfg
        self.model_dir = model_dir
        print("Loading data dictionaries")
        self.map = pickle.load(open(data_cfg["map_path"], "rb"))
        self.vocab = pickle.load(open(data_cfg["vocab_path"], "rb"))
        self.info = pickle.load(open(data_cfg["info_path"], "rb"))

        print("loading speech data from: {0:s}".format(data_cfg["speech_path"]))
        self.speech_data = pickle.load(open(data_cfg["speech_path"], "rb"))

        print("Organising data into buckets")
        self.buckets = prep_buckets.buckets_main(self.model_dir,
                                            data_cfg['buckets_num'],
                                            data_cfg['buckets_width'],
                                            key="sp",
                                            scale=data_cfg["train_scale"],
                                            seed='haha',
                                            info_path=data_cfg['info_path'])

        # Get total number of utterances
        self.n_utts = {}
        for key in self.buckets:
            self.n_utts[key] = len([u for bucket in self.buckets[key]["buckets"] for u in bucket])

        # print("Loading references for evaluation")
        # self.refs = {}
        # for key in self.buckets:
        #     evals_path = os.path.join(data_cfg['refs_path'], key)
        #     if os.path.exists(evals_path):
        #         print("loading refs for: {0:s}".format(key))
        #     self.refs[key] = Eval(evals_path, data_cfg['n_evals'])


    def _drop_frames(self, x_data, drop_rate):
        xp = cuda.cupy if self.gpuid >= 0 else np
        sp_mask = xp.ones(len(x_data), dtype=xp.float32)
        num_drop_frame = int(drop_rate * len(x_data))
        if num_drop_frame > 0:
            inds=np.random.choice(np.arange(len(x_data)),size=num_drop_frame)
            sp_mask[inds] = 0
            masked_x = x_data * sp_mask[:,xp.newaxis]
            return masked_x
        else:
            return x_data

    def _load_speech(self, utt, set_key, max_sp):
        xp = cuda.cupy if self.gpuid >= 0 else np
        x_data = xp.asarray(self.speech_data[set_key][utt][:max_sp])
        # Drop frames if training
        if "train" in set_key and self.data_cfg["zero_input"] > 0:
            x_data = self._drop_frames(x_data, self.data_cfg["zero_input"])

        return x_data


    def get_batch(self, batch_size, set_key, train, labels=False):
        xp = cuda.cupy if self.gpuid >= 0 else np

        batches = []

        num_b = self.buckets[set_key]["num_b"]
        width_b = self.buckets[set_key]["width_b"]
        max_sp = (num_b+1)*width_b


        if labels:
            dec_key = self.data_cfg["dec_key"]
            max_pred = self.data_cfg["max_pred"]

        for b, bucket in enumerate(self.buckets[set_key]["buckets"]):
            # Shuffle utterances in a bucket
            random.shuffle(bucket)
            for i in range(0, len(bucket), batch_size):
                # append utterances, and the width of the current batch
                # width of 10, implies 10 speech frames = 10 * 10ms = 100ms
                batches.append((bucket[i:i+batch_size], (b+1)*width_b))
        # end for

        # Shuffle all the batches
        random.shuffle(batches)

        # Generator for batches
        for (utts, b) in batches:
            batch_data = {"X": [], "utts": []}

            if labels:
                batch_data["y"] = []

            for u in utts:
                batch_data["X"].append(self._load_speech(u, set_key, max_sp))
                if labels:
                    en_ids = [self.vocab[dec_key]['w2i'].get(w, SYMBOLS.UNK_ID)
                              for w in self.map[set_key][u][dec_key]]

                    y_ids = [SYMBOLS.GO_ID] + en_ids[:max_pred-2] + [SYMBOLS.EOS_ID]
                    batch_data["y"].append(xp.asarray(y_ids, dtype=xp.int32))

            # end for utts
            # include the utt ids
            batch_data['utts'].extend(utts)
            batch_data['X'] = F.pad_sequence(batch_data['X'],
                                             padding=SYMBOLS.PAD_ID)
            batch_data['X'].to_gpu(self.gpuid)
            if labels:
                batch_data['y'] = F.pad_sequence(batch_data['y'],
                                             padding=SYMBOLS.PAD_ID)
                batch_data['y'].to_gpu(self.gpuid)

            yield batch_data


    def get_hyps(self, preds):
        dec_key = self.data_cfg["dec_key"]
        join_str = ' ' if dec_key.endswith('_w') else ''
        en_hyps = {}
        for utt, p in preds:
            en_hyps[utt] = []
            if type(p) == list:
                t_str = join_str.join([self.vocab[dec_key]['i2w'][i].decode()
                                       for i in p if i >= len(SYMBOLS.START_VOCAB)])
                if "bpe_w" in dec_key:
                    t_str = t_str.replace("@@ ", "")
                # if t_str.find(SYMBOLS.EOS.decode()) >= 0:
                #     t_str = t_str[:t_str.find(SYMBOLS.EOS.decode())]
                en_hyps[utt].extend(t_str.strip().split())
            # end if prediction contains text
        # end for all utts
        return en_hyps
