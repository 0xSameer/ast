# coding: utf-8

'''
Dataloader

Loads Fisher data, and creates batches, and evaluation files

Author: Sameer Bansal
'''

from preprocessing import prep_buckets
from eval import Eval

class DataLoader:
    def __init__(self):
        self.map = {}
        self.vocab = {}
        self.info = {}

    def get_batch(self, batch_size, set_key, labels=False):
        raise NotImplementedError


class FisherDataLoader(DataLoader):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        print("Loading data dictionaries")
        self.map = pickle.load(open(data_cfg["map_path"], "rb"))
        self.vocab = pickle.load(open(data_cfg["vocab_path"], "rb"))
        self.info = pickle.load(open(data_cfg["info_path"], "rb"))

        print("Organising data into buckets")
        self.buckets = prep_buckets.buckets_main(self.model_dir, 
                                            data_cfg['buckets_num'], 
                                            data_cfg['buckets_width'], 
                                            key="sp", 
                                            scale=data_cfg['buckets_width'], 
                                            seed='haha',
                                            info_path=data_cfg['info_path'])


        print("Loading references for evaluation")
        evals_path = os.path.join(data_cfg['refs_path'], 
                                  data_cfg['dev_set'])
        self.metrics = Eval(evals_path, data_cfg['n_evals'])

    def _drop_frames(self, x_data, drop_rate):
        sp_mask = xp.ones(len(x_data), dtype=xp.float32)
        num_drop_frame = int(drop_rate * len(x_data))
        if num_drop_frame > 0:
            inds=np.random.choice(np.arange(len(x_data)),size=num_drop_frame)
            sp_mask[inds] = 0
            masked_x = x_data * sp_mask[:,xp.newaxis]
            return masked_x
        else:
            return x_data

    def load_speech(self, utt, set_key, max_sp):
        # Path for speech files
        SP_PATH = os.path.join(self.cfg.train["speech_path"], set_key)
        utt_path = os.path.join(SP_PATH, "{0:s}.npy".format(utt))
        if not os.path.exists(utt_path):
            utt_path = os.path.join(SP_PATH, utt.split('_',1)[0], 
                                    "{0:s}.npy".format(utt))
        x_data = xp.load(utt_path)[:max_sp]
        # Drop frames if training
        if "train" in set_key and self.cfg.train["zero_input"] > 0:
            x_data = self._drop_frames(x_data, self.cfg.train["zero_input"])

        return x_data


    def get_batch(self, batch_size, set_key, labels=False):
        batches = []
        
        num_b = self.buckets[set_key]["num_b"]
        width_b = self.buckets[set_key]["width_b"]
        max_sp = (num_b+1)*width_b

        
        if labels:
            dec_key = self.cfg.train["dec_key"]
            max_pred = self.cfg.train["max_pred"]

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
            batch_data = {"X": []}
            if labels:
                batch_data["y"] = []

            for u in utts:
                batch_data["X"].append(self.load_speech(u, set_key, max_sp))
                if labels:                    
                    en_ids = [self.vocab[dec_key]['w2i'].get(w, UNK_ID) 
                              for w in self.map[set_key][u][dec_key]]

                    y_ids = [GO_ID] + en_ids[:max_pred-2] + [EOS_ID]
                    batch_data["y"].append(xp.asarray(y_ids, dtype=xp.int32))

            # end for utts
            batch_data['X'] = F.pad_sequence(batch_data['X'], padding=PAD_ID)
            if labels:
                batch_data['y'] = F.pad_sequence(batch_data['y'], 
                                                 padding=PAD_ID)

            yield batch_data
