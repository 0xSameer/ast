import os
import sys
import pickle
import numpy as np
from collections import Counter
from tqdm import tqdm
import argparse

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


def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("in_path", type=str)
    parser.add_argument("out_path", type=str)


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


# [TODO] -- change file extensions to be language agnostic, or ask user for
# abbreviation, e.g. fr, cr, etc.

args = check_argv()
in_path = args.in_path #e.g. "../../pretrain_AST_input/gpFR/"
out_path = args.out_path #e.g. "../../pretrain_AST_input/gpFR/"
sets = ['train', 'dev', 'test']

def read_bpe_text(c):
    all_words = []
    utt2words = {}
    with open(os.path.join(in_path, "{0:s}.BPE_1000".format(c)), "r") as text_f,\
              open(os.path.join(in_path, "{0:s}.ids".format(c)), "r") as id_f:
        for u, line in tqdm(zip(id_f, text_f)):
            t = line.strip().split()
            utt2words[u.strip()] = t
            all_words.extend(t)
        # end for line
    # end with
    return utt2words, dict(Counter(all_words))


train_bpe_text, train_bpe_words = read_bpe_text("train")

print("# of BPE word types: {0:d}".format(len(train_bpe_words)))

print("Creating vocabulary")
def create_new_vocab(words):
    out = {"w2i":{}, "i2w":{}, "freq":{}}
    START_VOCAB = [SYMBOLS.PAD, SYMBOLS.GO, SYMBOLS.EOS, SYMBOLS.UNK]
    for w in START_VOCAB:
        out['w2i'][w] = len(out["w2i"])
        out["freq"][w] = 1
    #for w in words_list['words']:
    sorted_w = sorted(words.items(), reverse=True, key=lambda t: t[1])
    for w in sorted_w:
        encoded_word = w[0].encode()
        out["w2i"][encoded_word] = len(out["w2i"])
        out["freq"][encoded_word] = w[1]

    out["i2w"] = {val:key for key, val in out["w2i"].items()}
    return out

vocab = {}
vocab["bpe_w"] = create_new_vocab(train_bpe_words)

print(len(vocab['bpe_w']['w2i']))

print("Creating map dictionary")
oov = {}
map_dict = {}
for c in sets:
    oov[c] = []
    print(c)
    map_dict[c] = {}
    all_words = []
    utt2words = {}

    with open(os.path.join(in_path, "{0:s}.BPE_1000".format(c)), "rb") as text_f, \
                            open(os.path.join(in_path, "{0:s}.ids".format(c)), "r") as id_f, \
                            open(os.path.join(in_path, "{0:s}.clean.text".format(c)), "rb") as words_f:
        for i, t, e in zip(id_f, text_f, words_f):
            map_dict[c][i.strip()] = {}
            map_dict[c][i.strip()]["bpe_w"] = t.strip().split()
            map_dict[c][i.strip()]["en_w"] = e.strip().split()
            for w in t.strip().split():
                if w not in vocab["bpe_w"]["w2i"]:
                    oov[c].append(w)
    # end with

print("Saving vocabulary and map")
pickle.dump(map_dict, open("{0:s}/bpe_map.dict".format(out_path), "wb"))
pickle.dump(vocab, open("{0:s}/bpe_train_vocab.dict".format(out_path), "wb"))

print("Finished saving vocab and map...")

print("Reading all the speech data np files into a dictionary")
gp_data = {}
for c in sets:
    gp_data[c] = {}
    for x in tqdm(os.listdir(os.path.join(in_path, c))):
        if x.endswith(".np"):
            temp = np.load(os.path.join(in_path, c, x))
            for k in temp:
                gp_data[c][k] = temp[k]
        # end for
    # end for
# end for

# list(gp_data["dev"].keys())[:10]
# np.mean(gp_data['dev']['FR087_19'])
# gp_data.keys()

print("Created info dictionary")
info = {}
durs = {}
for c in sets:
    print(c)
    info[c] = {}
    durs[c] = []
    for x in tqdm(map_dict[c], ncols=80):
        info[c][x] = {}
        t_data = gp_data[c][x]
        info[c][x]["sp"] = t_data.shape[0]
        info[c][x]["es_w"] = 0
        info[c][x]["es_c"] = 0
        info[c][x]["en_w"] = len(map_dict[c][x]["en_w"])
        info[c][x]["en_c"] = 0
        durs[c].append(t_data.shape[0])


for c in durs:
    print(c)
    print("total hrs = {0:.3f}".format(sum(durs[c]) / 100. / 3600))
    print("min = {0:.2f}, max = {1:.2f}, mean = {2:.2f}".format(np.min(durs[c])/100,
                                                                np.max(durs[c])/100, np.mean(durs[c])/100))


print("Saving info and data dictionaries ...")

pickle.dump(info, open("{0:s}/info.dict".format(out_path), "wb"))
pickle.dump(gp_data, open("{0:s}/data.dict".format(out_path), "wb"))

print("Finished saving info and data dictionaries ... ")


ref_file = "{0:s}/dev.clean".format(out_path)
ref_out_file = "{0:s}/dev.clean.wer.en".format(out_path)

for c in sets:
    with open(os.path.join(in_path, "{0:s}.ids".format(c)), "r", encoding="utf-8") as id_f,\
              open(os.path.join(in_path, "{0:s}.clean.text".format(c)), "r", encoding="utf-8") as words_f,\
              open(os.path.join(in_path, "{0:s}.clean.wer".format(c)), "w", encoding="utf-8") as out_f:
        for i, t in zip(id_f, words_f):
            out_f.write("{0:s} ({1:s})\n".format(t.strip(), i.strip()))


print("finished writing reference files for evaluation ...")

