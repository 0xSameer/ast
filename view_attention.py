
# coding: utf-8

# In[1]:

import os
import sys
import argparse
from nmt_run import *


program_descrp = """
beam search
"""

parser = argparse.ArgumentParser(description=program_descrp)

parser.add_argument('-o','--nmt_path', help='model path',
                    required=True)

parser.add_argument('-n','--N', help='number of hyps',
                    required=True)

parser.add_argument('-k','--K', help='softmax selection',
                    required=True)

parser.add_argument('-s','--S', help='dev/dev2/test',
                    required=True)

parser.add_argument('-w','--W', help='len normalization weight',
                    required=True)

parser.add_argument('--resume', action='store_true',
                        help='Resume the training from snapshot')

args = vars(parser.parse_args())
cfg_path = args['nmt_path']

N = int(args['N'])
K = int(args['K'])
W = float(args['W'])

set_key = args['S']

resume = bool(args['resume'])

# cfg_path = "interspeech/sp_20hrs"

print("-"*80)
print("Using model: {0:s}".format(cfg_path))
print("-"*80)

def get_batch(m_dict, x_key, y_key, utt_list, vocab_dict,
              max_enc, max_dec, input_path='', 
              limit_vocab=False, add_unk=False,
              drop_input_frames=0,
              switchboard=False):
    batch_data = {'X':[], 'y':[], 'r':[], "utts": []}
    # -------------------------------------------------------------------------
    # loop through each utterance in utt list
    # -------------------------------------------------------------------------
    for u in utt_list:
        # ---------------------------------------------------------------------
        #  add X data
        # ---------------------------------------------------------------------
        if x_key == 'sp':
            # -----------------------------------------------------------------
            # for speech data
            # -----------------------------------------------------------------
            if switchboard == False:
                # get path to speech file
                utt_sp_path = os.path.join(input_path, "{0:s}.npy".format(u))
                if not os.path.exists(utt_sp_path):
                    # for training data, there are sub-folders
                    utt_sp_path = os.path.join(input_path,
                                               u.split('_',1)[0],
                                               "{0:s}.npy".format(u))
                if os.path.exists(utt_sp_path):
                    x_data = xp.load(utt_sp_path)[:max_enc]
                    # Drop input frames logic
                    if drop_input_frames > 0:
                        # print("dropping input frames")
                        # print(x_data[:5,:2])
                        x_data = drop_frames(x_data, drop_input_frames)
                        # print(x_data[:5,:2])
                else:
                    # ---------------------------------------------------------
                    # exception if file not found
                    # ---------------------------------------------------------
                    raise FileNotFoundError("ERROR!! file not found: {0:s}".format(utt_sp_path))
                    # ---------------------------------------------------------
            else:
                # print("switchboard")
                x_data = swbd1_data[u][:max_enc]
                # print(x_data.shape)
                # Drop input frames logic
                if drop_input_frames > 0:
                    x_data = drop_frames(x_data, drop_input_frames)
        else:
            # -----------------------------------------------------------------
            # for text data
            # -----------------------------------------------------------------
            x_ids = [vocab_dict[x_key]['w2i'].get(w, UNK_ID) for w in m_dict[u][x_key]]
            x_data = xp.asarray(x_ids, dtype=xp.int32)[:max_enc]
            # -----------------------------------------------------------------
        # ---------------------------------------------------------------------
        if len(x_data) > 0:
            batch_data['X'].append(x_data)
            batch_data['utts'].append(u)
    # -------------------------------------------------------------------------
    # end for all utterances in batch
    # -------------------------------------------------------------------------
    if len(batch_data['X']) > 0:
        batch_data['X'] = F.pad_sequence(batch_data['X'], padding=PAD_ID)
    return batch_data

def get_utt_data(eg_utt, curr_set):
    # get shape
    if "in_path" in m_cfg:
        local_input_path = os.path.join(m_cfg['in_path'], curr_set)
    else:
        local_input_path = os.path.join(m_cfg['data_path'], curr_set)

    width_b = bucket_dict[dev_key]["width_b"]
    num_b = bucket_dict[dev_key]["num_b"]
    utt_list = [eg_utt]

    batch_data = get_batch(map_dict[curr_set],
                           enc_key,
                           dec_key,
                           utt_list,
                           vocab_dict,
                           num_b * width_b,
                           200,
                           input_path=local_input_path)

    return batch_data


# In[8]:


last_epoch, model, optimizer, m_cfg, t_cfg = check_model(cfg_path)

# train_key = m_cfg['train_set']
# dev_key = m_cfg['dev_set']
dev_key = set_key
batch_size=t_cfg['batch_size']
enc_key=m_cfg['enc_key']
dec_key=m_cfg['dec_key']

input_path = os.path.join(m_cfg['data_path'], dev_key)
# -------------------------------------------------------------------------
# get data dictionaries
# -------------------------------------------------------------------------
map_dict, vocab_dict, bucket_dict = get_data_dicts(m_cfg)
batch_size = {'max': 1, 'med': 1, 'min': 1, 'scale': 1}

# In[9]:


random.seed("meh")

# Eval parameters
ref_index = -1
min_pred_len, max_pred_len= 0, m_cfg['max_en_pred']
# min_len, max_len = 0, 10
displayN = 50
m_dict=map_dict[dev_key]
v_dict = vocab_dict[dec_key]
# key = m_cfg['dev_set']



def get_encoder_states():
    rnn_states = {"c": [], "h": []}
    # ---------------------------------------------------------------------
    # get the hidden and cell state (LSTM) of the first RNN in the decoder
    # ---------------------------------------------------------------------
    if model.m_cfg['bi_rnn']:
        for i, (enc, rev_enc) in enumerate(zip(model.rnn_enc,
                                     model.rnn_rev_enc)):
            h_state = F.concat((model[enc].h, model[rev_enc].h))
            rnn_states["h"].append(h_state)
            if model.m_cfg['rnn_unit'] == RNN_LSTM:
                c_state = F.concat((model[enc].c, model[rev_enc].c))
                rnn_states["c"].append(c_state)
    else:
        for enc, dec in zip(model.rnn_enc, model.rnn_dec):
            rnn_states["h"].append(model[enc].h)
            if model.m_cfg['rnn_unit'] == RNN_LSTM:
                rnn_states["c"].append(model[enc].c)
            # end if
        # end for all layers
    # end if bi-rnn
    return rnn_states
    # ---------------------------------------------------------------------


# In[15]:


def get_decoder_states():
    rnn_states = {"c": [], "h": []}
    # ---------------------------------------------------------------------
    # get the hidden and cell state (LSTM) of the first RNN in the decoder
    # ---------------------------------------------------------------------
    for i, dec in enumerate(model.rnn_dec):
        rnn_states["h"].append(model[dec].h)
        if model.m_cfg['rnn_unit'] == RNN_LSTM:
            rnn_states["c"].append(model[dec].c)
        # end if
    # end for all layers
    return rnn_states
    # ---------------------------------------------------------------------


# In[16]:


def set_decoder_states(rnn_states):
    # ---------------------------------------------------------------------
    # set the hidden and cell state (LSTM) for the decoder
    # ---------------------------------------------------------------------
    for i, dec in enumerate(model.rnn_dec):
        if model.m_cfg['rnn_unit'] == RNN_LSTM:
            model[dec].set_state(rnn_states["c"][i], rnn_states["h"][i])
        else:
            model[dec].set_state(rnn_states["h"][i])
        # end if
    # end for all layers
    # ---------------------------------------------------------------------


# In[17]:


def encode_utt_data(X):
    # get shape
    batch_size = X.shape[0]
    # encode input
    model.forward_enc(X)


# In[18]:


def init_hyp():
    beam_entry = {"hyp": [GO_ID], "score": 0}
    beam_entry["dec_state"] = get_encoder_states()
    a_units = m_cfg['attn_units']
    ht = Variable(xp.zeros((1, a_units), dtype=xp.float32))
    beam_entry["attn_v"] = ht
    beam_entry["attn_history"] = []
    return beam_entry



def decode_beam_step(decode_entry, beam_width=3):
    xp = cuda.cupy if model.gpuid >= 0 else np

    with chainer.using_config('train', False):

        word_id, dec_state, attn_v = (decode_entry["hyp"][-1],
                                        decode_entry["dec_state"],
                                        decode_entry["attn_v"])

        # set decoder state
        set_decoder_states(dec_state)
        #model.set_decoder_state()

        # intialize starting word symbol
        #print("beam step curr word", v_dict['i2w'][word_id].decode())
        curr_word = Variable(xp.full((1,), word_id, dtype=xp.int32))

        prob_out = {}
        prob_print_str = []

        # -----------------------------------------------------------------
        # decode and predict
        pred_out, ht, alphas = model.decode(curr_word, attn_v, get_alphas=True)
        # -----------------------------------------------------------------
        # printing conditional probabilities
        # -----------------------------------------------------------------
        pred_probs = xp.asnumpy(F.log_softmax(pred_out).data[0])
        top_n_probs = xp.argsort(pred_probs)[-beam_width:]

        new_entries = []

        curr_dec_state = get_decoder_states()

        # # check top prob EOS:
        # pruned_top_probs = []
        # for pi in top_n_probs:
        #     if top_n_probs[0] == EOS_ID:
        #         if pred_probs[EOS_ID] >= 3*pred_probs[top_n_probs[1]]:

        for pi in top_n_probs[::-1]:
            #print("{0:10s} = {1:5.4f}".format(v_dict['i2w'][pi].decode(), pred_probs[pi]))
            new_entry = {}
            new_entry["hyp"] = decode_entry["hyp"] + [pi]
            #print(new_entry["hyp"])
            new_entry["score"] = decode_entry["score"] + pred_probs[pi]
            new_entry["dec_state"] = curr_dec_state
            new_entry["attn_v"] = ht
            new_entry["attn_history"] = decode_entry["attn_history"] + [xp.squeeze(alphas.data)]

            new_entries.append(new_entry)

    # end with chainer test mode
    return new_entries


# In[24]:


def decode_beam(utt, curr_set, stop_limit=10, max_n=5, beam_width=3):
    with chainer.using_config('train', False):
        batch_data = get_utt_data(utt, curr_set)
        model.forward_enc(batch_data['X'])

        enc_states = F.squeeze(model.enc_states, axis=(0)).data
        # print(enc_states.shape, model.enc_states.shape)

        n_best = []

        if len(batch_data['X']) > 0:
            n_best.append(init_hyp())

            for i in range(stop_limit):
                #print("-"*40)
                #print(i)
                #print("-"*40)
                all_non_eos = [1 if e["hyp"][-1] != EOS_ID else 0 for e in n_best]
                if sum(all_non_eos) == 0:
                    #print("all eos at step={0:d}".format(i))
                    break

                curr_entries = []
                for e in n_best:
                    if e["hyp"][-1] != EOS_ID:
                        #print("feeding", v_dict["i2w"][e["hyp"][-1]])
                        curr_entries.extend(decode_beam_step(e, beam_width=beam_width))
                    else:
                        curr_entries.append(e)

                n_best = sorted(curr_entries, reverse=True, key=lambda t: t["score"])[:max_n]
    return n_best, enc_states



all_valid_utts = [u for b in bucket_dict[set_key]["buckets"] for u in b]

random.shuffle(all_valid_utts)

# all_valid_utts = all_valid_utts[:10]

if resume == False:
    print("Resume={0:d}, starting decoding process".format(resume))
    utt_hyps = {}
    utt_enc = {}
    utt_enc_maxpool = {}
    for u in tqdm(all_valid_utts, ncols=80):
        with chainer.using_config('train', False):
            n_best, enc_states = decode_beam(u, set_key, 
                                 stop_limit=max_pred_len, 
                                 max_n=N, beam_width=K)
            utt_hyps[u] = [(e["hyp"], e["score"], e["attn_history"]) for e in n_best]
            enc_states = xp.asnumpy(enc_states)
            utt_enc[u] = enc_states
            enc_4d = F.expand_dims(F.expand_dims(enc_states,0),0)
            enc_states_pool = F.squeeze(F.max_pooling_nd(enc_4d, 
                                        (enc_4d.shape[-2], 1)))
            utt_enc_maxpool[u] = xp.asnumpy(enc_states_pool.data)

    print("saving hyps")
    pickle.dump(utt_hyps, open(os.path.join(m_cfg["model_dir"],
                                "{0:s}_attn_N-{1:d}_K-{2:d}.dict".format(set_key,N,K)),
                                "wb"))
    # print("saving encoder states")
    # pickle.dump(utt_enc, open(os.path.join(m_cfg["model_dir"],
    #                             "encoder_states.dict"),
    #                             "wb"))
    # print("saving encoder states - max pool")
    # pickle.dump(utt_enc_maxpool, open(os.path.join(m_cfg["model_dir"],
    #                             "encoder_states_maxpool.dict"),
    #                             "wb"))

else:
    print("Resume={0:d}, loading decoding results".format(resume))
    utt_hyps = pickle.load(open(os.path.join(m_cfg["model_dir"],
                                "{0:s}_attn_N-{1:d}_K-{2:d}.dict".format(set_key,N,K)),
                                "rb"))


def clean_out_str(out_str):
    out_str = out_str.replace("`", "")
    out_str = out_str.replace('"', '')
    out_str = out_str.replace('¿', '')
    out_str = out_str.replace("''", "")

    # for BPE
    out_str = out_str.replace("@@ ", "")
    out_str = out_str.replace("@@", "")

    out_str = out_str.strip()
    return out_str


# In[46]:


def get_out_str(h):
    out_str = ""
    if dec_key == "en_w":
        for w in h:
            out_str += "{0:s}".format(w) if (w.startswith("'") or w=="n't") else " {0:s}".format(w)
    elif "bpe_w" in dec_key:
        out_str = " ".join(h)

    elif dec_key == "en_c":
        out_str = "".join(h)

    else:
        out_str = "".join(h)

    out_str = clean_out_str(out_str)
    return out_str


# In[47]:


# MIN_LEN=0
# MAX_LEN=300


# In[50]:

def rerank_hypothesis(beam_hyps, weight=0.8):
    return sorted([(i[0], i[1]/math.pow(len(i[0])-2,weight), len(i[0])) for i in beam_hyps],
       reverse=True, key=lambda t: t[1])

def write_to_file_len_filtered_preds(utts_beam, min_len, max_len):
    filt_utts = []
    for u in utts_beam:
        if (len(map_dict[set_key][u]["es_w"]) >= min_len and
           len(map_dict[set_key][u]["es_w"]) <= max_len):
            filt_utts.append(u)

    filt_utts = sorted(filt_utts)
    print("Utts matching len filter={0:d}".format(len(filt_utts)))

    hyp_path = os.path.join(m_cfg["model_dir"],
                "{0:s}_beam_min-{1:d}_max-{2:d}_N-{3:d}_K-{4:d}.en".format(set_key,
                                                                     min_len,
                                                                     max_len,
                                                                     N,
                                                                     K))
    print("writing hyps to: {0:s}".format(hyp_path))
    with open(hyp_path, "w", encoding="utf-8") as out_f:
        for u in filt_utts:
            hyp = [v_dict['i2w'][i].decode() for i in utts_beam[u][0][0] if i >= 4]
            out_str = get_out_str(hyp)
            out_f.write("{0:s}\n".format(out_str))
    print("written beam")

    len_norm_path = os.path.join(m_cfg["model_dir"],
                "{0:s}_beam_len-norm_min-{1:d}_max-{2:d}_N-{3:d}_K-{4:d}_W-{5:.1f}.en".format(set_key,
                                                                     min_len,
                                                                     max_len,
                                                                     N,
                                                                     K,
                                                                     W))
    print("writing len norm hyps to: {0:s}".format(len_norm_path))
    with open(len_norm_path, "w", encoding="utf-8") as out_f:
        for u in filt_utts:
            if len(utts_beam[u]) > 0:
                new_hyps = rerank_hypothesis(utts_beam[u], weight=W)

                hyp = [v_dict['i2w'][i].decode() for i in new_hyps[0][0] if i >= 4]
            else:
                hyp = []
            out_str = get_out_str(hyp)
            out_f.write("{0:s}\n".format(out_str))
    print("Written len norm utts")

# In[51]:


print("writing to file")
write_to_file_len_filtered_preds(utt_hyps, 0, 300)

print("all done")