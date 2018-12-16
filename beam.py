# coding: utf-8

"""
Beam search
"""

from nn import NN
from eval import Eval

import argparse
import os
from chainer import serializers

program_descrp = """Beam search to find best predictions for NN model"""


def init_hyp():
    beam_entry = {"hyp": [GO_ID], "score": 0}
    beam_entry["dec_state"] = get_encoder_states()
    a_units = m_cfg['attn_units']
    ht = Variable(xp.zeros((1, a_units), dtype=xp.float32))
    beam_entry["attn_v"] = ht
    beam_entry["attn_history"] = []
    return beam_entry

def rerank_hypothesis(beam_hyps, weight=0.8):
    return sorted([(i[0], i[1]/math.pow(len(i[0])-2,weight), len(i[0])) for i in beam_hyps],
       reverse=True, key=lambda t: t[1])

def get_best_hyps(utts_beam):
    hyps = {}
    for u in utts_beam:

        rerank_hyp = rerank_hypothesis(utts_beam[u], weight=W)

        hyps[u] = [i for i in rerank_hyp[0][0] if i >= 4]


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=program_descrp)
    parser.add_argument('-m','--cfg_path', help='path for model config',
                        required=True)

    parser.add_argument('-n','--N', help='number of hyps',
                    required=True)

    parser.add_argument('-k','--K', help='softmax selection',
                        required=True)

    parser.add_argument('-s','--S', help='dev/dev2/test',
                        required=True)

    parser.add_argument('-w','--W', help='len normalization weight',
                        required=True)

    args = vars(parser.parse_args())

    cfg_path = args['cfg_path']

    N = int(args['N'])
    K = int(args['K'])
    W = float(args['W'])

    set_key = args['S']

    """
    Create the model and load previously stored parameters
    """
    nn = NN(cfg_path)

    """
    Load references for evaluation
    """
    refs_path = os.path.join(nn.cfg.train["data"]["refs_path"],
                             dev_key)
    metrics = Eval(refs_path, nn.cfg.train["data"]["n_evals"])

    random.seed("meh")

    print("-"*80)
    print("Beam for: {0:s} gpu: {1:d}".format(cfg_path, nn.gpuid))
    print("-"*80)

    # Maximum prediction length
    stop_limit = nn.cfg.train["data"]["max_pred"]

    # For each utterance, store the beam results
    utt_hyps = {}

    # Loop over all utterances
    with tqdm(total=n_utts, ncols=80) as pbar, \
         chainer.using_config('train', False):
        for utt in self.data_loader.get_batch(1,
                                              set_key,
                                              train=False,
                                              labels=False):

            # Training mode not enabled
            n_best, enc_states = decode_beam(u, set_key,
                                 stop_limit=max_pred_len,
                                 max_n=N, beam_width=K)
            utt_hyps[u] = [(e["hyp"], e["score"], e["attn_history"])
                            for e in n_best]


    # Save beam results
    print("saving hyps")
    pickle.dump(utt_hyps, open(os.path.join(cfg_path,
                "{0:s}_attn_N-{1:d}_K-{2:d}.beam".format(set_key,N,K)),
                "wb"))


    # Train model
    epoch_loss = nn.train_epoch(train_key)
    # print("Loss = {0:.4f}".format(epoch_loss))
    with open(nn.train_log, mode='a') as train_log:
        # log train loss
        train_log.write("{0:d}, {1:.4f}\n".format(epoch, epoch_loss))
    # Evaluate model
    preds = nn.predict(dev_key)
    hyps = nn.data_loader.get_hyps(preds)
    bleu = metrics.calc_bleu(hyps) * 100

    with open(nn.dev_log, mode='a') as dev_log:
        # log dev bleu
        dev_log.write("{0:d}, {1:.2f}\n".format(epoch, bleu))
    print("BLEU = {0:.2f}".format(bleu))
    print("-"*80)

    # Save model
    if ((epoch % iters_save == 0) or (epoch == max_epoch-1)):
        print("Saving model")
        serializers.save_npz(model_fil.replace(".model", "_{0:d}.model".format(epoch)), nn.model)
        print("Finished saving model")
# -----------------------------------------------------------------------------
