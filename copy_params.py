from nn import NN
from eval import Eval

import argparse
import os
from chainer import serializers
import cupy
from chainer import cuda, Function, utils, Variable

xp = cuda.cupy

model1_path = "../safe-copy-ast/experiments/asr_sw_GOLD"
model2_path = "../safe-copy-ast/experiments/pretrain_sw_GOLD"

nn_1 = NN(model1_path)
nn_2 = NN(model2_path)

print(nn_1.model.embed_dec.W.shape)

enc_components = ['CNN_0', 'CNN_0_bn', 'CNN_1', 'CNN_1_bn',
                  'L0_enc', 'L1_enc', 'L2_enc',
                  'L0_rev_enc', 'L1_rev_enc', 'L2_rev_enc']
attn_components = ['attn_Wa', 'context']
dec_components = ['L0_dec', 'L1_dec', 'L2_dec', 'embed_dec', 'out']

def copy_encoder_params(src_model, target_model, copy_cnn=True, copy_lstm=True):
    if copy_cnn:
        target_model.CNN_0 = src_model.CNN_0
        target_model.CNN_0_bn = src_model.CNN_0_bn

        target_model.CNN_1 = src_model.CNN_1
        target_model.CNN_1_bn = src_model.CNN_1_bn

    if copy_lstm:
        target_model.L0_enc = src_model.L0_enc
        target_model.L1_enc = src_model.L1_enc
        target_model.L2_enc = src_model.L2_enc

        target_model.L0_rev_enc = src_model.L0_rev_enc
        target_model.L1_rev_enc = src_model.L1_rev_enc
        target_model.L2_rev_enc = src_model.L2_rev_enc

    return target_model

def copy_attention_params(src_model, target_model):
    target_model.attn_Wa = src_model.attn_Wa
    target_model.context = src_model.context
    return target_model

def copy_decoder_params(src_model, target_model):
    target_model.L0_dec = src_model.L0_dec
    target_model.L1_dec = src_model.L1_dec
    target_model.L2_dec = src_model.L2_dec
    target_model.embed_dec = src_model.embed_dec
    target_model.out = src_model.out
    return target_model

new_model = copy_encoder_params(src_model=nn_1.model, target_model=nn_2.model,
                                copy_cnn=True, copy_lstm=True)

print(xp.all(nn_1.model.CNN_0.W.data == nn_2.model.CNN_0.W.data))
print(xp.all(nn_1.model.CNN_1.W.data == nn_2.model.CNN_1.W.data))
print(xp.all(nn_1.model.L0_enc.lateral.W.data == nn_2.model.L0_enc.lateral.W.data))
# print(xp.all(nn_1.model.embed_dec.W.data == nn_2.model.embed_dec.W.data))
print(nn_1.model.embed_dec.W.shape, nn_2.model.embed_dec.W.shape)

print("Saving model")
serializers.save_npz(os.path.join(model2_path, "seq2seq_0.model"), new_model)
print("Finished saving model ...")
