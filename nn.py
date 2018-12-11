# coding: utf-8

'''
Train AST model

-m : specifies the model directory which has the model_cfg.json
     and the train_cfg.json files
-e : specifies the number of epochs to train for

Program checks for existing model files, and loads the last saved
model if found

Author: Sameer Bansal
'''

from seq2seq import SpeechEncoderDecoder
from config import Config
from dataloader import FisherDataLoader

import cupy
from chainer import cuda, Function, utils, Variable
import numpy as np
import os
import chainer.functions as F
from chainer import optimizers, serializers

import random

import pickle

from chainer.optimizer import WeightDecay, GradientNoise, GradientClipping

program_descrp = """run nmt experiments"""

_ADAM = 0
_SGD = 1


class NN:
    def __init__(self, cfg_path):
        self.cfg = Config(cfg_path)

        # Define model data related paths
        self.model_dir = self.cfg.model["model_dir"]

        # Store gpuid locally
        self.gpuid = self.cfg.train["gpuid"]

        # Load data dictionaries
        self.data_loader = FisherDataLoader(self.cfg.train["data"],
                                            self.model_dir,
                                            self.gpuid)

        """
        Seq2Seq model
        Load model if it already exists in the directory.
        Otherwise, create a new model object
        """
        self.get_model()

        # Initialize optimizer
        self.init_optimizer(self.cfg.train["optimizer"])

        # Define log files path
        self.train_log = os.path.join(self.model_dir, "train.log")
        self.dev_log = os.path.join(self.model_dir, "dev.log")

    def init_optimizer(self, opt_cfg):
        print("Setting up optimizer")
        if opt_cfg['type'] == _ADAM:
            print("using ADAM")
            self.optimizer = optimizers.Adam(alpha=opt_cfg['lr'],
                                             beta1=0.9,
                                             beta2=0.999,
                                             eps=1e-08,
                                             amsgrad=True)
        else:
            print("using SGD")
            self.optimizer = optimizers.SGD(lr=opt_cfg['lr'])

        # Attach optimizer
        self.optimizer.setup(self.model)

        # Add Weight decay
        if opt_cfg['l2'] > 0:
            print("Adding WeightDecay: {0:f}".format(opt_cfg['l2']))
            self.optimizer.add_hook(WeightDecay(opt_cfg['l2']))

        # Gradient clipping
        print("Clipping gradients at: {0:d}".format(opt_cfg['grad_clip']))
        self.optimizer.add_hook(GradientClipping(threshold=
                                            opt_cfg['grad_clip']))

        # Gradient noise
        if opt_cfg['grad_noise_eta'] > 0:
            print("Adding gradient noise: {0:f}".format(opt_cfg['grad_noise_eta']))
            self.optimizer.add_hook(chainer.optimizer.GradientNoise(eta=opt_cfg['grad_noise_eta']))

        # Freeze weights
        for l in opt_cfg['freeze']:
            if l in self.model.__dict__:
                print("freezing: {0:s}".format(l))
                self.model[l].disable_update()
            else:
                print("layer {0:s} not in model".format(l))
        # end for


    def get_model(self):
        # Define training related paths
        self.model_fname = os.path.join(self.model_dir, "seq2seq.model")
        
        """
        Create seq2seq model object
        """
        self.model = SpeechEncoderDecoder(self.gpuid, 
                                          self.cfg.model)

        # Move model to gpu id
        self.model.to_gpu(self.gpuid)
        # ---------------------------------------------------------------------
        # check last saved model
        # ---------------------------------------------------------------------
        max_epoch = 0

        print("Checking for model in: {0:s}".format(self.model_dir))

        model_fil = self.model_fname
        model_files = [f for f in os.listdir(os.path.dirname(model_fil))
                       if os.path.basename(model_fil).replace('.model','') in f]
        if len(model_files) > 0:
            print("-"*80)
            max_model_fil = max(model_files, key=lambda s: int(s.split('_')[-1].split('.')[0]))
            max_model_fil = os.path.join(os.path.dirname(model_fil),
                                         max_model_fil)
            print('model found = \n{0:s}'.format(max_model_fil))
            serializers.load_npz(max_model_fil, self.model)
            print("finished loading ..")
            max_epoch = int(max_model_fil.split('_')[-1].split('.')[0])
        else:
            print("-"*80)
            print('model not found')


    def train_epoch(self):
        total_loss = 0
        for batch in self.data_loader.get_batch(32, train=True, labels=True):print(b["X"].shape, b["y"].shape);


    


cfg_path = "./experiments/es_en_20h"

ha = NN(cfg_path)


# metrics = Eval(cfg.eval_path, cfg.n_evals)
# print(metrics.refs[:5])


xp = cuda.cupy if ha.cfg.train["gpuid"] >= 0 else np


test_file = "../speech2text/mfcc_13dim/fisher_dev/20051016_180547_265_fsp-B-80.npy"

if ha.cfg.train["gpuid"] >= 0:
    cuda.get_device(ha.cfg.train["gpuid"]).use()
haha = F.expand_dims(xp.load(test_file), 0)

# model = SpeechEncoderDecoder(cfg)
# model.to_gpu(cfg.train["gpuid"])


# def main():
#     parser = argparse.ArgumentParser(description=program_descrp)
#     parser.add_argument('-m','--cfg_path', help='path for model config',
#                         required=True)
#     parser.add_argument('-e','--epochs', help='num epochs',
#                         required=True)

#     args = vars(parser.parse_args())

#     cfg_path = args['cfg_path']

#     epochs = int(args['epochs'])

#     print("number of epochs={0:d}".format(epochs))

    

# # end main
# # -----------------------------------------------------------------------------
# # -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     main()
# # -----------------------------------------------------------------------------
