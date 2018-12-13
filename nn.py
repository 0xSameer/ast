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

import argparse
from seq2seq import SpeechEncoderDecoder
from config import Config
from dataloader import FisherDataLoader, SYMBOLS
from eval import Eval

import chainer
import cupy
from chainer import cuda, Function, utils, Variable
import numpy as np
import os
import chainer.functions as F
from chainer import optimizers, serializers
from chainer.optimizer import WeightDecay, GradientNoise, GradientClipping


from tqdm import tqdm
import random
import pickle



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
        print("learning rate: {0:f}".format(opt_cfg['lr']))
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
        self.max_epoch = 0

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
            self.max_epoch = int(max_model_fil.split('_')[-1].split('.')[0])
        else:
            print("-"*80)
            print('model not found')


    def train_epoch(self, set_key):
        total_loss = 0
        n_utts = self.data_loader.n_utts[set_key]
        batch_size = self.cfg.train["batch_size"]
        n_batches = 0

        with tqdm(total=n_utts, ncols=80) as pbar:
            for batch in self.data_loader.get_batch(batch_size, 
                                                    set_key, 
                                                    train=True, 
                                                    labels=True):
                # print(batch["X"].shape, batch["y"].shape)

                with chainer.using_config('train', True):
                    loss = self.model.forward_loss(X=batch['X'], 
                                                   y=batch['y'])
                    self.model.cleargrads()
                    loss.backward()
                    self.optimizer.update()
                # end weights update

                """
                Collect loss values for reporting
                Loss is normalized with the sequence length
                """
                loss_val = float(loss.data) / len(batch['y'])
                n_batches += 1
                total_loss += loss_val
                avg_loss = total_loss / n_batches
                pbar.set_description('loss={0:0.4f}'.format(avg_loss))
                pbar.update(len(batch["X"]))
                
            # end for each batch
        # end progress bar
        # print("Epoch complete")
        # print("Avg epoch loss = {0:.4f}".format(avg_loss))
        return avg_loss

    def predict(self, set_key):
        n_utts = self.data_loader.n_utts[set_key]
        batch_size = self.cfg.train["batch_size"]
        n_batches = 0
        preds = []
        stop_limit = self.cfg.train["data"]["max_pred"]

        with tqdm(total=n_utts, ncols=80) as pbar:
            for batch in self.data_loader.get_batch(batch_size, 
                                                    set_key, 
                                                    train=True, 
                                                    labels=True):

                # Training mode not enabled
                with chainer.using_config('train', False):
                    p = self.model.predict(batch['X'], 
                                           SYMBOLS.GO_ID,
                                           SYMBOLS.EOS_ID,
                                           stop_limit)
                    preds.extend(zip(batch["utts"], p.tolist()))

                pbar.update(len(batch["X"]))

                n_batches += 1

                # if n_batches > 2:
                #     break
                
            # end for each batch
        # end progress bar
        print("predictions complete", len(preds))
        return preds
