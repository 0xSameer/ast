#coding: utf-8

"""
Encoder-Decoder with Attention.
This file creates a model based on hyperparameters defined in
a json file: model.cfg

Author: Sameer Bansal
"""

import chainer
import chainer.links as L
import chainer.functions as F
import cupy
from chainer import cuda, Function, utils, Variable
import math
import numpy as np
import random

from dataloader import SYMBOLS

class SpeechEncoderDecoder(chainer.Chain):
    def __init__(self, gpuid, cfg):
        self.gpuid = gpuid
        if self.gpuid >= 0:
            cuda.get_device(self.gpuid).use()

        super(SpeechEncoderDecoder, self).__init__()

        self.cfg = cfg

        self.init_cnn(cfg["cnn_config"])
        self.init_enc_dec_attn(cfg["rnn_config"])

    def init_cnn(self, CNN_CONFIG):
        """
        Add each CNN layer defined in cofig
        Using He initializer for weights
        Each CNN layer is stored in a list
        """
        self.cnns = []

        self.cnn_bn = CNN_CONFIG['bn']

        w = chainer.initializers.HeNormal()
        # Add CNN layers
        for i, l in enumerate(CNN_CONFIG["cnn_layers"]):
            if "dilate" not in l:
                l["dilate"] = 1
            lname = "CNN_{0:d}".format(i)
            self.cnns.append(lname)
            self.add_link(lname, L.Convolution2D(**l,
                                                 initialW=w,
                                                 nobias=self.cnn_bn))
            if self.cnn_bn:
                #Add batch normalization
                self.add_link('{0:s}_bn'.format(lname), L.BatchNormalization((l["out_channels"])))
            # end if bn
        # end for each layer
    # end init_deep_cnn_model

    def init_enc_dec_attn(self, RNN_CONFIG):
        xp = cuda.cupy if self.gpuid >= 0 else np
        """
        Add encoder RNN layers
        if bi-rnn, then hidden units in each direction = hidden units / 2
        """
        self.rnn_enc = ["L{0:d}_enc".format(i)
                         for i in range(RNN_CONFIG['enc_layers'])]

        self.bi_rnn = RNN_CONFIG['bi_rnn']
        if RNN_CONFIG['bi_rnn']:
            enc_lstm_units = RNN_CONFIG['hidden_units'] // 2
            self.rnn_rev_enc = ["L{0:d}_rev_enc".format(i) for i in range(RNN_CONFIG['enc_layers'])]
        else:
            enc_lstm_units = RNN_CONFIG['hidden_units']
            self.rnn_rev_enc = []


        # Add each layer
        self.rnn_ln = RNN_CONFIG['ln']
        for i, rnn_name in enumerate(self.rnn_enc + self.rnn_rev_enc):
            self.add_link(rnn_name, L.LSTM(None, enc_lstm_units))
            # Add layer normalization
            if RNN_CONFIG['ln']:
                self.add_link("{0:s}_ln".format(rnn_name),
                              L.LayerNormalization(enc_lstm_units))
        # end for enc rnn
        # Add linear projection with Batch Norm
        self.rnn_linear_proj = False
        if 'linear_proj' in RNN_CONFIG and RNN_CONFIG['linear_proj']:
            self.rnn_linear_proj = True
            proj_units = RNN_CONFIG['hidden_units']
            self.add_link(f"enc_proj", L.Linear(proj_units, proj_units))
            self.add_link(f"enc_proj_bn", L.BatchNormalization((proj_units)))
            # proj_units = RNN_CONFIG['hidden_units']
            # for i in range(RNN_CONFIG['enc_layers']):
            #     self.add_link(f"enc_proj{i}", L.Linear(proj_units, proj_units))
            #     self.add_link(f"enc_proj{i}_bn",
            #                   L.BatchNormalization((proj_units)))
        print(f"RNN linear projection layer: {self.rnn_linear_proj}")
        """
        Add attention layers
        """
        a_units = RNN_CONFIG['attn_units']
        self.add_link("attn_Wa", L.Linear(RNN_CONFIG['hidden_units'],
                                          RNN_CONFIG['hidden_units']))
        self.n_attn = 1 if 'n_attn' not in RNN_CONFIG else RNN_CONFIG['n_attn']
        if self.n_attn > 1:
            print(f"# Attention layers: {self.n_attn}")

        feed_attn = True if 'feed_attn' not in self.cfg["rnn_config"] else \
                            self.cfg["rnn_config"]['feed_attn']
        print(f"Feed attention to next decode step: {feed_attn}")
        for i in range(1, self.n_attn):
            self.add_link(f"attn_Wa{i}", L.Linear(RNN_CONFIG['hidden_units'],
                                                  RNN_CONFIG['hidden_units']))
        # Context layer = 1*h_units from enc + 1*h_units from dec
        # self.add_link("context", L.Linear(2*RNN_CONFIG['hidden_units'],
        #                                   a_units))
        self.add_link("context", L.Linear(
                                    (self.n_attn+1)*RNN_CONFIG['hidden_units'],
                                     a_units))
        """
        Add decoder layers
        Embedding layer
        """
        e_units = RNN_CONFIG['embedding_units']
        self.add_link("embed_dec",
                       L.EmbedID(RNN_CONFIG["dec_vocab_size"],
                       e_units))

        # Add decoder rnns
        self.rnn_dec = ["L{0:d}_dec".format(i)
                        for i in range(RNN_CONFIG['dec_layers'])]

        # decoder rnn input = emb + prev. context vector
        dec_lstm_units = RNN_CONFIG['hidden_units']
        for i, rnn_name in enumerate(self.rnn_dec):
            self.add_link(rnn_name, L.LSTM(None, dec_lstm_units))
            # Add layer normalization
            if RNN_CONFIG['ln']:
                self.add_link("{0:s}_ln".format(rnn_name),
                          L.LayerNormalization(RNN_CONFIG['hidden_units']))
        # end for

        """
        Add output layers
        """
        self.add_link("out",
                       L.Linear(a_units, RNN_CONFIG["dec_vocab_size"]))
        # create masking array for pad id
        with cupy.cuda.Device(self.gpuid):
            self.mask_pad_id = xp.ones(RNN_CONFIG["dec_vocab_size"],
                                       dtype=xp.float32)
        # set PAD ID to 0, so as to not compute any loss for it
        self.mask_pad_id[0] = 0

    def forward_cnn(self, h):
        # Check and prepare for 2d convolutions
        h = F.expand_dims(h, 2)
        h = F.swapaxes(h,1,2)
        # Apply each CNN layer
        for i, cnn_layer in enumerate(self.cnns):
            # cnn pass
            h = self[cnn_layer](h)
            # Apply batch normalization
            if self.cnn_bn:
                bn_lname = '{0:s}_bn'.format(cnn_layer)
                h = self[bn_lname](h)
            # Apply non-linearity
            h = F.relu(h)

        """
        Prepare return
        batch size * num time frames after pooling * cnn out dim
        """
        h = F.swapaxes(h,1,2)
        h = F.reshape(h, h.shape[:2] + tuple([-1]))
        h = F.rollaxis(h,1)
        return h

    def reset_rnn_state(self):
        """
        reset the state of LSTM layers
        """
        for rnn_name in self.rnn_enc + self.rnn_rev_enc + self.rnn_dec:
            self[rnn_name].reset_state()

        # reset loss
        self.loss = 0

    def feed_rnn(self, rnn_in, rnn_layers):
        hs = rnn_in
        for rnn_layer in rnn_layers:
            """
            Apply rnn
            """
            hs = F.dropout(self[rnn_layer](hs), ratio=self.cfg["dropout"]["rnn"])
            # layer normalization
            if self.rnn_ln:
                ln_name = "{0:s}_ln".format(rnn_layer)
                hs = self[ln_name](hs)
        return hs

    def forward_rnn_encode(self, X):
        # Reset rnn state
        self.reset_rnn_state()
        # Get input shape
        in_size, batch_size, in_dim = X.shape
        # For each time step
        for i in range(in_size):
            # Store all hidden states
            if i > 0:
                h_fwd = F.concat((h_fwd,
                                  F.expand_dims(self.feed_rnn(X[i],
                                    self.rnn_enc), 0)), axis=0)
                if self.bi_rnn:
                    h_rev = F.concat((h_rev,
                                      F.expand_dims(self.feed_rnn(X[-i],
                                      self.rnn_rev_enc), 0)), axis=0)
            else:
                h_fwd = F.expand_dims(self.feed_rnn(X[i], self.rnn_enc), 0)
                if self.bi_rnn:
                    h_rev = F.expand_dims(self.feed_rnn(X[-i],
                                          self.rnn_rev_enc), 0)
        """
        Concatenate fwd and rev RNN hidden states
        Flip reverse RNN hidden state order
        """
        if self.bi_rnn:
            h_rev = F.flipud(h_rev)
            rnn_states = F.concat((h_fwd, h_rev), axis=2)
        else:
            rnn_states = h_fwd

        """
        Check if linear projection layer required
        """
        if self.rnn_linear_proj == False:
            self.enc_states = rnn_states
        else:
            for i in range(0, in_size):
                currH = F.relu(self.enc_proj_bn(self.enc_proj(rnn_states[i])))
                if i > 0:
                    self.enc_states = F.concat((self.enc_states,
                                      F.expand_dims(currH, 0)), axis=0)
                else:
                    self.enc_states = F.expand_dims(currH, 0)
            # end for all hidden states
        # end


        # Make the batch size as the first dimension
        self.enc_states = F.swapaxes(self.enc_states, 0, 1)

    def encode(self, X, add_noise=0):
        # ---------------------------------------------------------------------
        # check whether to add noise to speech input
        # ---------------------------------------------------------------------
        if add_noise > 0 and chainer.config.train:
            # due to CUDA issues with random number generator
            # creating a numpy array and moving to GPU
            noise = Variable(np.random.normal(1.0,
                                              add_noise,
                                              size=X.shape).astype(np.float32))
            if self.gpuid >= 0:
                noise.to_gpu(self.gpuid)
            X = X * noise

        # call cnn logic
        h = self.forward_cnn(X)
        # print("cnn out", h.shape)
        # call rnn logic
        self.forward_rnn_encode(h)
        # print("rnn out", self.enc_states[0].shape)


    def init_decoder_state(self):
        """
        Set the hidden and cell state (LSTM) of the first RNN in the decoder
        """
        if self.bi_rnn:
            for enc, rev_enc, dec in zip(self.rnn_enc,
                                         self.rnn_rev_enc,
                                         self.rnn_dec):
                h_state = F.concat((self[enc].h, self[rev_enc].h))

                c_state = F.concat((self[enc].c, self[rev_enc].c))
                self[dec].set_state(c_state, h_state)

        else:
            for enc, dec in zip(self.rnn_enc, self.rnn_dec):
                self[dec].set_state(self[enc].c, self[enc].h)
        # ---------------------------------------------------------------------

    def compute_context_vector(self, dec_h, attn_Wa):
        batch_size, n_units = dec_h.shape
        # attention weights for the hidden states of each word in the input list
        # ---------------------------------------------------------------------
        # compute weights
        ht = attn_Wa(dec_h)
        weights = F.batch_matmul(self.enc_states, ht)
        # ---------------------------------------------------------------------
        # '''
        # this line is valid when no max pooling or sequence length manipulation is performed
        # weights = F.where(self.mask, weights, self.minf)
            # '''
        # ---------------------------------------------------------------------
        # softmax to compute alphas
        # ---------------------------------------------------------------------
        alphas = F.softmax(weights)
        # ---------------------------------------------------------------------
        # compute context vector
        # ---------------------------------------------------------------------
        cv = F.squeeze(F.batch_matmul(F.swapaxes(self.enc_states, 2, 1), alphas), axis=2)
        # ---------------------------------------------------------------------
        return cv, alphas
        # ---------------------------------------------------------------------


    def decode_step(self, word, ht):
        # ---------------------------------------------------------------------
        # get embedding
        # ---------------------------------------------------------------------
        embed_id = F.dropout(self.embed_dec(word), ratio=self.cfg["dropout"]['embed'])
        # ---------------------------------------------------------------------
        # apply rnn - input feeding, use previous ht
        # ---------------------------------------------------------------------
        feed_attn = True if 'feed_attn' not in self.cfg["rnn_config"] else \
                    self.cfg["rnn_config"]['feed_attn']
        if feed_attn:
            rnn_in = F.concat((embed_id, ht), axis=1)
        else:
            rnn_in = embed_id
        h = self.feed_rnn(rnn_in, self.rnn_dec)
        # ---------------------------------------------------------------------
        # compute context vector
        # ---------------------------------------------------------------------
        cv, alphas = self.compute_context_vector(h, self.attn_Wa)

        for i in range(1, self.n_attn):
            new_cv, _ = self.compute_context_vector(h, self[f'attn_Wa{i}'])
            cv = F.concat((cv, new_cv), axis=1)

        # concatenate with decoder hidden state
        cv_hdec = F.concat((cv, h), axis=1)
        # ---------------------------------------------------------------------
        # compute attentional hidden state
        # ---------------------------------------------------------------------
        ht = F.tanh(self.context(cv_hdec))
        # ---------------------------------------------------------------------
        # make prediction
        # ---------------------------------------------------------------------
        predicted_out = F.dropout(self.out(ht), ratio=self.cfg["dropout"]['out'])
        # ---------------------------------------------------------------------
        return predicted_out, ht, alphas


    def forward_loss(self, X, y, teach_ratio, random_out=0, add_noise=0):
        xp = cuda.cupy if self.gpuid >= 0 else np
        batch_size = X.shape[0]

        # encode input
        self.encode(X, add_noise=add_noise)

        # initialize decoder LSTM to final encoder state
        self.init_decoder_state()

        # swap axes of the decoder batch
        y = F.swapaxes(y, 0, 1)
        """
        Initialize loss
        Compute loss at each predicted step for target text
        """
        loss = 0
        """
        Initialize attention to zeros
        """
        a_units = self.cfg["rnn_config"]['attn_units']
        ht = Variable(xp.zeros((batch_size, a_units), dtype=xp.float32))

        # for all sequences in the batch, feed the characters one by one
        for i, (curr_word, next_word) in enumerate(zip(y, y[1:])):
            # print("decode", i, curr_word, next_word)

            """
            Check whether to use predicted token or true token from
            previous time step
            Always use true token for GO and EOS
            """
            if (i > 0) and (i < (len(y)-2)):
                use_true = random.random() < teach_ratio
                if use_true:
                    decoder_input = curr_word
            else:
                decoder_input = curr_word

            """
            Decode current token -- we get the softmax output
            """
            predicted_out, ht, _ = self.decode_step(decoder_input, ht)
            # predicted_out, _ = self.decode_step(decoder_input)

            """
            Set the decoder_input for the next step to the current
            most likely prediction
            """
            decoder_input = F.argmax(predicted_out, axis=1)

            """
            Compute loss between predicted and true word

            If random out enabled, replace target word with another
            word type from the vocabulary
            """
            target_word = xp.copy(next_word.data)
            if random_out > 0:
                # sample and replace each element in the batch
                # if not special symbol < 4
                n_special = len(SYMBOLS.START_VOCAB)
                for i in range(len(target_word)):
                    if ((int(target_word[i]) >= n_special) and
                        (random.random() > random_out)):
                        target_word[i] = xp.random.randint(n_special,
                                    self.cfg["rnn_config"]["dec_vocab_size"]+1)
            # end replace target word

            curr_loss = F.softmax_cross_entropy(predicted_out, target_word,
                                               class_weight=self.mask_pad_id)
            loss += curr_loss
            # -----------------------------------------------------------------

        return loss

    def predict(self, X, start_token, end_token, stop_limit):

        xp = cuda.cupy if self.gpuid >= 0 else np
        batch_size = X.shape[0]

        # encode input
        self.encode(X)

        # initialize decoder LSTM to final encoder state
        self.init_decoder_state()

        # Flags to keep track whether EOS has been predicted for all
        with cupy.cuda.Device(self.gpuid):
            check_if_all_eos = xp.full((batch_size,), False, dtype=xp.bool_)
        """
        Initialize attention to zeros
        """
        a_units = self.cfg["rnn_config"]['attn_units']
        ht = Variable(xp.zeros((batch_size, a_units), dtype=xp.float32))

        npred = 0

        # Starting token
        curr_word = Variable(xp.full((batch_size,),
                             start_token, dtype=xp.int32))

        # loop till prediction limit or end_token predicted
        while npred < (stop_limit):
            # decode and predict
            pred_out, ht, _ = self.decode_step(curr_word, ht)
            # pred_out, _ = self.decode_step(curr_word)
            pred_word = F.argmax(pred_out, axis=1)
            # -----------------------------------------------------------------
            # save prediction at this time step
            # -----------------------------------------------------------------
            if npred == 0:
                pred_sents = xp.expand_dims(pred_word.data, 0)
            else:
                pred_sents = xp.vstack((pred_sents, pred_word.data))
            # -----------------------------------------------------------------

            curr_word = pred_word
            # check if EOS is predicted for all sentences
            # -----------------------------------------------------------------
            check_if_all_eos[pred_word.data == end_token] = True
            if xp.all(check_if_all_eos):
                break
            # -----------------------------------------------------------------
            # increment number of predictions made
            npred += 1
            # -----------------------------------------------------------------

        return pred_sents.T

    def get_encoder_states(self):
        rnn_states = {"c": [], "h": []}
        # ---------------------------------------------------------------------
        # get the hidden and cell state (LSTM) of the first RNN in the decoder
        # ---------------------------------------------------------------------
        if self.bi_rnn:
            for i, (enc, rev_enc) in enumerate(zip(self.rnn_enc,
                                         self.rnn_rev_enc)):
                h_state = F.concat((self[enc].h, self[rev_enc].h))
                rnn_states["h"].append(h_state)
                c_state = F.concat((self[enc].c, self[rev_enc].c))
                rnn_states["c"].append(c_state)
        else:
            for enc, dec in zip(self.rnn_enc, self.rnn_dec):
                rnn_states["h"].append(self[enc].h)
                rnn_states["c"].append(self[enc].c)
            # end for all layers
        # end if bi-rnn
        return rnn_states
        # ---------------------------------------------------------------------

    def get_decoder_states(self):
        rnn_states = {"c": [], "h": []}
        # ---------------------------------------------------------------------
        # get the hidden and cell state (LSTM) of the first RNN in the decoder
        # ---------------------------------------------------------------------
        for i, dec in enumerate(self.rnn_dec):
            rnn_states["h"].append(self[dec].h)
            rnn_states["c"].append(self[dec].c)
        # end for all layers
        return rnn_states
        # ---------------------------------------------------------------------

    def set_decoder_states(self, rnn_states):
        # ---------------------------------------------------------------------
        # set the hidden and cell state (LSTM) for the decoder
        # ---------------------------------------------------------------------
        for i, dec in enumerate(self.rnn_dec):
            self[dec].set_state(rnn_states["c"][i], rnn_states["h"][i])
        # end for all layers
        # ---------------------------------------------------------------------



