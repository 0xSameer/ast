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

class SpeechEncoderDecoder(chainer.Chain):
    def __init__(self, gpuid, cfg):
        self.gpuid = gpuid
        if self.gpuid >= 0:
            cuda.get_device(self.gpuid).use()

        super(SpeechEncoderDecoder, self).__init__()

        self.dropout = cfg["dropout"]

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
            # print("nobias = ", self.cnn_bn)
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
        """
        Add attention layers
        """
        a_units = RNN_CONFIG['attn_units']
        self.add_link("attn_Wa", L.Linear(RNN_CONFIG['hidden_units'], 
                                          RNN_CONFIG['hidden_units']))
        # Context layer = 1*h_units from enc + 1*h_units from dec
        self.add_link("context", L.Linear(2*RNN_CONFIG['hidden_units'], 
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
                print("hahahaha")
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
            hs = F.dropout(self[rnn_layer](hs), ratio=self.dropout["rnn"])
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
            self.enc_states = F.concat((h_fwd, h_rev), axis=2)
        else:
            self.enc_states = h_fwd
        
        # Make the batch size as the first dimension
        self.enc_states = F.swapaxes(self.enc_states, 0, 1)

    def encode(self, X):
        # call cnn logic
        h = self.forward_cnn(X)
        print(h.shape)
        # call rnn logic
        self.forward_rnn_encode(h)
        print(self.enc_states[0].shape)

    def set_decoder_state(self):
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

    def compute_context_vector(self, dec_h):
        batch_size, n_units = dec_h.shape
        # attention weights for the hidden states of each word in the input list
        # ---------------------------------------------------------------------
        # compute weights
        ht = self.attn_Wa(dec_h)
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
        embed_id = F.dropout(self.embed_dec(word), ratio=self.dropout['embed'])
        # ---------------------------------------------------------------------
        # apply rnn - input feeding, use previous ht
        # ---------------------------------------------------------------------
        rnn_in = F.concat((embed_id, ht), axis=1)
        h = self.feed_rnn(rnn_in, self.rnn_dec)
        # ---------------------------------------------------------------------
        # compute context vector
        # ---------------------------------------------------------------------
        cv, alphas = self.compute_context_vector(h)
        cv_hdec = F.concat((cv, h), axis=1)
        # ---------------------------------------------------------------------
        # compute attentional hidden state
        # ---------------------------------------------------------------------
        ht = F.tanh(self.context(cv_hdec))
        # ---------------------------------------------------------------------
        # make prediction
        # ---------------------------------------------------------------------
        predicted_out = F.dropout(self.out(ht), ratio=self.dropout['out'])
        # ---------------------------------------------------------------------
        return predicted_out, ht, alphas
        

    def train_decode(self, decoder_batch, teacher_ratio):
        xp = cuda.cupy if self.gpuid >= 0 else np
        batch_size = decoder_batch.shape[1]
        loss = 0
        # ---------------------------------------------------------------------
        # initialize hidden states as a zero vector
        # ---------------------------------------------------------------------
        a_units = self.m_cfg['attn_units']
        ht = Variable(xp.zeros((batch_size, a_units), dtype=xp.float32))
        # ---------------------------------------------------------------------
        decoder_input = decoder_batch[0]
        # for all sequences in the batch, feed the characters one by one
        for curr_word, next_word in zip(decoder_batch, decoder_batch[1:]):
            # -----------------------------------------------------------------
            # teacher forcing logic
            # -----------------------------------------------------------------
            use_label = True if random.random() < teacher_ratio else False
            if use_label:
                decoder_input = curr_word
            # -----------------------------------------------------------------
            # encode tokens
            # -----------------------------------------------------------------
            predicted_out, ht = self.decode(decoder_input, ht)
            decoder_input = F.argmax(predicted_out, axis=1)
            # -----------------------------------------------------------------
            # compute loss
            # -----------------------------------------------------------------
            if "random_out" in self.m_cfg and self.m_cfg["random_out"] == True:
                t_alt = xp.copy(next_word.data)
                # sample and replace each element in the batch
                for i in range(len(t_alt)):
                    # use_sample = True if random.random() > self.m_cfg["sample_out_prob"] else False
                    if int(t_alt[i]) >= 4 and random.random() > self.m_cfg["random_out_prob"]:
                        t_alt[i] = xp.random.randint(4, self.v_size_en+1)

                loss_arr = F.softmax_cross_entropy(predicted_out, t_alt,
                                               class_weight=self.mask_pad_id)
            else:
                loss_arr = F.softmax_cross_entropy(predicted_out, next_word,
                                               class_weight=self.mask_pad_id)
            loss += loss_arr
            # -----------------------------------------------------------------
        return loss

    def predict_batch(self, batch_size, pred_limit, y=None, display=False):
        xp = cuda.cupy if self.gpuid >= 0 else np
        # max number of predictions to make
        # if labels are provided, this variable is not used
        stop_limit = pred_limit
        # to track number of predictions made
        npred = 0
        # to store loss
        loss = 0
        # if labels are provided, use them for computing loss
        compute_loss = True if y is not None else False
        # ---------------------------------------------------------------------
        if compute_loss:
            stop_limit = len(y)-1
            # get starting word to initialize decoder
            curr_word = y[0]
        else:
            # intialize starting word to GO_ID symbol
            curr_word = Variable(xp.full((batch_size,), GO_ID, dtype=xp.int32))
        # ---------------------------------------------------------------------
        # flag to track if all sentences in batch have predicted EOS
        # ---------------------------------------------------------------------
        with cupy.cuda.Device(self.gpuid):
            check_if_all_eos = xp.full((batch_size,), False, dtype=xp.bool_)
        # ---------------------------------------------------------------------
        a_units = self.m_cfg['attn_units']
        ht = Variable(xp.zeros((batch_size, a_units), dtype=xp.float32))
        # ---------------------------------------------------------------------
        while npred < (stop_limit):
            # -----------------------------------------------------------------
            # decode and predict
            pred_out, ht = self.decode(curr_word, ht)
            pred_word = F.argmax(pred_out, axis=1)
            # -----------------------------------------------------------------
            # save prediction at this time step
            # -----------------------------------------------------------------
            if npred == 0:
                pred_sents = xp.expand_dims(pred_word.data, 0)
            else:
                pred_sents = xp.vstack((pred_sents, pred_word.data))
            # -----------------------------------------------------------------
            if compute_loss:
                # compute loss
                loss += F.softmax_cross_entropy(pred_out, y[npred+1],
                                                   class_weight=self.mask_pad_id)
            # -----------------------------------------------------------------
            curr_word = pred_word
            # check if EOS is predicted for all sentences
            # -----------------------------------------------------------------
            check_if_all_eos[pred_word.data == EOS_ID] = True
            # if xp.all(check_if_all_eos == EOS_ID):
            if xp.all(check_if_all_eos):
                break
            # -----------------------------------------------------------------
            # increment number of predictions made
            npred += 1
            # -----------------------------------------------------------------
        return pred_sents.T, loss

    def forward(self, X, add_noise=0, teacher_ratio=0, y=None):
        # get shape
        X.to_gpu(self.gpuid)
        batch_size = X.shape[0]
        # check whether to add noi, start=1se
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
        # ---------------------------------------------------------------------
        # encode input
        # ---------------------------------------------------------------------
        self.forward_enc(X)
        # -----------------------------------------------------------------
        # initialize decoder LSTM to final encoder state
        # -----------------------------------------------------------------
        self.set_decoder_state()
        # -----------------------------------------------------------------
        # swap axes of the decoder batch
        if y is not None:
            y = F.swapaxes(y, 0, 1)
        # -----------------------------------------------------------------
        # check if train or test
        # -----------------------------------------------------------------
        if chainer.config.train:
            # -------------------------------------------------------------
            # decode
            # -------------------------------------------------------------
            self.loss = self.decode_batch(y, teacher_ratio)
            # -------------------------------------------------------------
            # make return statements consistent
            return [], self.loss
        else:
            # -------------------------------------------------------------
            # predict
            # -------------------------------------------------------------
            # make return statements consistent
            return(self.predict_batch(batch_size=batch_size,
                                      pred_limit=self.m_cfg['max_en_pred'],
                                      y=y))
        # -----------------------------------------------------------------

    def add_lstm_weight_noise(self, rnn_layer, mu, sigma):
        xp = cuda.cupy if self.gpuid >= 0 else np
        # W_shape = self[rnn_layer].W.W.shape
        # b_shape = self[rnn_layer].W.b.shape
        rnn_params = ["upward", "lateral"]
        for p in rnn_params:
            # add noise to W
            s_w = xp.random.normal(mu,
                                   sigma,
                                   self[rnn_layer][p].W.shape,
                                   dtype=xp.float32)

            self[rnn_layer][p].W.data = self[rnn_layer][p].W.data + s_w

            if p == "upward":
                s_b = xp.random.normal(mu,
                                       sigma,
                                       self[rnn_layer][p].b.shape,
                                       dtype=xp.float32)
                self[rnn_layer][p].b.data = self[rnn_layer][p].b.data + s_b

    def add_weight_noise(self, mu, sigma):
        xp = cuda.cupy if self.gpuid >= 0 else np
        # add noise to rnn weights
        if self.bi_rnn:
            rnn_layers = self.rnn_enc + self.rnn_rev_enc + self.rnn_dec
        else:
            rnn_layers = self.rnn_enc + self.rnn_dec

        for rnn_layer in rnn_layers:
            self.add_lstm_weight_noise(rnn_layer, mu, sigma)

        # add noise to decoder embeddings
        self.embed_dec.W.data = (self.embed_dec.W.data +
                                   xp.random.normal(mu,
                                                    sigma,
                                                    self.embed_dec.W.shape,
                                                    dtype=xp.float32))

# In[ ]:

