#coding: utf-8

"""
Encoder-Decoder with Attention.
This file creates a model based on hyperparameters defined in 
a json file: model.cfg

Author: Sameer Bansal
"""

from common import *

class SpeechEncoderDecoder(chainer.Chain):
    def __init__(self, m_cfg, gpuid):
        self.m_cfg = m_cfg
        self.gpuid = gpuid

        self.init_params()

        if self.gpuid >= 0:
            cuda.get_device(self.gpuid).use()

        super(SpeechEncoderDecoder, self).__init__()
        self.init_model()

    def init_params(self):
        #----------------------------------------------------------------------
        # determine rnn type
        #----------------------------------------------------------------------
        self.RNN = L.LSTM
        #----------------------------------------------------------------------
        # get vocab size
        #----------------------------------------------------------------------
        vocab_dict = pickle.load(open(self.m_cfg["vocab_path"], "rb"))
        if self.m_cfg['enc_key'] != 'sp':
            self.v_size_es = len(vocab_dict[self.m_cfg['enc_key']]['w2i'])
        else:
            self.v_size_es = 0
        self.v_size_en = len(vocab_dict[self.m_cfg['dec_key']]['w2i'])
        #----------------------------------------------------------------------

    def add_rnn_layers(self, layer_names, in_units, out_units):
        w = chainer.initializers.HeNormal()
        for i, rnn_name in enumerate(layer_names):
            #------------------------------------------------------------------
            # for first layer, use in_units
            #------------------------------------------------------------------
            curr_in = in_units if i == 0 else out_units
            #------------------------------------------------------------------
            # add rnn layer
            #------------------------------------------------------------------
            self.add_link(rnn_name, self.RNN(curr_in, out_units))
            #------------------------------------------------------------------
            # add layer normalization
            #------------------------------------------------------------------
            if self.m_cfg['ln']:
                self.add_link("{0:s}_ln".format(rnn_name), L.LayerNormalization(out_units))
            #------------------------------------------------------------------

    def init_rnn_model(self, in_dim):
        h_units = self.m_cfg['hidden_units']
        #----------------------------------------------------------------------
        # add encoder layers
        #----------------------------------------------------------------------
        self.rnn_enc = ["L{0:d}_enc".format(i)
                         for i in range(self.m_cfg['enc_layers'])]
        self.add_rnn_layers(self.rnn_enc, in_dim, h_units)

        if self.m_cfg['bi_rnn']:
            #------------------------------------------------------------------
            # if bi rnn, add rev rnn layer
            #------------------------------------------------------------------
            self.rnn_rev_enc = ["L{0:d}_rev_enc".format(i) for i in range(self.m_cfg['enc_layers'])]
            self.add_rnn_layers(self.rnn_rev_enc, in_dim, h_units)

        #------------------------------------------------------------------
        # add attention layers
        #------------------------------------------------------------------
        a_units = self.m_cfg['attn_units']
        if self.m_cfg['bi_rnn']:
            self.add_link("attn_Wa", L.Linear(2*h_units, 2*h_units))
            #--------------------------------------------------------------
            # context layer = 2*h_units from enc + 2*h_units from dec
            #--------------------------------------------------------------
            self.add_link("context", L.Linear(4*h_units, a_units))
        else:
            self.add_link("attn_Wa", L.Linear(h_units, h_units))
            #--------------------------------------------------------------
            # context layer = 1*h_units from enc + 1*h_units from dec
            #--------------------------------------------------------------
            self.add_link("context", L.Linear(2*h_units, a_units))
        #------------------------------------------------------------------
        # add decoder layers
        #------------------------------------------------------------------
        e_units = self.m_cfg['embedding_units']
        # first layer appends previous ht, and therefore,
        # in_units = embed units + hidden units
        self.rnn_dec = ["L{0:d}_dec".format(i)
                        for i in range(self.m_cfg['dec_layers'])]
        #------------------------------------------------------------------
        # decoder rnn input = emb + prev. context vector
        #------------------------------------------------------------------
        if self.m_cfg['bi_rnn']:
            self.add_rnn_layers(self.rnn_dec, e_units+a_units, 2*h_units)
        else:
            self.add_rnn_layers(self.rnn_dec, e_units+a_units, h_units)
        #------------------------------------------------------------------

    def init_deep_cnn_model(self):
        CNN_IN_DIM = (self.m_cfg['sp_dim'] if self.m_cfg['enc_key'] == 'sp'
                             else self.m_cfg['embedding_units'])
        # ---------------------------------------------------------------------
        # initialize list of cnn layers
        # ---------------------------------------------------------------------
        self.cnns = []
        if len(self.m_cfg['cnn_layers']) > 0:
            # -----------------------------------------------------------------
            # using He initializer
            # -----------------------------------------------------------------
            w = chainer.initializers.HeNormal()
            # add CNN layers
            cnn_out_dim = 0
            self.reduce_dim_len = 1
            reduce_dim = CNN_IN_DIM
            for i, l in enumerate(self.m_cfg['cnn_layers']):
                if "dilate" not in l:
                    l["dilate"] = 1
                lname = "CNN_{0:d}".format(i)
                cnn_out_dim += l["out_channels"]
                self.cnns.append(lname)
                print("nobias = ", self.m_cfg['bn'])
                self.add_link(lname, L.Convolution2D(**l,
                                                     initialW=w,
                                                     nobias=self.m_cfg['bn']))
                reduce_dim = math.ceil(reduce_dim / l["stride"][1])
                self.reduce_dim_len *= l["stride"][0]
                if self.m_cfg['bn']:
                    # ---------------------------------------------------------
                    # add batch normalization
                    # ---------------------------------------------------------
                    self.add_link('{0:s}_bn'.format(lname), L.BatchNormalization((l["out_channels"])))
                    # ---------------------------------------------------------
            self.cnn_out_dim = self.m_cfg['cnn_layers'][-1]["out_channels"]
            # -----------------------------------------------------------------
            # cnn output has reduced dimensions based on strides
            # -----------------------------------------------------------------
            self.cnn_out_dim *= reduce_dim
            # -----------------------------------------------------------------
        else:
            # -----------------------------------------------------------------
            # no cnns added
            # -----------------------------------------------------------------
            self.cnn_out_dim = CNN_IN_DIM
            # -----------------------------------------------------------------
    # end init_deep_cnn_model()

    def init_model(self):
        xp = cuda.cupy if self.gpuid >= 0 else np
        # ---------------------------------------------------------------------
        # add enc embedding layer if text model
        # ---------------------------------------------------------------------
        if self.m_cfg['enc_key'] != 'sp':
            self.add_link("embed_enc", L.EmbedID(self.v_size_es,
                                                self.m_cfg['embedding_units']))
        # ---------------------------------------------------------------------
        # add cnn layer
        # ---------------------------------------------------------------------
        self.init_deep_cnn_model()
        rnn_in_units = self.cnn_out_dim
        # ---------------------------------------------------------------------
        # add rnn layers
        # ---------------------------------------------------------------------
        print("cnn_out_dim = rnn_in_units = ", rnn_in_units)
        self.init_rnn_model(rnn_in_units)
        # ---------------------------------------------------------------------
        # add decoder
        # ---------------------------------------------------------------------
        # add dec embedding layer
        # -----------------------------------------------------------------
        print("-"*80)
        print("using randomly initialized embeddings")
        print("-"*80)
        initial_emb_W = None
        self.add_link("embed_dec",
                       L.EmbedID(self.v_size_en,
                       self.m_cfg['embedding_units'],
                       initialW=initial_emb_W))
        # -----------------------------------------------------------------
        # add output layers
        # -----------------------------------------------------------------
        self.add_link("out",
                       L.Linear(self.m_cfg['attn_units'],
                       self.v_size_en))
        # -----------------------------------------------------------------
        # create masking array for pad id
        # -----------------------------------------------------------------
        with cupy.cuda.Device(self.gpuid):
            self.mask_pad_id = xp.ones(self.v_size_en, dtype=xp.float32)
        # make the class weight for pad id equal to 0
        # this way loss will not be computed for this predicted loss
        self.mask_pad_id[0] = 0        


    def reset_state(self):
        # ---------------------------------------------------------------------
        # reset the state of LSTM layers
        # ---------------------------------------------------------------------
        if self.m_cfg['bi_rnn']:
            for rnn_name in self.rnn_enc + self.rnn_rev_enc:
                self[rnn_name].reset_state()
        else:
            for rnn_name in self.rnn_enc:
                self[rnn_name].reset_state()

        for rnn_name in self.rnn_dec:
            self[rnn_name].reset_state()

        self.loss = 0

    def set_decoder_state(self):
        # ---------------------------------------------------------------------
        # set the hidden and cell state (LSTM) of the first RNN in the decoder
        # ---------------------------------------------------------------------
        if self.m_cfg['bi_rnn']:
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

    def feed_rnn(self, rnn_in, rnn_layers, highway_layers=None):
        hs = rnn_in
        for rnn_layer in rnn_layers:
            # -----------------------------------------------------------------
            # apply rnn
            # -----------------------------------------------------------------
            if self.m_cfg['rnn_dropout'] > 0:
                hs = F.dropout(self[rnn_layer](hs),
                               ratio=self.m_cfg['rnn_dropout'])
            else:
                hs = self[rnn_layer](hs)
            # -----------------------------------------------------------------
            # layer normalization
            # -----------------------------------------------------------------
            if self.m_cfg['ln']:
                ln_name = "{0:s}_ln".format(rnn_layer)
                hs = self[ln_name](hs)
            # -----------------------------------------------------------------
            # RELU activation
            # -----------------------------------------------------------------
            if 'rnn_relu' in self.m_cfg and self.m_cfg['rnn_relu'] == True:
                hs = F.relu(hs)
            # -----------------------------------------------------------------
        return hs

    def encode(self, data_in, rnn_layers):
        h = self.feed_rnn(data_in, rnn_layers)
        return h

    def decode(self, word, ht, get_alphas=False):
        # ---------------------------------------------------------------------
        # get embedding
        # ---------------------------------------------------------------------
        if 'embed_dropout' in self.m_cfg:
            embed_id = F.dropout(self.embed_dec(word),
                                 ratio=self.m_cfg['rnn_dropout'])
        else:
            embed_id = self.embed_dec(word)
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
        if self.m_cfg['out_dropout'] > 0:
            predicted_out = F.dropout(self.out(ht),
                                      ratio=self.m_cfg['out_dropout'])
        else:
            predicted_out = self.out(ht)
        # ---------------------------------------------------------------------
        if get_alphas:
            return predicted_out, ht, alphas
        else:
            return predicted_out, ht

    def decode_batch(self, decoder_batch, teacher_ratio):
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


    def forward_deep_cnn(self, h):
        # ---------------------------------------------------------------------
        # check and prepare for 2d convolutions
        # ---------------------------------------------------------------------
        h = F.expand_dims(h, 2)
        h = F.swapaxes(h,1,2)
        # ---------------------------------------------------------------------
        for i, cnn_layer in enumerate(self.cnns):
            # -----------------------------------------------------------------
            # apply cnn
            # -----------------------------------------------------------------
            h = self[cnn_layer](h)
            if "cnn_pool" in self.m_cfg:
                time_pool = self.m_cfg['cnn_pool'][i][0]
                if time_pool == -1:
                    time_pool = h.shape[-2]

                freq_pool = self.m_cfg['cnn_pool'][i][1]
                if freq_pool == -1:
                    freq_pool = h.shape[-1]

                # print(time_pool, freq_pool)
                # print("before", h.shape)

                h = F.max_pooling_nd(h, (time_pool, freq_pool))
                # print("after", h.shape)
            # -----------------------------------------------------------------
            # batch normalization before non-linearity
            # -----------------------------------------------------------------
            if self.m_cfg['bn']:
                bn_lname = '{0:s}_bn'.format(cnn_layer)
                h = self[bn_lname](h)
            # -----------------------------------------------------------------
            if 'leaky_relu' in self.m_cfg and self.m_cfg['leaky_relu'] == True:
                h = F.leaky_relu(h)
            else:
                h = F.relu(h)
            # -----------------------------------------------------------------

        # ---------------------------------------------------------------------
        # prepare return
        # if RNN
        #   batch size * num time frames after pooling * cnn out dim
        # else
        #
        # ---------------------------------------------------------------------
        h = F.swapaxes(h,1,2)
        h = F.reshape(h, h.shape[:2] + tuple([-1]))
        h = F.rollaxis(h,1)
        # ---------------------------------------------------------------------
        return h


    def forward_rnn(self, X):
        # ---------------------------------------------------------------------
        # reset rnn state
        # ---------------------------------------------------------------------
        self.reset_state()
        # ---------------------------------------------------------------------
        in_size, batch_size, in_dim = X.shape
        for i in range(in_size):
            if i > 0:
                h_fwd = F.concat((h_fwd,
                                  F.expand_dims(self.encode(X[i],
                                    self.rnn_enc), 0)),
                                  axis=0)
                if self.m_cfg['bi_rnn']:
                    h_rev = F.concat((h_rev,
                                      F.expand_dims(self.encode(X[-i],
                                        self.rnn_rev_enc), 0)),
                                      axis=0)
            else:
                h_fwd = F.expand_dims(self.encode(X[i], self.rnn_enc), 0)
                if self.m_cfg['bi_rnn']:
                    h_rev = F.expand_dims(self.encode(X[-i], self.rnn_rev_enc), 0)
        # ---------------------------------------------------------------------
        if self.m_cfg['bi_rnn']:
            h_rev = F.flipud(h_rev)
            self.enc_states = F.concat((h_fwd, h_rev), axis=2)
        else:
            self.enc_states = h_fwd
        # ---------------------------------------------------------------------
        self.enc_states = F.swapaxes(self.enc_states, 0, 1)
        # ---------------------------------------------------------------------

    def forward_enc(self, X, l=None):
        if self.m_cfg['enc_key'] != 'sp':
            # -----------------------------------------------------------------
            # get encoder embedding for text input
            # -----------------------------------------------------------------
            h = self.embed_enc(X)
            # -----------------------------------------------------------------
        else:
            h = X
        # ---------------------------------------------------------------------
        # call cnn logic
        # ---------------------------------------------------------------------
        h = self.forward_deep_cnn(h)
        # # -----------------------------------------------------------------------
        # call rnn logic
        # ---------------------------------------------------------------------
        self.forward_rnn(h)
        # ---------------------------------------------------------------------


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
        if self.m_cfg['bi_rnn']:
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

