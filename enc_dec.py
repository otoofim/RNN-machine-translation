# coding: utf-8
#---------------------------------------------------------------------
'''
Neural Machine Translation - Encoder Decoder model
    Chainer implementation of an encoder-decoder sequence to sequence
    model using bi-directional LSTM encoder
'''
#---------------------------------------------------------------------

# In[]:

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.functions.array import concat


# In[]:
# Import configuration file

from nmt_config import *

# In[]:

class EncoderDecoder(Chain):

    '''
    Constructor to initialize model
    Params:
        vsize_enc   - vocabulary size for source language (fed into encoder)
        vsize_dec   - vocabulary size for target language (fed into decoder)
        n_units     - size of the LSTMs
        attn        - specifies whether to use attention
    '''
    def __init__(self, vsize_enc, vsize_dec,
                 nlayers_enc, nlayers_dec,
                 n_units, gpuid, attn=False):
        super(EncoderDecoder, self).__init__()
        #--------------------------------------------------------------------
        # add encoder layers
        #--------------------------------------------------------------------
        # add embedding layer
        self.add_link("embed_enc", L.EmbedID(vsize_enc, n_units))

        '''
        ___QUESTION-1-DESCRIBE-A-START___

        - Explain the following lines of code
        - Think about what add_link() does and how can we access Links added in Chainer.
        - Why are there two loops or adding links?


	In this part of the code, the layers of RNN is built. Each layer is a fully-connected LSTM layer and will be added to our Chainer class.
 	By calling for example self[lstm_enc[0]] which is a dictionary we can access the first layer in encoder. 
	Since we use bidirectional encoder, there are two "for loops". The first one is for the forward RNN (reads input sequence from left to right) and the second one is for backward RNN 		(reads the sequence in the reverse order).
        '''
        self.lstm_enc = ["L{0:d}_enc".format(i) for i in range(nlayers_enc)]
        for lstm_name in self.lstm_enc:
            self.add_link(lstm_name, L.LSTM(n_units, n_units))

        self.lstm_rev_enc = ["L{0:d}_rev_enc".format(i) for i in range(nlayers_enc)]
        for lstm_name in self.lstm_rev_enc:
            self.add_link(lstm_name, L.LSTM(n_units, n_units))
        '''
        ___QUESTION-1-DESCRIBE-A-END___
        '''

        #--------------------------------------------------------------------
        # add decoder layers
        #--------------------------------------------------------------------
        # add embedding layer
        '''
        ___QUESTION-1-DESCRIBE-B-START___
        Comment on the input and output sizes of the following layers:
        - L.EmbedID(vsize_dec, 2*n_units)
        - L.LSTM(2*n_units, 2*n_units)
        - L.Linear(2*n_units, vsize_dec)

        Why are we using multipliers over the base number of units (n_units)?

        Since we use bidirectional encoder we need to embed the result of encoder to feed it to decoder. Thus, we use 2 times n_units for the output of embedID. In addition, because the input of encoder LSTM layers is EmbedID so the first dimention and second dimention (this is the designer decision to set the second dimension to whatever he wants.) of the LSTM layers are 2*n_units. Moreover, the size of the first dimension of linear layer or output layer should be equall to the second dimension of LSTM layer because LSTM layer is the input of linear layer and its second dimension should be equall to size of vocabulary since it is the probability distribution over our vocabulary to choose the best word.  

        '''

        self.add_link("embed_dec", L.EmbedID(vsize_dec, 2*n_units))

        # add LSTM layers
        self.lstm_dec = ["L{0:d}_dec".format(i) for i in range(nlayers_dec)]
        for lstm_name in self.lstm_dec:
            self.add_link(lstm_name, L.LSTM(2*n_units, 2*n_units))

        if attn > 0:
            # __QUESTION Add attention
            pass

        # Save the attention preference
        # __QUESTION you should use this flag to check if attention
        # has been selected. Your code should work with and without attention
        self.attn = attn

        # add output layer
        self.add_link("out", L.Linear(2*n_units, vsize_dec))
        '''
        ___QUESTION-1-DESCRIBE-B-END___
        '''

        # Store GPU id
        self.gpuid = gpuid
        self.n_units = n_units

    def reset_state(self):
        # reset the state of LSTM layers
        for lstm_name in self.lstm_enc + self.lstm_rev_enc + self.lstm_dec:
            self[lstm_name].reset_state()
        self.loss = 0

    '''
        ___QUESTION-1-DESCRIBE-C-START___

        Describe what the function set_decoder_state() is doing. What are c_state and h_state?
	This function feeds cell state and output of the last encoder layer(forward and backward direction) to the first decoder layer. In other words, it transfers output of the encoder to the decoder.
    '''
    def set_decoder_state(self):
        xp = cuda.cupy if self.gpuid >= 0 else np
        c_state = F.concat((self[self.lstm_enc[-1]].c, self[self.lstm_rev_enc[-1]].c))
        h_state = F.concat((self[self.lstm_enc[-1]].h, self[self.lstm_rev_enc[-1]].h))
        self[self.lstm_dec[0]].set_state(c_state, h_state)

    '''___QUESTION-1-DESCRIBE-C-END___'''

    '''
    Function to feed an input word through the embedding and lstm layers
        args:
        embed_layer: embeddings layer to use
        lstm_layer:  list of names of lstm layers to use
    '''
    def feed_lstm(self, word, embed_layer, lstm_layer_list, train):
        # get embedding for word
        embed_id = embed_layer(word)
        #embed_id=F.dropout(embed_id, ratio=0.2, train=train)
        # feed into first LSTM layer
        hs = self[lstm_layer_list[0]](embed_id)
        # feed into remaining LSTM layers
        for lstm_layer in lstm_layer_list[1:]:
            #hs=F.dropout(hs, ratio=0.2, train=train)
            hs = self[lstm_layer](hs)

    # Function to encode an source sentence word
    def encode(self, word, lstm_layer_list, train):
        self.feed_lstm(word, self.embed_enc, lstm_layer_list, train)

    # Function to decode a target sentence word
    def decode(self, word, train):
        self.feed_lstm(word, self.embed_dec, self.lstm_dec, train)

    '''

    '''
    def encode_list(self, in_word_list, train=True):
        xp = cuda.cupy if self.gpuid >= 0 else np
        # convert list of tokens into chainer variable list
        var_en = (Variable(xp.asarray(in_word_list, dtype=np.int32).reshape((-1,1)),
                           volatile=(not train)))

        var_rev_en = (Variable(xp.asarray(in_word_list[::-1], dtype=np.int32).reshape((-1,1)),
                           volatile=(not train)))

        # array to store hidden states for each word
        # enc_states = xp.empty((0,2*self.n_units), dtype=xp.float32)
        first_entry = True

        # encode tokens
        for f_word, r_word in zip(var_en, var_rev_en):
           '''
           ___QUESTION-1-DESCRIBE-D-START___

           - Explain why we are performing two encode operations
	   To consider the importance of words' location into account we have two encoder that the first one goes from left to right and the second one goes from right to left. The following two lines feed the sentence in forward and backward direction word by word to the corresponding encoder (the first one for forward and the scond one foe backward traverse). 
           '''
           self.encode(f_word, self.lstm_enc, train)
           self.encode(r_word, self.lstm_rev_enc, train)

           '''___QUESTION-1-DESCRIBE-D-END___'''


           # __QUESTION -- Following code is to assist with ATTENTION
           # enc_states stores the hidden state vectors of the encoder
           # this can be used for implementing attention
           if first_entry == False:
              forward_states = F.concat((forward_states, self[self.lstm_enc[-1]].h), axis=0)
              backward_states = F.concat((self[self.lstm_rev_enc[-1]].h, backward_states), axis=0)
           else:
              forward_states = self[self.lstm_enc[-1]].h
              backward_states = self[self.lstm_rev_enc[-1]].h
              first_entry = False
    
        enc_states = F.concat((forward_states, backward_states), axis=1)
        return enc_states


    # Select a word from a probability distribution
    # should return a chainer variable
    def select_word(self, prob, train=True, sample=False):
        xp = cuda.cupy if self.gpuid >= 0 else np
        if not sample:
            indx = xp.argmax(prob.data[0])
            pred_word = Variable(xp.asarray([indx], dtype=np.int32), volatile=not train)
        else:

            sam=np.random.choice(range(len(prob.data[0])), p=prob.data[0])

            pred_word = Variable(xp.asarray([sam], dtype=np.int32), volatile=not train)
            '''
            ___QUESTION-2-SAMPLE

            - Add code to sample from the probability distribution to
            choose the next word
            '''
        return pred_word

    def encode_decode_train(self, in_word_list, out_word_list, train=True, sample=False):
        xp = cuda.cupy if self.gpuid >= 0 else np
        self.reset_state()
        # Add GO_ID, EOS_ID to decoder input
        decoder_word_list = [GO_ID] + out_word_list + [EOS_ID]
        # encode list of words/tokens
        enc_states = self.encode_list(in_word_list, train=train)
        # initialize decoder LSTM to final encoder state
        self.set_decoder_state()
        # decode and compute loss
        # convert list of tokens into chainer variable list
        var_dec = (Variable(xp.asarray(decoder_word_list, dtype=np.int32).reshape((-1,1)),
                            volatile=not train))
        # Initialise first decoded word to GOID
        pred_word = Variable(xp.asarray([GO_ID], dtype=np.int32), volatile=not train)

        # compute loss
        self.loss = 0
        # decode tokens
        for next_word_var in var_dec[1:]:
            self.decode(pred_word, train=train)
            if self.attn == NO_ATTN:
                predicted_out = self.out(self[self.lstm_dec[-1]].h)
            else:
                
                alpha_arr = xp.empty((0,enc_states.shape[0]), dtype=xp.float32)
                alpha_arr = (xp.dot(self[self.lstm_dec[-1]].h.data,enc_states.data.T))
                alpha_arr = F.softmax(alpha_arr)

                ct = Variable(xp.dot(alpha_arr.data,enc_states.data), volatile="auto")
                h_o = F.concat((ct, self[self.lstm_dec[-1]].h), axis=1)

                predicted_out = F.tanh(self.att(h_o))
                predicted_out = self.out(predicted_out)
            # compute loss
            prob = F.softmax(predicted_out)

            pred_word = self.select_word(prob, train=train, sample=False)
            # pred_word = Variable(xp.asarray([pred_word.data], dtype=np.int32), volatile=not train)
            '''
            ___QUESTION-1-DESCRIBE-E-START___
            Explain what loss is computed with an example
            What does this value mean?
	    Soft max cross entropy computes the probability distribution over each possible word in  our vocabulary and then computes the difference between predicted probabilities and the actual word that should appear
            '''
            self.loss += F.softmax_cross_entropy(predicted_out, next_word_var)
            '''___QUESTION-1-DESCRIBE-E-END___'''

        report({"loss":self.loss},self)

        return self.loss

    def decoder_predict(self, start_word, enc_states, max_predict_len=MAX_PREDICT_LEN, sample=False):
        xp = cuda.cupy if self.gpuid >= 0 else np

        # __QUESTION -- Following code is to assist with ATTENTION
        # alpha_arr should store the alphas for every predicted word
        alpha_arr = xp.empty((0,enc_states.shape[0]), dtype=xp.float32)

        # return list of predicted words
        predicted_sent = []
        # load start symbol
        pred_word = Variable(xp.asarray([start_word], dtype=np.int32), volatile=True)
        pred_count = 0

        # start prediction loop
        while pred_count < max_predict_len and (int(pred_word.data) != (EOS_ID)):
            self.decode(pred_word, train=False)

            if self.attn == NO_ATTN:
                prob = F.softmax(self.out(self[self.lstm_dec[-1]].h))
            else:
                
                alpha_arr2 = xp.empty((0,enc_states.shape[0]), dtype=xp.float32)
                alpha_arr2 = (xp.dot(self[self.lstm_dec[-1]].h.data,enc_states.data.T))
                alpha_arr2 = F.softmax(alpha_arr2)

                alpha_arr = F.concat((alpha_arr, alpha_arr2), axis=0)

                ct = Variable(xp.dot(alpha_arr2.data,enc_states.data), volatile="auto")
                h_o = F.concat((ct, self[self.lstm_dec[-1]].h), axis=1)
                predicted_out = F.tanh(self.att(h_o))
                predicted_out = self.out(predicted_out)

            pred_word = self.select_word(prob, train=False, sample=sample)
            # add integer id of predicted word to output list
            predicted_sent.append(int(pred_word.data))
            pred_count += 1
        # __QUESTION Add attention
        # When implementing attention, make sure to use alpha_array to store
        # your attention vectors.
        # The visualisation function in nmt_translate.py assumes such an array as input.
        return predicted_sent, alpha_arr

    def encode_decode_predict(self, in_word_list, max_predict_len=20, sample=False):
        xp = cuda.cupy if self.gpuid >= 0 else np
        self.reset_state()
        # encode list of words/tokens
        in_word_list_no_padding = [w for w in in_word_list if w != PAD_ID]
        enc_states = self.encode_list(in_word_list, train=False)
        # initialize decoder LSTM to final encoder state
        self.set_decoder_state()
        # decode starting with GO_ID
        predicted_sent, alpha_arr = self.decoder_predict(GO_ID, enc_states,
                                                         max_predict_len, sample=sample)
        return predicted_sent, alpha_arr

# In[]:
