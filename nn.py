#! /usr/bin/env python
#coding=utf-8
import numpy as np
from keras.layers import *
from keras_layer_normalization import LayerNormalization
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from nn_new import *
from util import *

def lstm(v_size):
    x_in = Input(shape=(None,))
       
    embedding = Embedding(v_size, emb_dim)
    e = embedding(x_in)
       
    h = LayerNormalization()(e)
    h = LSTM(h_dim)(h)
   
    out = Dense(5, activation='sigmoid')(h)
    
    model = Model(x_in, out)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model

def transformer(v_size):
    x_in = Input(shape=(None,))
       
    embedding = Embedding(v_size, emb_dim, trainable=True)
    e = embedding(x_in)
    
    h = Attention(8, 16)([e, e, e])
    
    # pooling
    h = GlobalAveragePooling1D()(h)

    out = Dense(5, activation='sigmoid')(h)
    
    model = Model(input=x_in, output=out)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model

def transformer_deepwalk(g_size,embeddings_matrix):
    x_in = Input(shape=(None,))
       
    embedding = Embedding(g_size, 64, weights = [embeddings_matrix], trainable = False)
    e = embedding(x_in)
    
    h = Attention(8, 16)([e, e, e])
    
    # pooling
    h = GlobalAveragePooling1D()(h)

    out = Dense(1, activation='sigmoid')(h)
    
    model = Model(input=x_in, output=out)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model