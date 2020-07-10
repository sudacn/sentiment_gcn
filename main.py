#! /usr/bin/env python
#coding=utf-8
from data import read_file
from util import *
from nn import lstm,transformer
from gnn import gcn

trains,tests=read_file('Automotive_5')

V=get_vocabrary(trains)

# format
train_x,y=format_data(trains,V)
test_x,_=format_data(tests,V)

train_x=padding(train_x)
test_x=padding(test_x)

# LSTM
#model=lstm(len(V))
#model.fit(train_x, y, nb_epoch=10)
#pred_y=model.predict(test_x)

# Transformer
model=transformer(len(V))
model.fit(train_x, y, nb_epoch=10,shuffle=False)
pred_y=model.predict(test_x)

# Evaluation
eval(tests,pred_y)