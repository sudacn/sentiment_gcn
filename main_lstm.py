#! /usr/bin/env python
#coding=utf-8
from data import read_file
from util import *
from nn import lstm,transformer
#from gnn import gcn,gcn_chebyshev
import os 

os.environ["CUDA_VISIBLE_DEVICES"]="1"

PATH = "/data2/nchen/sentiment/data/finalV2.csv"
trains,tests = read_file(PATH)
V=get_vocabrary(trains)

reviews=trains+tests
n=len(trains)
x,y=format_data(trains,V)
x=padding(x)

test_x,test_y=format_data(tests,V)
test_x=padding(test_x)

# local pool

# train mask
train_mask=get_train_mask(n,trains)
#print(x.shape[-1],A.shape[0])
model=lstm(len(V)) # vocab_size,x_shape,A_shape,label_size
model.summary()
model.fit(x, y,epochs=100,validation_data=(test_x, test_y),batch_size=batch_size,shuffle=True,verbose=1)
preds = model.predict(test_x, batch_size=batch_size)

pred_y = preds
print(pred_y)
# Evaluation
eval(tests,pred_y)