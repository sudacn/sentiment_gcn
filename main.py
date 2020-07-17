#! /usr/bin/env python
#coding=utf-8
from data import read_file
from util import *
from nn import lstm,transformer
from gnn import gcn,gcn_chebyshev
import os 

os.environ["CUDA_VISIBLE_DEVICES"]="1"

PATH = "/data2/nchen/sentiment/data/finalV2.csv"
#trains,tests=read_file('Automotive_5')
trains,tests = read_file(PATH)
V=get_vocabrary(trains)

reviews=trains+tests
n=len(reviews)
x,y=format_data(reviews,V)
x=padding(x)
edges=get_basic_edges(reviews)


# local pool
A=preprocess(edges,n)
_A=preprocess_adj(A,n)
graph = [x, _A]

# train mask
train_mask=get_train_mask(n,trains)
#print(x.shape[-1],A.shape[0])
model=gcn(len(V),x.shape[-1],A.shape[0],5) # vocab_size,x_shape,A_shape,label_size
model.summary()
model.fit(graph, y,epochs=100,sample_weight=train_mask,batch_size=A.shape[0],shuffle=False)
preds = model.predict(graph, batch_size=A.shape[0])

pred_y=preds[len(trains):]
print(pred_y)
# Evaluation
eval(tests,pred_y)
