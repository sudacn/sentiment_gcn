#! /usr/bin/env python
#coding=utf-8
from data import read_file
from util import *
from nn import lstm,transformer
from gnn import gcn,gcn_chebyshev

trains,tests=read_file('Automotive_5')

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
train_mask=get_train_mask(n,len(trains))

model=gcn(len(V),x.shape[-1],A.shape[0],1) # vocab_size,x_shape,A_shape,label_size
model.fit(graph, y,nb_epoch=100,sample_weight=train_mask,batch_size=A.shape[0],shuffle=False)
preds = model.predict(graph, batch_size=A.shape[0])
pred_y=preds[len(trains):]

# Evaluation
eval(tests,pred_y)
