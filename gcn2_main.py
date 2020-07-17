#! /usr/bin/env python
#coding=utf-8
#喜、怒、惊、恐、哀 五种标签
# keras== 2.2.4
# tensorflow == 1.15.2

from data import *
from util import *
from nn import lstm,transformer
from gnn import gcn,gcn_chebyshev
import pandas as pd
import numpy as np

def main():
    # PATH = "/data2/nchen/sentiment/data/finalV2.csv"

    # trains,tests = read_file(PATH)
    # #trains,tests=read_file('Automotive_5')
    # V=get_vocabrary(trains)
    
    # reviews=trains+tests
    # n=len(reviews)
    # x,y=format_data(reviews,V)
    # x=padding(x)
    # edges=get_basic_edges(reviews)

# [1200,12]
# [1200,12,128]
# cnn1d
# [1200, dim]
# [1200,1200]

    # local pool
    # A=preprocess(edges,n)
    # _A=preprocess_adj(A,n)
    # graph = [x, _A]
    _A = np.ones([10,10])
    x = np.random.randn(10,20)
    graph = [x, _A]
    y = np.random.randn(10,5)
    # train mask
    # train_mask=get_train_mask(n,len(trains))
    #model =lstm(len(V))
    model=gcn(10,x.shape[-1],5,5) # vocab_size,x_shape,A_shape,label_size
    model.summary()
    model.fit(graph, y,epochs=100,batch_size=x.shape[0],shuffle=False)
    preds = model.predict(graph, batch_size=A.shape[0])
    pred_y=preds[len(trains):]

    # Evaluation
    eval(tests,pred_y)

main()   