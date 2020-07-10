from data import read_file
from util import *
from nn import lstm,transformer
from gnn import gcn,gcn_chebyshev

trains,tests=read_file('Automotive_5')

V=get_vocabrary(trains)

x,y,A,_A,train_mask=prepare_data(trains,tests,V)

graph = [x, _A]

model=gcn(len(V),x.shape[-1],A.shape[0],1) # vocab_size,x_shape,A_shape,label_size
model.fit(graph, y,nb_epoch=100,sample_weight=train_mask,batch_size=A.shape[0],shuffle=False)
preds = model.predict(graph, batch_size=A.shape[0])
pred_y=preds[len(trains):len(trains+tests)]

# Evaluation
eval(tests,pred_y)