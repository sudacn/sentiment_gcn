#! /usr/bin/env python
#coding=utf-8

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
from data import Review

maxlen=50
emb_dim = 128
h_dim = 128
batch_size=128
epoch=10

MAX_DEGREE=2


PATH="/data2/nchen/sentiment/data/finalV2.csv"
LOCAL2ID = {'安徽': 0, '陕西': 1, '浙江': 2, '湖北': 3, '西藏': 4, '其他': 5, '甘肃': 6, '贵州': 7, '云南': 8, '湖南': 9, '内蒙古': 10, '江苏': 11, '福建': 12, '海外': 13, '山东': 14, '吉林': 15, '河北': 16, '重庆': 17, '四川': 18, '香港': 19, '上海': 20, '青海': 21, '新疆': 22, '天津': 23, '北京': 24, '广西': 25, '海南': 26, '山西': 27, '辽宁': 28, '河南': 29, '广东': 30, '江西': 31, '黑龙江': 32}
SENTI2ID = {'喜':0,'怒':1,'惊':2,'恐':3,'哀':4}
SENTILABED = [0,0,0,0,0]

def get_vocabrary(reviews):
    # DF
    df = {}
    for r in reviews:
        for w in r.text.split():
            if w not in df:
                df[w] = 0
            df[w] += 1
    
    df=sorted([(df[w],w) for w in df])
    df.reverse()
    V = {}
    for score,w in df:
        V[w] = len(V) 
    return V

def format_k(text,V):
    return [V[w] for w in text.split() if w in V]

def format_data(reviews,V):
    X=[]
    Y=[]
    
    for r in reviews:
        X.append(format_k(r.text,V))
        Y.append(r.label)
    X = np.array(X)
    Y = np.array(Y)

    return X,Y

def eval(tests,pred_y):
    # acc
    n=0
    pred_y = np.array(pred_y)
    pred_y = np.where(pred_y<0.5, 0, 1)
    #pred_y = pred_y.tolist()
    #print(pred_y)
    for i in range(len(tests)):
        
        # if pred_y[i]>0.5:
        #     py=1
        # else:
        #     py=0
        
        if (pred_y[i]==tests[i].label).all():
            print(pred_y[i],tests[i].label)
            print(tests[i].text)
            n+=1
            
    print('Acc:',n/len(tests))

def padding(x,ml=maxlen):
    new_x = []
    for i in x:
        if len(i)<ml:
            x0 = i + [0] * (ml-len(i))
        else:
            x0 = i[:ml]
        new_x.append(x0)
        
    return np.array(new_x)

def get_basic_edges(reviews):
    n=len(reviews)

    edges=[]
    for i in range(n):
        for j in range(n):
            # same user id
            if reviews[i].uid==reviews[j].uid:
                edges.append([i,j])
            #same location
            if reviews[i].location==reviews[j].location:
                edges.append([i,j])
            #same gender
            if reviews[i].gender==reviews[j].gender:
                edges.append([i,j])
    print('Edges:',len(edges))

    return edges

def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm

def preprocess_adj(adj,n,symmetric=True):
    # prepare
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)

    return adj


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print(("Calculating Chebyshev polynomials up to order {}...".format(k)))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k

def preprocess(edges_unordered,n,symmetric=True):
    # matrix
    edges_unordered=np.array(edges_unordered)
    idx_map = {i: i for i in range(n)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj

def preprocess_chebyshev_adj(adj,n,symmetric=True):
    L = normalized_laplacian(adj, symmetric)
    L_scaled = rescale_laplacian(L)
    T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
    support = MAX_DEGREE + 1

    return T_k,support


def get_train_mask(n,review):
    train_mask=[False for i in range(n)]

    for i in range(len(review)):
        #print(review[i])
        if review[i].label.tolist() == [0,0,0,0,0]:
            #print(i)
            continue
        train_mask[i]=True

    train_mask=np.array(train_mask)

    return train_mask


def prepare_data(trains,tests,V):
    reviews=trains+tests

    u_dict={}
    for r in reviews:
        if r.uid not in u_dict:
            u_dict[r.uid]=''
        u_dict[r.uid]+=' '+r.text

    u_reviews=[Review(0,uid,-1,text) for uid,text in list(u_dict.items())]

    reviews=reviews+u_reviews

    n=len(reviews)
    x,y=format_data(reviews,V)
    x=padding(x)
    
    edges=[]
    m=len(trains+tests)
    for i in range(m):
        for j in range(m,n):
            # same user id
            if reviews[i].uid==reviews[j].uid:
                edges.append([i,j])
    
    print('Edges:',len(edges))


    # local pool
    A=preprocess(edges,n)
    _A=preprocess_adj(A,n)
    graph = [x, _A]

    # train mask
    train_mask=get_train_mask(n,len(trains))

    return x,y,A,_A,train_mask


