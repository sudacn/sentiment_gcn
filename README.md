# GCN SA

## Need to do
* 理解GCN  
## information
labeled 10415
## have done
* 数据预处理  
* 以LSTM做实验  
 1.在激活层前加BN效果更差  

## requirement
* 这个就是三部分，labeled+unlabeled+tests  
* unlabeled和test的标签都被mask，注意把mask那块修改一下  
* 测试只测试test的样本     ？无标签   
* 图：同location的连个边，同gender的连个边  

| model | LSTM | Transformer | GCN |
|:---:|:---:|:---:|:---:|
|acc(9000Train1000test)|0.57|0.421|(全无标签)|
||0.61|0.473|
||0.6|0.431|


 

## Solved Question
>结果都一样  原程序没有*jieba*分词

## Question 
LSTM 也半监督？
gcn 半监督 数据集划分
* transformer 训练的loss特别小
GCN 低ACC
* valid_split 不行 不能交叉验证？
* [GCN 忽略词序？]( https://zhuanlan.zhihu.com/p/56879815)
* GCN 结果都一样
没训练成功 Graph出错