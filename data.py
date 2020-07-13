#! /usr/bin/env python
#coding=utf-8

#Automotive_5
#Cell_Phones_and_Accessories_5
#Software_5
#Video_Games_5

class Review:
    def __init__(self,label,uid,pid,text):
        self.label=label # binary
        self.uid=uid # user id
        self.pid=pid # product id
        self.text=text

def read_file(domain):
    reviews=[]
    for line in open('../data/%s.txt' %domain,'rb'):
        line=line.strip()
        if len(line)>0:
            p=line.split('###')
            if len(p)==4:
                label,uid,pid,text=p

                label=int(label)
                uid=uid.strip()
                pid=pid.strip()
                text=text.strip()

                reviews.append(Review(label,uid,pid,text))
    print(len(reviews))

    trains=reviews[:3000]
    tests=reviews[3000:4000]

    return trains,tests
