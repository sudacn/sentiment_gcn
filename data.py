#-*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
#from util import *
#! /usr/bin/env python
#coding=utf-8

#Automotive_5
#Cell_Phones_and_Accessories_5
#Software_5
#Video_Games_5
SENTI2ID = {'喜':0,'怒':1,'惊':2,'恐':3,'哀':4}
LOCAL2ID = {'安徽': 0, '陕西': 1, '浙江': 2, '湖北': 3, '西藏': 4, '其他': 5, '甘肃': 6, '贵州': 7, '云南': 8, '湖南': 9, '内蒙古': 10, '江苏': 11, '福建': 12, '海外': 13, '山东': 14, '吉林': 15, '河北': 16, '重庆': 17, '四川': 18, '香港': 19, '上海': 20, '青海': 21, '新疆': 22, '天津': 23, '北京': 24, '广西': 25, '海南': 26, '山西': 27, '辽宁': 28, '河南': 29, '广东': 30, '江西': 31, '黑龙江': 32}

class Review:
    def __init__(self,label,review,location,gender,uid):
        self.label=label # binary
        self.text=review # user review
        self.location=location # location
        self.gender=gender
        self.uid=uid

def sen2word(sen, stopwords):
    seg_list = jieba.cut(sen)
    emoji = re.findall(r'\[(.*?)\]', sen)
    words = []
    for word in seg_list:
        transword = word.encode('utf-8').strip()
        if len(transword) > 0:
            if transword not in stopwords:
                words.append(transword)
    words = words + emoji
    words_set = set(words)
    return words, words_set

def wordsstatistic(sen, stopwords):
    seg_list = jieba.cut(sen)
    emoji = re.findall(r'\[(.*?)\]', sen)
    words = []
    for word in seg_list:
        transword = word.encode('utf-8').strip()
        if len(transword) > 0:
            if transword not in stopwords:
                words.append(transword)
    words = words + emoji
    sta = {}
    for word in words:
        if word not in sta.keys():
            sta[word] = 1
        else:
            sta[word] += 1
    return sta

    
def local2id(file):
    data = pd.read_csv(file)
    locations = data['location'].values.tolist()
    locations = set(locations)
    localdict={}
    for num,loca in enumerate(locations):
        localdict[loca] = num
    return localdict

def gender2id(gender):
    if gender =='f':
        return 0
    else:
        return 1

def location2id(location):
    local = LOCAL2ID[location]
    return local

def emoji2id(emoji):
    senti = []
    SENTILABED = [0,0,0,0,0]
    if emoji != "nan":
        for i in emoji:
            num = SENTI2ID[i]    
            SENTILABED[num]=1
        
    return SENTILABED


def read_file(domain):
    reviews=[]
    data = pd.read_csv(domain)

    for index,row in data.iterrows():
        
        label = emoji2id(str(row['senti']))
        label = np.array(label)

        location = location2id(row["location"])

        text = row["review"]

        gender = row["gender"]
        gender = gender2id(gender)

        uid = row["id"]
        uid = int(uid)
        # if index < 3000:
        #     print(label,text,location,gender,uid)
        review = Review(label,text,location,gender,uid)
        #print(review)
        reviews.append(review)
        #break

    # for line in open(domain,'r'):
    #     line=line.strip()
    #     if len(line)>0:
    #         p=line.split('###')
    #         if len(p)==4:
    #             label,uid,pid,text=p
    #             label=int(label)
    #             uid=uid.strip()
    #             pid=pid.strip()
    #             text=text.strip()
    #             reviews.append(Review(label,uid,pid,text))

    # print(len(reviews))

    trains=reviews[:5000]
    tests=reviews[5000:6000]

    return trains,tests
