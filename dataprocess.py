import os 
import re
import pandas as pd
import codecs
#读取labeled_corpus获得5个喜怒哀恐惊 对应句子
#读取id文件获取 id location gender review 
#方案二 id location gender review 
#status 与labeled 进行匹配
#数据清洗
#to labels 

def readFile(file):
    reviews = []
    sentis = []
    nameids = []
    f = open(file,"r")
    flines = f.readlines()
    for line in flines:
        if "</emo>" in line:
            review = re.findall("</emo>(.*)<num>",line)
            senti = re.findall("<emo>(.*?)</emo>",line)
            senti = "".join(senti)
        else :
            review = re.findall("(.*)<num>",line)
            senti = None
        review = "".join(review)
        nameid = re.findall("<num>(.*)</num>",line)
        nameid =int("".join(nameid)) 
        #print(nameid,"1")
        reviews.append(review)
        sentis.append(senti)
        nameids.append(nameid)
    #print(nameids)
    return nameids,reviews,sentis

def drop_dupliacte(file):
    dataframe = pd.read_csv(file,index_col=0,low_memory=False)
    final = dataframe.drop_duplicates(subset='review', keep='first', inplace=False)
    final.to_csv("/data2/nchen/sentiment/labeledV2.csv",encoding="utf_8_sig")

def mergecsv(labeledpath,csv):
    right = pd.DataFrame(csv,columns=['review','location','gender','id'])
    
    #right = right
    #right.to_csv('./right.csv')
    #right = right[]
    left = pd.read_csv(labeledpath,index_col=0,low_memory=False)
    left = pd.DataFrame(left,columns = ['senti','review'])
    print(left)
    print(right)
    # = left[]
    merge = pd.merge(left,right,on=['review'],how='right')
    merge = pd.DataFrame(merge,columns = ['senti','review','location','gender','id'])
    #left.update(right)
    merge.to_csv('/data2/nchen/sentiment/mergeV5.csv')
    #res = pd.concat([left, right], axis=0, join='outer',ignore_index=True)
    #merge.to_csv('/data2/nchen/sentiment/merge2.csv')

def mergemain(left,right):
    #left = pd.read_csv(left)
    right = pd.read_csv(right,index_col=0,low_memory=False)

    mergecsv(left,right)

def savecsv(filepath,data):
    labeled_data = pd.DataFrame(data)
    labeled_data.to_csv(filepath)
    print("ok")

def openidfile(filedir,PATHsave):
    listdir = os.listdir(filedir)
    for dir in listdir:
        filepath = os.path.join(filedir,dir)
        csv = read_idfile_extr(filepath)
        mergecsv(PATHsave,csv)

        

def read_idfile_extr(filepath):
    
    lines = codecs.open(filepath,'r', 'gbk').readlines()
    #print(lines[7])
    targetline = lines[1]
    #test = re.search("[0-9]{10}",targetline)
    location = re.findall(r"location=([\u4e00-\u9fa5]+)",targetline)
    #location = "".join(location)
    userid = re.findall(r"\d+",targetline)
    userid = "".join(userid)
    userid = [int(userid)]
    #print(userid)
    gender = re.findall("gender=(\w+)",targetline)
    #gender = "".join(gender)
    usefullines = []

    for line in lines[7:]:
        #取出网站 http://t.cn/
        garbas = re.findall("http://t.cn/\w{7}",line)
        if len(garbas)!=0:
            for garba in garbas:
                line = re.sub(garba,"",line)
                #print (garba)
        line = re.sub("\n","",line)
        line = re.sub("\r","",line)
        usefullines.append(line)
    length = len(usefullines)
    location = location*length 
    userid = userid*length
    gender = gender*length
    #print(usefullines)
    csv = {
        'id':userid,
        'location':location,
        'gender':gender,
        'review':usefullines,
    }
    return csv
    #return location,userid,gender,usefullines
    #print(gender)


if __name__ == "__main__":
    PATH = "/data2/nchen/sentiment/code/data_process/labeled_corpus.txt"
    idDir = "/data2/nchen/sentiment/code/data_process/data/1Kuser"
    PATHsave = "/data2/nchen/sentiment/merge1.csv"
    PATHtest = "/data2/nchen/sentiment/code/data_process/data/1Kuser/1003582744.txt"
    labelpath = "/data2/nchen/sentiment/labeled_data.csv"
    # nameids,reviews,sentis = readFile(PATH)

    # csv = {
    #     'senti':sentis,
    #     'id':nameids,
    #     'review':reviews,
    # }
    mergemain(merge1,concatfile)
    drop_dupliacte("/data2/nchen/sentiment/labeled_data_backupV1.csv")
    #openidfile(idDir,PATHsave)
    #csv = read_idfile_extr(PATHtest)
   # mergecsv(labelpath,csv)
    #print(csv['id'])
    #savecsv(PATHsave,csv)
    
    
    