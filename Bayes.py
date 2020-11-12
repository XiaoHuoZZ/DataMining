from math import fabs
import jieba
import numpy as py
import pandas as pd
import csv
import random
import re
from multiprocessing import Pool,Manager,Lock
import time

data_path='./data/data.csv'

def random_index(rate):
    # """随机变量的概率函数"""
    # 参数rate为list<int>
    # 返回概率事件的下标索引
    start = 0
    index = 0
    randnum = random.randint(1, sum(rate))
    for index, scope in enumerate(rate):
        start += scope
        if randnum <= start:
            break
    return index


def partition():
    with open(data_path, 'r',encoding='utf-8') as f:
        reader = csv.reader(f)
        tr=open('./data/train.csv', 'a', newline='',encoding='utf-8')
        te=open('./data/test.csv', 'a', newline='',encoding='utf-8')
        writer1 = csv.writer(tr) 
        writer2 = csv.writer(te)
        for row in reader:
            i=random_index([90,10])
            if i==0:
                writer1.writerow(row) 
            else:
                writer2.writerow(row) 
        f.close
        tr.close
        te.close
    print('划分完毕')


def pretreatment():
    words_list=[]
    with open('./data/train.csv', 'r',encoding='utf-8') as f:
        data = pd.read_csv(f)
        size = data.index.size
        for i in range(size):
            t = data.iloc[i,:]['content']
            s = re.sub(u'[^\u4e00-\u9fa5|\s]', "", t).replace('\u3000','')
            jlist = jieba.lcut(s, cut_all=True)  #为每个文档分词
            newList = []
            for j in jlist:
                if j not in stop_list:
                    newList.append(j)
                    if j not in words_list:
                        words_list.append(j)
            print(i)
            data.iloc[i,:]['content'] = ','.join(newList)
        
    # with open('./data/train.csv', 'r',encoding='utf-8') as f:
    #     reader = csv.reader(f)
    #     first = True  #判断是否为表头
    #     print('开始预处理')
    #     for row in reader:
    #         if first is True:
    #             first=False
    #             continue
    #         s = row[2]
    #         con = re.sub(u'[^\u4e00-\u9fa5|\s]', "", s)
    #         jlist = jieba.lcut(con, cut_all=True)  #为每个文档分词
    #         with open('') as w:
                
    #         # print(jlist)
    #         # for j in jlist:
    #         #     if j not in stop_list and j not in  words_list:
    #         #         words_list.append(j)
    print('预处理完毕')
    print(words_list)
    return

def handle(data,m,n,stop_list):
    for i in range(m,n):
        t = data[i]
        s = re.sub(u'[^\u4e00-\u9fa5|\s]', "", t).replace('\u3000','')
        jlist = jieba.lcut(s, cut_all=True)  #为每个文档分词
        doc = []
        for wd in jlist:
            if wd not in stop_list:
                doc.append(wd)
        data[i] =  ','.join(doc)
    

if __name__ == '__main__':
    t_start=time.time()
    res_list=[]
    pool = Pool(11)
    words_list = []
    stop_list=[]
    category = []
    #加载停用词表
    with open('stop_words.txt','r') as f:
        for line in f:
            stop_list.append(line.strip('\n'))
        f.close()
    with open('./data/train.csv', 'r',encoding='utf-8') as f:
        manager = Manager()
        reader = csv.reader(f)
        data = manager.list()
        for i,row in enumerate(reader):
            if i != 0:
                data.append(row[2])
                category.append(row[0])

        print('load')
        size = len(data)
        ratio = 10000
        t = size//ratio
        offset = size-t*ratio
        for i in range(t):
            res = pool.apply_async(func=handle, args=(data,i*ratio,(i+1)*ratio,stop_list,))
            res_list.append(res)
        res = pool.apply_async(func=handle, args=(data,t*ratio,t*ratio+offset,stop_list,))
        res_list.append(res)
        # res = pool.map(handle,[data,data,data,data,data])    
        f.close

    
    pool.close()
    pool.join()


    t_end=time.time()
    time=t_end-t_start

    print('complete')
    print ('the program time is :%s' %time)

