from math import fabs
from os import write
import jieba_fast as jieba
import numpy as py
import pandas as pd
import csv
import random
import re
from multiprocessing import Pool,Manager,Lock
import time as tu

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


def handle(data,m,n,stop_list):
    part_list = []
    for i in range(m,n):
        t = data[i]
        s = re.sub(u'[^\u4e00-\u9fa5|\s]', "", t).replace('\u3000','')
        jlist = jieba.lcut(s, cut_all=True)  #为每个文档分词
        doc = []
        for wd in jlist:
            part_list.append(wd)
            if wd not in stop_list:
                doc.append(wd)
        data[i] =  ','.join(doc)
    print('done:'+str(n))
    return list(set(part_list))
    
def pretreatment():
    t_start=tu.time()
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

    print('\n generate words_list')
    l_start = tu.time()

    suml = []
    for res in res_list:
        temp = res.get()
        suml = suml + temp
    
    words_list = list(set(suml))

    l_end = tu.time()
    l_time = l_end-l_start
    print ('the wordslist time is :%s' %l_time)

    #保存处理结果
    with open('temp_data.csv','w',encoding='utf-8') as f:
        writer = csv.writer(f)
        for i in range(size):
            row = [data[i],category[i]]
            writer.writerow(row)
        f.close()

    with open('words_list','w',encoding='utf-8') as f:
        f.write(','.join(words_list))
        f.close

    print("words len:"+str(len(words_list)))
    

    t_end=tu.time()
    time=t_end-t_start

    print('complete')
    print ('the program time is :%s' %time)
    

if __name__ == '__main__':
   pretreatment()

