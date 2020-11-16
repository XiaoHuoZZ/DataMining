from math import fabs
from os import write
import jieba
import numpy as np
import pandas as pd
import csv
import random
import re
from multiprocessing import Pool,Manager,Lock,Array
import time as tu



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

def partition(data_path):
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


def handle_fenci(data,m,n,stop_list,cat,dic):
    for i in range(m,n):
        t = data[i]
        s = re.sub(u'[^\u4e00-\u9fa5|\s]', "", t).replace('\u3000','')
        jlist = jieba.lcut(s, cut_all=False)  #为每个文档分词
        doc = []
        for wd in jlist:
            if wd not in stop_list:
                dic[cat[i]].append(wd)
                doc.append(wd)
        data[i] =  ','.join(doc)
    print('done:'+str(n))
    return dic
    
def handle_transform():
    return

def pretreatment():
    t_start=tu.time()
    res_list=[]
    pool = Pool(11)
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
        
        cat_list = list(set(category))
        dic = {}
        for cat in cat_list:
            dic[cat] = []
        
        print('load')
        size = len(data)
        ratio = 10000
        t = size//ratio
        offset = size-t*ratio
        for i in range(t):
            res = pool.apply_async(func=handle_fenci, args=(data,i*ratio,(i+1)*ratio,stop_list,category,dic,))
            res_list.append(res)
        res = pool.apply_async(func=handle_fenci, args=(data,t*ratio,t*ratio+offset,stop_list,category,dic,))
        res_list.append(res)
        f.close

    
    pool.close()
    pool.join()

    #去重
    print('\n generate words_list')
    l_start = tu.time()

    dic = {}
    for cat in cat_list:
        dic[cat] = []
    
    for res in res_list:
        temp = res.get()
        for cat in cat_list:
            nL = dic[cat] + temp[cat]
            dic[cat] = list(set(nL))
            
    l_end = tu.time()
    l_time = l_end-l_start
    print ('the wordslist time is :%s' %l_time)

    #保存处理结果
    for i in range(size):
        doc = data[i]
        cat = category[i]
        with open('./temp/'+ cat +'.csv','a',encoding='utf-8',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([doc])
            f.close

    with open('words_list','w',encoding='utf-8') as f:
        f.write(str(dic))
        f.close

    t_end=tu.time()
    time=t_end-t_start

    
    print ('the program time is :%s' %time)
    print('complete')
    
def transform():
    print('start')
    with open('words_list',encoding='utf-8') as f:
        dic = eval(f.read())
        f.close()
    
    cat_list = list(dic.keys())

    start = tu.time()

    bow = {}
    for cat in cat_list:
        szie  = len(dic[cat])
        bow[cat] = np.zeros(szie)

    for cat in cat_list:
        with open('temp/'+ cat +'.csv',encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                doc = row[0].split(',')
                for d in doc:
                    i = dic[cat].index(d) #查找该词在词典中的位置
                    bow[cat][i] = bow[cat][i] + 1
    
    end = tu.time()

    print(end-start)


   



if __name__ == '__main__':
    # pretreatment()
    # cat_list = ['2008','auto','business','career','category','cul','health','house','it','learning','mil','news','sports','travel','women','yule']
    transform()

