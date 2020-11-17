from math import fabs
from os import write
import jieba
import numpy as np
import pandas as pd
import csv
import random
import re
from multiprocessing import Pool,Manager,Lock,Array,Process
import time as tu
from functools import partial



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
    with open('./data/data.csv', 'r',encoding='utf-8') as f:
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
        f.close()
        tr.close()
        te.close()
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
    
def handle_tf(cat,dic):
    pool = Pool(2)
    bow = np.zeros(len(dic[cat]))
    with open('temp/'+ cat +'.csv',encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            doc = row[0].split(',')
            res = pool.map(partial(handle_tf1,l=dic[cat]),doc)
            for i in res:
                if i>=0:
                    bow[i] = bow[i] + 1
            # for d in doc:
            #     i = dic[cat].index(d) #查找该词在词典中的位置
            #     bow[i] = bow[i] + 1
        pool.close()
        f.close()
    print('done:'+cat)
    return bow

def handle_tf1(d,l):
    if d == '':
        print('error')
        return -1
    i = l.index(d) #查找该词在词典中的位置
    return i


def pretreatment():
    t_start=tu.time()
    res_list=[]
    pool = Pool(12)
    manager = Manager()
    stop_list=[]
    category = []
    #加载停用词表
    with open('stop_words.txt','r') as f:
        for line in f:
            stop_list.append(line.strip('\n'))
        f.close()
    #加载数据
    with open('./data/train.csv', 'r',encoding='utf-8') as f:
        reader = csv.reader(f)
        data = manager.list()
        for i,row in enumerate(reader):
            if i != 0:
                data.append(row[2])
                category.append(row[0])
                if row[0] == 'category':
                    print(row[2])
        f.close()
    print('load')

    #获得种类列表
    cat_list = list(set(category))
    dic = {}
    for cat in cat_list:
        dic[cat] = []
        
   
    #任务分解
    size = len(data)
    ratio = 10000   #每个进程处理文档数
    t = size//ratio
    offset = size-t*ratio
    for i in range(t):
        res = pool.apply_async(func=handle_fenci, args=(data,i*ratio,(i+1)*ratio,stop_list,category,dic,))
        res_list.append(res)
    res = pool.apply_async(func=handle_fenci, args=(data,t*ratio,t*ratio+offset,stop_list,category,dic,))
    res_list.append(res)
    pool.close()
    pool.join()

    print('\n start save')
    l_start = tu.time()

    dic = {}
    for cat in cat_list:
        dic[cat] = []
    
    #取分词结果并保存到dic
    for res in res_list:
        temp = res.get()
        for cat in cat_list:
            nL = dic[cat] + temp[cat]
            dic[cat] = list(set(nL))  #去重
            
    #保存处理结果
    for i in range(size):
        doc = data[i]
        cat = category[i]
        with open('./temp/'+ cat +'.csv','a',encoding='utf-8',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([doc])
            f.close()

    with open('words_list','w',encoding='utf-8') as f:
        f.write(str(dic))
        f.close()

    l_end = tu.time()
    l_time = l_end-l_start
    print ('the save time is :%s' %l_time)


    t_end=tu.time()
    time=t_end-t_start

    
    print ('the program time is :%s' %time)
    print('data size:' + str(size))
    print('complete')
    
def transform():
    start = tu.time()
    res_list=[]
    manager = Manager()
    #加载字典
    with open('words_list',encoding='utf-8') as f:
        dic = manager.dict(eval(f.read()))
        f.close()
    
    #种类列表
    cat_list = list(dic.keys())

    #词袋模型
    # bow ={}
    # for cat in cat_list:
    #     szie  = len(dic[cat])
    #     bow[cat] = np.zeros(szie)

    print('start transform')

    
    for cat in cat_list:
        # res = pool.apply_async(func=handle_tf, args=(cat,dic,))
        # res_list.append(res)
        process = Process(target=handle_tf, args=(cat,dic))
        process.start()
        res_list.append(process)
    for res in res_list:
        res.join()


    
    # for cat in cat_list:
    #     with open('temp/'+ cat +'.csv',encoding='utf-8') as f:
    #         reader = csv.reader(f)
    #         for row in reader:
    #             doc = row[0].split(',')
    #             res = pool.map(partial(handle_tf1,l=dic[cat]),doc)
    #             print('done')
            # for d in doc:
            #     i = dic[cat].index(d) #查找该词在词典中的位置
            #     bow[i] = bow[i] + 1
            # f.close()

    end = tu.time()

    print(end-start)


   



if __name__ == '__main__':
    # partition()
    # pretreatment()
    # cat_list = ['2008','auto','business','career','category','cul','health','house','it','learning','mil','news','sports','travel','women','yule']
    transform()

