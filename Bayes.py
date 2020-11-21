from math import fabs
import math
from os import error, write
import os
import jieba
import numpy as np
from numpy.core.fromnumeric import size
from numpy.core.numeric import ones
from numpy.lib.polynomial import poly
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
        train_size = 0
        test_size = 0
        tr=open('./data/train.csv', 'a', newline='',encoding='utf-8')
        te=open('./data/test.csv', 'a', newline='',encoding='utf-8')
        writer1 = csv.writer(tr) 
        writer2 = csv.writer(te)
        for row in reader:
            i=random_index([50,50])
            if i==0:
                writer1.writerow(row) 
                train_size = train_size +1
            else:
                writer2.writerow(row) 
                test_size = test_size + 1
        f.close()
        tr.close()
        te.close()
    print('划分完毕')
    print('train:' + str(train_size))
    print('test' + str(test_size))


def handle_fenci(data,m,n,stop_list,cat,dic):
    for i in range(m,n):
        t = data[i]
        s = re.sub(u'[^\u4e00-\u9fa5|\s]', "", t).replace('\u3000','')
        jlist = jieba.lcut(s, cut_all=False)  #为每个文档分词
        doc = []
        for wd in jlist:
            if wd not in stop_list:
                if len(jlist) > 5:   #分词结果大于5才能加入字典和处理后的结果
                    dic[cat[i]].append(wd)
                    doc.append(wd)
        # data[i] =  ','.join(doc)
        data[i] =  doc
    print('done:'+str(n))
    return dic
    
def handle_bow(cat,dic,cat_list):
    bow = np.zeros(len(dic[cat]),dtype=np.int)
    idf = {}
    for c in cat_list:
        idf[c] = np.zeros(len(dic[c]),dtype=np.int)

    with open('temp/'+ cat +'.csv',encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            doc = row[0].split(',')  
            for d in doc:
                if d != '':    #防止空文档
                    i = dic[cat][d] #查找该词在词典中的位置
                    bow[i] = bow[i] + 1  #词袋模型加1
            #生成部分idf
            for d in list(set(doc)):
                if d != '':    #防止空文档
                    for c in cat_list:
                        try:
                            i = dic[c][d] #查找该词在词典中的位置
                            idf[c][i] = idf[c][i] + 1   #如果该词存在于该种类字典，则文档数加1
                        except KeyError:
                            pass    #如果不存在，则略过
        f.close()
    print('done:'+cat)
    return (cat,bow,idf)
    
#预处理，包括分词，产生temp目录数据集，dic目录
def pretreatment(task):
    t_start=tu.time()
    res_list=[]
    pool = Pool(12)
    manager = Manager()
    stop_list=[]
    category = []
    #加载停用词表
    with open('stop_words.txt','r',encoding='utf-8') as f:
        for line in f:
            stop_list.append(line.strip('\n'))
        f.close()
    #加载数据
    with open('./data/'+ task +'.csv', 'r',encoding='utf-8') as f:
        reader = csv.reader(f)
        data = manager.list()
        for i,row in enumerate(reader):
            if i != 0:
                data.append(row[2])
                category.append(row[0])
                if row[0] == 'category':
                    print('error')
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

    if task =='train':
        #取分词结果并按种类保存到dic
        dic = {}
        for cat in cat_list:
            dic[cat] = []
    
        for res in res_list:
            temp = res.get()
            for cat in cat_list:
                nL = dic[cat] + temp[cat]
                dic[cat] = list(set(nL))  #去重
    
        for cat in cat_list:
            wd_indx = {}    #词字典
            for i,wd in enumerate(dic[cat]):
                wd_indx[wd] = i
            dic[cat] = wd_indx
        
        #保存分词后的结果
        for i in range(size):    
            doc = data[i]
            cat = category[i]
            if  doc:           #分词结果大于4的才保存
                with open('./temp/'+ cat +'.csv','a',encoding='utf-8',newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([','.join(doc)])
                    f.close()
         #保存字典
        for cat in cat_list:          
            with open('./dic/'+ cat + '_dic','w',encoding='utf-8') as f:
                f.write(str(dic[cat]))
                f.close()

    elif task == 'test':
        #测试集保存分词结果的格式与训练集不同
        with open('./temp/temp_test.csv','w',encoding='utf-8',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['doc','category'])
            for i in range(size):    
                doc = data[i]
                cat = category[i]
                if  doc :  #分词结果大于4才存
                    writer.writerow([','.join(doc),cat])
            f.close()
       

    l_end = tu.time()
    l_time = l_end-l_start
    print ('the save time is :%s' %l_time)


    t_end=tu.time()
    time=t_end-t_start

    
    print ('the program time is :%s' %time)
    print('data size:' + str(size))
    print('complete')

#产生tf-idf目录 ndic目录
def training(n):
    start = tu.time()
    res_list=[]
    
    #种类列表
    cat_list = []
    dirs = os.listdir('./dic') #列出文件夹下所有的目录与文件
    for d in dirs:
        cat_list.append(d.split('_')[0])

    print('start transform')

    #加载字典
    dic = {}
    for cat in cat_list:
        with open('dic/'+ cat +'_dic',encoding='utf-8') as f:
            t = eval(f.read())
            dic[cat] = t
    # m_dic = manager.dict(dic)


    pool = Pool(12)
    for cat in cat_list:
        res = pool.apply_async(func=handle_bow, args=(cat,dic,cat_list,))
        res_list.append(res)
    pool.close()
    pool.join()

    #取结果    1.按种类生成词袋模型  2.按种类合成idf
    bow = {}
    idf = {}
    size = 0


    for cat in cat_list: #获取文档总数
        with open('./temp/'+ cat +'.csv',encoding='utf-8') as f:  
            reader = csv.reader(f)
            for row in reader:
                
                size = size + 1

            f.close()

    for c in cat_list:
        idf[c] = np.zeros(len(dic[c]),dtype=np.int) 

    for res in res_list:
        temp = res.get()
        bow[temp[0]] = temp[1]
        for c in cat_list:
            idf[c] = idf[c] + temp[2][c]
    for c in cat_list:
        all_doc = np.zeros(len(dic[c]),dtype=np.int) + size  #总文档数
        idf[c] =np.log( all_doc / (idf[c]+1) )   #计算idf

    end = tu.time()
    print(end-start)
    print('end transform')
    # for cat in cat_list:
    #     np.save('./idf/'+ cat,idf[cat])
    #     bow[cat] = bow[cat] / bow[cat].sum()  #得到词频
    #     np.save('./tf/'+ cat,bow[cat])

    print('strat training')
    #计算TF-IDF 并选前n个保存

    v = []

    tj = {} #条件概率
    for cat in cat_list:
        tf = bow[cat] / bow[cat].sum()  #得到词频
        tf_idf = tf * idf[cat] #得到tf-idf
        indexs = (-tf_idf).argsort()[:n]  #前n个词坐标
    

        ntf_idf = np.zeros(n)
        for i,idx in enumerate(indexs):  #得到新的tf-idf
            ntf_idf[i] = tf_idf[idx]

        ndic = {}
        tdic = {v : k for k, v in dic[cat].items()}

        t = np.zeros(n)  #筛选后的词出现次数
        for i,idx in enumerate(indexs):     #得到新的字典
            wd = tdic[idx]  #对应的词
            ndic[wd] = i     
            t[i] = bow[cat][idx] #暂存出现次数
            v.append(wd)
        tj[cat] = t

        np.save('./tf-idf/'+ cat,ntf_idf)
        with open('./ndic/'+ cat +'_dic','w',encoding='utf-8') as f:
            f.write(str(ndic))

    v_size = len(set(v))  #筛选后的总词数

    #计算条件概率并保存
    for cat in cat_list:
        tj[cat] = (tj[cat] + 1) / (n + v_size)
        np.save('./tj/'+ cat,tj[cat])  #save

    
    pv = {}  #先验概率
    for c in cat_list:
        pv[c] = 0
        with open('./temp/'+ c +'.csv',encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                pv[c] = pv[c] + 1
            f.close()
    for cat in cat_list:
        pv[cat] =pv[cat] / size
    with open('pv','w',encoding='utf-8') as f:
        f.write(str(pv))
        f.close
    print('end training')
    
def cal(df,dic,tj,cat_list,pv,v_size):
    right = 0
    for i,row in df.iterrows():
        doc = row[0]
        cat = row[1]
        rc = 'news'  #预测出来的类
        isFirst = True
        m = 0
        for c in cat_list:
            res = math.log(pv[c])
            c_size = len(dic[c])
            for d in doc:
                if d != '':
                    try:
                        i = dic[c][d]
                        # print(tf_idf[c][i])
                        res = res + math.log(tj[c][i])
                    except KeyError:
                        t = 1 / (c_size + v_size)
                        res = res + math.log(t)
            if isFirst:
                m = res
                isFirst = False
            if res >= m:
                m = res
                rc = c
        if rc == cat:
            right = right + 1
    return right

def forecast():
    #种类列表
    cat_list = []
    dirs = os.listdir('./dic') #列出文件夹下所有的目录与文件
    for d in dirs:
        cat_list.append(d.split('_')[0])
    #加载 tj
    tj = {}
    for cat in cat_list:
        tj[cat] = np.load('./tj/'+ cat + '.npy')

    # for cat in cat_list:
    #     tf_idf[cat] = tf_idf[cat] / tf_idf[cat].sum()
    #加载字典
    dic = {}
    for cat in cat_list:
        with open('ndic/'+ cat +'_dic',encoding='utf-8') as f:
            t = eval(f.read())
            dic[cat] = t
    #加载测试集
    df = pd.read_csv('./temp/temp_test.csv')
    with open('pv',encoding='utf-8') as f:
        pv = eval(f.read())

    #计算总词数
    v_size = 0
    v = []
    for cat in cat_list:
        v = v + list(dic[cat].keys())
    v_size = len(set(v))
    
    print('start forecast')
    strat = tu.time()
    size = df.index.size
    right = 0

    pool = Pool(12)
    res_list = []

    #任务分解
    ratio = 10000   #每个进程处理文档数
    t = size//ratio
    offset = size-t*ratio
    print(size)
    for i in range(t):
        try:
            res = pool.apply_async(func=cal, args=(df[i*ratio,(i+1)*ratio],dic,tj,cat_list,pv,v_size,))
            res_list.append(res)
        except KeyError:
            print(i*ratio)
            print((i+1)*ratio)
    res = pool.apply_async(func=cal, args=(df[t*ratio,t*ratio+offset],dic,tj,cat_list,pv,v_size,))
    res_list.append(res)
    pool.close()
    pool.join()




    for res in res_list:
        right = right + res.get()

    ratio = right / size
    end  = tu.time
    l_time = end - strat

    print(ratio)
    print ('the time is :%s' %l_time)
    
    
if __name__ == '__main__':
    # partition()
    # pretreatment('test')
    # training(200)
    forecast()
    

