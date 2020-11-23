# -*- coding: utf-8 -*-
from multiprocessing import Process
from multiprocessing import Pool
from multiprocessing import Manager
from os import write
from os.path import join
import time
import random
import csv
from time import sleep
import os
import numpy as np
import FileUtils
import pandas as pd
from sklearn.metrics import classification_report


def f(x):
    x['a'] = 2
    


if __name__ == '__main__':
    # cat_list = []
    # list = os.listdir('./dic') #列出文件夹下所有的目录与文件
    # for l in list:
    #     cat_list.append(l.split('_')[0])
    # tj = {}
    # for cat in cat_list:
    #     tj[cat] = np.load('./tj/'+ cat + '.npy')
    #     print(len(tj[cat]))

    # print('hello')
    
    # with open('./temp/temp_test.csv',encoding='utf-8') as f:
    #     reader = csv.reader(f)
    #     isF = True
    #     for row in reader:
    #         if isF:
    #             isF = False
    #             continue
    #         doc = row[0]
    #         cat = row[1]
    #         with open('./test/' + cat +'.csv','a',encoding='utf-8',newline='') as f1:
    #             writer = csv.writer(f1)
    #             writer.writerow([doc])
    #             f1.close()
    #     f.close()
    
    csv.field_size_limit(500 * 1024 * 1024)
    # sports = 150000
    # news = 150000

    # fw = open('t.csv','w',encoding='utf-8',newline='')
    # writer = csv.writer(fw)
    # l = ['career','mil','cul','travel','learning']
    # with open('./data.csv',encoding='utf-8') as f:
    #     reader = csv.reader(f)
    #     for i,row in enumerate(reader):
    #         cat = row[0]
    #         if cat not in l:
    #             if cat == 'sports' and sports>0:
    #                 sports = sports - 1
    #             elif cat == 'news' and news>0:
    #                 news = news -1
    #             else:
    #                 writer.writerow(row)

       


    dic = {}
    i = 0
    with open('./t.csv',encoding='utf-8') as f:
        reader = csv.reader(f)
        isF = True
        for row in reader:
            if isF:
                isF = False
                continue
            cat = row[0]
            try:
                dic[cat] = dic[cat] + 1
            except KeyError:
                dic[cat] = 1
            i = i+1
    for k in dic.keys():
        print(k)
        print(dic[k])
    print(i)




    


    
    
    

# 结果 
#[ 1, 2]
# [1]
# 2

