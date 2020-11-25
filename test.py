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
    csv.field_size_limit(500 * 1024 * 1024)
    # cat_list = []
    # list = os.listdir('./dic') #列出文件夹下所有的目录与文件
    # for l in list:
    #     cat_list.append(l.split('_')[0])
    # tj = {}
    # for cat in cat_list:
    #     tj[cat] = np.load('./tj/'+ cat + '.npy')
    #     print(len(tj[cat]))

    # print('hello')
    
    # df = pd.read_csv('./data/train.csv')
    # print(np.unique(df['category']))

    news = 100000
    sports = 70000
    business = 100000

    fw = open('t.csv','w',encoding='utf-8',newline='')
    writer = csv.writer(fw)
    l = ['career','mil','cul','travel','learning']
    with open('./data/data.csv',encoding='utf-8') as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            cat = row[0]
            if cat not in l:
                if cat == 'sports' and sports>0:
                    sports = sports - 1
                elif cat == 'news' and news>0:
                    news = news -1
                elif cat == 'business' and business>0:
                    business = business -1
                else:
                    writer.writerow(row)

       


    # dic = {}
    # i = 0
    # with open('./data/data.csv',encoding='utf-8') as f:
    #     reader = csv.reader(f)
    #     isF = True
    #     for row in reader:
    #         if isF:
    #             isF = False
    #             continue
    #         cat = row[0]
    #         try:
    #             dic[cat] = dic[cat] + 1
    #         except KeyError:
    #             dic[cat] = 1
    #         i = i+1
    # for k in dic.keys():
    #     print(k)
    #     print(dic[k])
    # print(i)


