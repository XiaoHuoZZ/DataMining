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
    dic = {'a':3}
    f(dic)

    print(dic)
    

    
    
    

# 结果 
#[ 1, 2]
# [1]
# 2

