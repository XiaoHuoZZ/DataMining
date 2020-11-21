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
    i = random.randint(1,5)
    sleep(i)
    return print(x)


if __name__ == '__main__':
    # cat_list = []
    # list = os.listdir('./dic') #列出文件夹下所有的目录与文件
    # for l in list:
    #     cat_list.append(l.split('_')[0])
    # tj = {}
    # for cat in cat_list:
    #     tj[cat] = np.load('./tj/'+ cat + '.npy')
    # print(len(tj[cat]))
   

    
    
   df = pd.read_csv('./temp/temp_test.csv')
   d = df.iloc[0:20]
   for i,row in d.iterrows():
       print(row['category'])
    

# 结果 
#[ 1, 2]
# [1]
# 2

