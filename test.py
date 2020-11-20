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


def f(x):
    i = random.randint(1,5)
    sleep(i)
    return print(x)


if __name__ == '__main__':
    cat_list = []
    list = os.listdir('./dic') #列出文件夹下所有的目录与文件
    for l in list:
        cat_list.append(l.split('_')[0])
    tf_idf = {}
    for cat in cat_list:
        tf_idf[cat] = np.load('./tf-idf/'+ cat + '.npy')
    print(tf_idf['2008'])
   

    
    
    # a_dict = {'s':'007', 'b': '003','w':'003','z':'00'}
    b = {}
    # print(list(a_dict.keys())) # key 列表
    # print(list(a_dict.keys())[list(a_dict.values()).index('007')]) # 对应的索引值
    # print(list(a_dict.keys())[list(a_dict.values()).index('002')])


    

# 结果 
#[ 1, 2]
# [1]
# 2

