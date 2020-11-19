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
    tf = {}
    idf = {}
    for cat in cat_list:
        tf_idf[cat] = np.load('./tf-idf/'+ cat + '.npy')
    for cat in cat_list:
        tf[cat] = np.load('./tf/'+ cat + '.npy')
    for cat in cat_list:
        idf[cat] = np.load('./idf/'+ cat + '.npy')

    
    for cat in cat_list:
        tf_idf[cat] = np.log(tf[cat]/tf[cat].sum())
    
    print(tf_idf)

    
    
    # with open('words_list') as f:
    #     s = f.read()
    #     ab = s.split(',')
    #     print(ab)
        
