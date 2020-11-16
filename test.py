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

def f(x):
    i = random.randint(1,5)
    sleep(i)
    return print(x)


if __name__ == '__main__':
    # m = 3
    # n = 2
    # dp = [[0 for i in range(n)] for j in range(m)]
    # print(dp)
    
    with open('news.sohunews.010806.txt','r',encoding='utf-8') as f:
        s = f.read()
        ns = s.replace('&','&amp')
        res = '<docs>\n' + ns + '</docs>'
        f.close
    with open('news.sohunews.010806.txt','w',encoding='utf-8') as f:
        f.write(res)
        f.close
    # with open('words_list') as f:
    #     s = f.read()
    #     ab = s.split(',')
    #     print(ab)
        
