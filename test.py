# -*- coding: utf-8 -*-
from multiprocessing import Process
from multiprocessing import Pool
from multiprocessing import Manager
from os.path import join
import time
import random
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
    manager = Manager()
    dic = manager.dict()
    s = 'h'
    dic[s] = 'hello'
    print(dic)
        
