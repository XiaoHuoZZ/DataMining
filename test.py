# -*- coding: utf-8 -*-
from multiprocessing import Process
from multiprocessing import Pool
from os.path import join
import time
import random
from time import sleep

def f(x):
    i = random.randint(1,5)
    sleep(i)
    return print(x)


if __name__ == '__main__':
    data = ['a,b,c','1,2,3',',12,']
    with open('words_list','w') as f:
        f.writelines(','.join(data))
        f.close
    # with open('words_list') as f:
    #     s = f.read()
    #     ab = s.split(',')
    #     print(ab)
        
