# -*- coding: utf-8 -*-
from multiprocessing import Process
from multiprocessing import Pool
import time
import random
from time import sleep

def f(x):
    i = random.randint(1,5)
    sleep(i)
    return print(x)


if __name__ == '__main__':
    # res_list=[]
    # pool = Pool(5)
    # for i in range(10):
    #     res = pool.apply_async(func=f, args=(i,))
    #     res_list.append(res)

    # pool.close()
    # pool.join()

    # print(res_list)

    for i in range(10):
        print(i)
        
