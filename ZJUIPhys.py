"""
    This dataset is created using data mining skills using the raw data on SmartPhysics, ZJUI section.
    We created the dataset after getting the consent of the Professors in ZJUI.
    The dataset doesn't have private information of any student, so it is used legitimately.
    The last step of creating the dataset.
    Author: SRTPX0522
    Date: Dec 30, 2022
"""
import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import sys
from math import floor,log,sqrt
import random

"""
    Config
"""
usr_num = 500
itm_num = 777

info = np.load('./Stat/information.npy')

User_Info = np.load('./Stat/usrclaim.npy')
Item_Info = np.load('./Stat/iteminfo.npy')
UI_Graph = np.load('./Stat/pn.npy') # if need ui_graph with KE, please use pnK instead on pn

mark = np.zeros([500,777])
train_list = []
valid_list = []
test_list = []

count = 0
for i in range(usr_num):
    for j in range(itm_num):
        if UI_Graph[i][j] == -1:
            continue
        count += UI_Graph[i][j]
        rand=random.randint(0, 9)
        if (rand == 8):
            valid_list.append((i,j))
        elif (rand == 9):
            test_list.append((i,j))
        else:
            train_list.append((i,j))
print(count)

random.shuffle(train_list)
random.shuffle(valid_list)
random.shuffle(test_list)

Data_order = []
Data_order.extend(train_list)
Data_order.extend(valid_list)
Data_order.extend(test_list)

"""
    We have .txt version and .csv version of the ZJUIPhys dataset
"""
name_list = ['label','I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13',
             'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13',
             'C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27']

cnt = 0
with open('ZJUIPhys.csv', 'w', encoding = 'utf-8') as f:
    f.write(','.join(name_list)+'\n')
    for i, j in Data_order:
        cnt += UI_Graph[i][j]
        data_list = [str(UI_Graph[i][j])]
        l1 = np.around(Item_Info[j], 3).tolist()
        l2 = User_Info[i].tolist()
        l1 = [str(i) for i in l1]
        #l2 = [str(i) for i in l2]
        l2 = [str(i) if i!=0 else '0' for i in l2]
        data_list.extend(l1)
        data_list.extend(l2)
        f.write(','.join(data_list)+'\n')
    f.close()
print(cnt)
    
with open('ZJUIPhys.txt', 'w', encoding = 'utf-8') as f:
    for i, j in Data_order:
        data_list = [str(UI_Graph[i][j])]
        l1 = np.around(Item_Info[j], 3).tolist()
        l2 = User_Info[i].tolist()
        l1 = [str(i) for i in l1]
        #l2 = [str(i) for i in l2]
        l2 = [str(i) if i!=0 else '0' for i in l2]
        data_list.extend(l1)
        data_list.extend(l2)
        f.write('\t'.join(data_list)+'\n')
    f.close()
