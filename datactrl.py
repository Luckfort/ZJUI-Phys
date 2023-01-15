import os
import json
import numpy as np
from tqdm import tqdm
import sys
from math import floor,log,sqrt

"""
    Config
"""
usr_num = 500
itm_num = 777
longest_time = 5400

def SaveDataset(filename, dataset):
    dict_json = json.dumps(dataset)
    with open(filename,'w+') as f:
        f.write(dict_json)
        f.close()

def LoadDataset(filename):
    with open(filename,'r+') as f:
        read_dict = f.read()
        f.close()
    read_dict = json.loads(read_dict)
    return read_dict

"""
    get the timestamp of user's last behavior
"""
def get_last_time(C, l1, cate):
    l2 = -1
    for i in C:
        if (i[0] < l1 and i[1] != cate):
            l2 = i[0]
        if (i[0] >= l1):
            break
    return l2

"""
    data_load()
        - load all the necessary datasets
"""
def data_load():
    user_rec = LoadDataset('./Data/user_rec.json')
    user_time = LoadDataset('./Data/user_time.json')
    item_label = LoadDataset('./Data/item_label.json')
    item_name = LoadDataset('./Data/item_name.json')
    item_stat = LoadDataset('./Data/item_stat.json')
    return user_rec, user_time, item_label, item_name, item_stat

if __name__=='__main__':
    user_rec, user_time, item_label, item_name, item_stat = data_load()
    ui_time = np.zeros([usr_num,itm_num])
    ui_edge = np.zeros([usr_num,itm_num]) # is the edge (U,I) existed or not
    edge_num = 0
    
    for i in tqdm(range(usr_num)):
        I = str(i)
        prob_itv = {}
        last = ''
        for j in range(itm_num):
            J = str(j)
            ui_time[i][j] = longest_time
            que_tot = user_rec[J]['Que']
            # 1. The interval between first submit time t1 and last sumbit 
            #    time t2 of one user
            itv_r = 0
            itv_l = 2147483647.0
            start = 0
            if ('0' not in user_rec[J].keys()):
                start = 1
            for k in range(start,que_tot+1):
                K = str(k)
                if K not in user_rec[J].keys():
                    continue
                if I not in user_rec[J][K].keys():
                    continue
                user_dict = user_rec[J][K][I]
                itv_l = min(itv_l, min(user_dict['Time']))
                itv_r = max(itv_l, max(user_dict['Time']))
            itv = itv_r - itv_l
            if (itv < 0):
                continue
            ui_edge[i][j] = 1
            edge_num += 1
            prob_itv[J] = [itv_l, itv_r]
            # 2. Calculate the interval of t1 and the last submit time t3
            #    of one user
            itv_3 = get_last_time(user_time[I], itv_l, J)
            if (itv_3 == -1):
                if (user_rec[J]['Que'] == 0):
                    sub_time = 600 # Assume finishs a big problem if t3 is not applicable.
                else:
                    sub_time = 300 # Assume finishs a small blank if t3 is not applicable.
                itv = itv + sub_time
            else:
                itv = itv + itv_l - itv_3
            if (itv < ui_time[i][j]): # At most 90 minutes per problemset
                ui_time[i][j] = itv
    
    info=np.array([usr_num,itm_num,edge_num])
    
    np.save("./Stat/information.npy", info)
    np.save("./Stat/uitime.npy", ui_time)
    np.save("./Stat/uiedge.npy", ui_edge)