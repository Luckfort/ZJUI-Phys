import os
import json
import numpy as np
from tqdm import tqdm
import sys
from math import *
from datactrl import *
from scipy.stats import rankdata
from scipy.linalg import norm

def Discrete(num):
    if (num <= 0.5):
        return 0.2
    if (num <= 0.75):
        return 0.4
    if (num <= 1):
        return 0.6
    if (num <= 2):
        return 0.8
    return 1


"""
    Config
"""
usr_num = 500
itm_num = 777
longest_time = 5400
ke_dim = 100
chap_num = 150

def getrate(rank):
    if (rank <= 0.03):
        return 5
    if (rank <= 0.08):
        return 4
    if (rank <= 0.13):
        return 3
    if (rank <= 0.18):
        return 2
    return 1

"""
    We are calculating the rates of each (U,I) pair.
    Output: uivec.npy
"""

time_rate = 0.6
match_rate = 1 - time_rate

"""
    related finish time takes up 40% of the rate.
    matching degree takes up 60% of the rate.
"""

rela_time = np.zeros([usr_num,itm_num])
match_deg = np.zeros([usr_num,itm_num])
ui_vec = np.zeros([usr_num,itm_num])

user_rec, user_time, item_label, item_name, item_stat = data_load()
uitime = np.load('./Stat/uitime.npy')
uiedge = np.load('./Stat/uiedge.npy')
user_rec = LoadDataset('./Data/user_rec.json')
item_stat = LoadDataset('./Data/item_stat.json')

# p stands for user vector, q stands for item vector
# apply max ranking in rankdata
p = rankdata(uitime, method = 'max', axis = 0)

# apply sigmoid-like function
sigm_range = e
p = 1 - 1 / (1 + e ** (-(sigm_range - p * (sigm_range * 2 / usr_num))))

for i in range(usr_num):
    p[i] /= norm(p[i])

item_label = LoadDataset('./Data/item_label.json')
q = np.zeros([itm_num,chap_num])

for i in item_label:
    for j in item_label[i]:
        q[int(i)][int(j)] = 1

for i in range(itm_num):
    q[i] /= np.sum(q[i])

chap_wei = q.copy() # chap_wei is new q

uc_rank = p.dot(q) # gain the avg rank.
uc_rank = rankdata(uc_rank, method = 'max', axis = 0)
uc_rank = 1 - 1 / (1 + e ** (-(sigm_range - uc_rank * (sigm_range * 2 / usr_num))))

"""
    Matrix Factorization
    Neural Collaborative Filtering
    uc_rank is the ranking of the user's mastery of each knowledge point. 
    After calculating the average ranking, it will be ranked again
    
    We proposed to do linear transformation between q(item_vec) and graph_vec,
    for the sake of recommend more precisely.
"""

graphvec = np.load('./Stat/graphvec.npy')
#uc_rank = uc_rank.dot(graphvec)
#chap_wei = chap_wei.dot(graphvec)

"""
    After knowledge embedding to 100 dimension, we are going to figure out the result
"""
for i in range(usr_num):
    uc_rank[i] /= norm(uc_rank[i])
for i in range(itm_num):
    chap_wei[i] /= norm(chap_wei[i]) # new p and new q

for i in tqdm(range(usr_num)):
    for j in range(itm_num):
        temp = 0
        for k in range(ke_dim):
            temp += uc_rank[i][k]*chap_wei[j][k]
        match_deg[i][j] = temp
#np.save('./Stat/match.npy',match_deg) # if need match with KE, please use matchK instead on rating
#match_deg = np.load('./Stat/match.npy')

for i in range(usr_num):
    for j in range(itm_num):
        Avg = item_stat[str(j)]['Avg']
        if (Avg > 5400):
            Avg = 5400
        uitime[i][j] /= Avg
        rela_time[i][j] = Discrete(uitime[i][j])

# np.save('./Stat/rela_time.npy',rela_time)
#rela_time = np.load('./Stat/rela_time.npy')

ui_vec = time_rate * rela_time + match_rate * match_deg
print(np.max(ui_vec))
ui_vec = ui_vec / np.max(ui_vec)
np.save('./Stat/rating.npy',ui_vec) # if need rating with KE, please use ratingK instead on rating
ui_rank = rankdata(ui_vec, method = 'min').reshape(ui_vec.shape)
for i in range(usr_num):
    for j in range(itm_num):
        ui_rank[i][j] = getrate(ui_rank[i][j]/(usr_num*itm_num)) * uiedge[i][j]

count=0
for i in range(usr_num):
    for j in range(itm_num):
        if (uiedge[i][j]==0):
            ui_vec[i][j] = 0
            continue
        if (ui_rank[i][j] >= 4):
            count+=1
            ui_vec[i][j]=1
        else:
            ui_vec[i][j]=0

print(np.sum(uiedge))
print(np.sum(ui_vec))
for i in range(usr_num):
    for j in range(itm_num):
        if (uiedge[i][j]==0):
            ui_vec[i][j] = -1

count=0
for i in range(usr_num):
    for j in range(itm_num):
        if(ui_vec[i][j]>=0):
            count+=ui_vec[i][j]
print(count)
np.save('./Stat/pn.npy',ui_vec) # if need ui_graph with KE, please use pnK instead on pn