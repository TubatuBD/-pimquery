from piq_feature import ImSim
import numpy as np
import pandas as pd
from time import time
import json

def savesim(simdict, filename):
    save_dict = json.dumps(simdict)
    with open(filename, 'w') as f:
        f.write(save_dict)

def loadsim(filename):
    with open(filename, 'r') as f:
        res = json.loads(f.readlines()[0])
    return res

print('load des data ...')
load_des = loadsim('data/piq_imdes.json')
deslist = [np.array(des, dtype=np.uint8) for des in load_des]

imsim = ImSim(k=50)

def fp2des(fp):
    kp_num = int(len(fp) / (64))
    ut8arr = np.array([int(fp[i:i+2], 16) for i in range(0, len(fp), 2)], dtype=np.uint8)
    return ut8arr.reshape(kp_num, 32)

def findAllSimOfIndex(idx):
    des1 = deslist[idx]
    sims = list()
    start = time()
    for i in range(len(deslist)):
        if idx != i:
            des2 = deslist[i]
            sim = imsim.calcSim(des1, des2)
            if sim > 0.01:
                sims.append(i)
    print('findAllSimOfIndex {} cost time: {}s'.format(idx, time() - start))
    return (idx, sims)

def querySimFrom(begin, end, limit_time=3600):
    _allsims = list()
    start = time()
    for i in range(begin, end):
        sims = findAllSimOfIndex(i)
        _allsims.append(sims)
        # if time() - start > limit_time:
        #     break
    return _allsims

def simsN(N, allsims):
    begin = allsims[-1][0] + 1
    end = begin + N

    allsims2 = querySimFrom(begin, end)

    return allsims + allsims2

if __name__ == '__main__':
    print('start gen test data ...')
    allsims = loadsim('data/piq_test_data.json')
    allsims = simsN(2000, allsims)
    savesim(allsims, 'data/piq_test_data.json')
    # allsims = querySimFrom(0, 1)
    # savesim(allsims, 'data/piq_test_data.json')