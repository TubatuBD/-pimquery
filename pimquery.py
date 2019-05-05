import numpy as np
import pandas as pd
from int16hash import int16hash, search_hash
from imgfeature import ImSim
from time import time

def fp2des(fp):
    kp_num = int(len(fp) / (64))
    ut8arr = np.array([int(fp[i:i+2], 16) for i in range(0, len(fp), 2)], dtype=np.uint8)
    return ut8arr.reshape(kp_num, 32)

class PImQuery:
    def __init__(self, short_csv, long_csv):
        self.imsim = ImSim(k=50)
        self.short_df = pd.read_csv(short_csv)
        self.long_df = pd.read_csv(long_csv)
    def query(self, img_path):
        imsim = self.imsim
        short_df = self.short_df
        long_df = self.long_df

        short_hash = int16hash(img_path)[1]
        query_df = long_df.loc[search_hash(short_df, short_hash).index]

        des = imsim.getFeature(img_path)[1]
        query_res = list()

        start = time()
        for fp, i in zip(query_df['fp_long'], query_df.index):
            sim = imsim.calcSim(fp2des(fp), des)
            if sim > 0.0001:
                query_res.append((i, sim))
        # print('query cost time: {} s'.format(time() - start))

        query_res = sorted(query_res, key=lambda x:x[1], reverse=True)
        return [(query_df.loc[item[0]].path, item[1]) for item in query_res]
    def querySimIdxesByIdx(self, idx):
        query_res = self.queryByIdx(idx)
        return [s[0] for s in query_res if s[0] != idx]
    def queryByIdx(self, idx):
        imsim = self.imsim
        short_df = self.short_df
        long_df = self.long_df

        short_hash = short_df.loc[idx]['hash_short']
        query_df = long_df.loc[search_hash(short_df, short_hash).index]

        des = fp2des(long_df.loc[idx]['fp_long'])
        query_res = list()

        start = time()
        for fp, i in zip(query_df['fp_long'], query_df.index):
            sim = imsim.calcSim(fp2des(fp), des)
            if sim > 0.0001:
                query_res.append((i, sim))
        # print('query cost time: {} s'.format(time() - start))

        query_res = sorted(query_res, key=lambda x:x[1], reverse=True)
        return query_res