import numpy as np
import pandas as pd
from piq_hash import PIQHash
from piq_feature import ImSim
from time import time

def fp2des(fp):
    kp_num = int(len(fp) / (64))
    ut8arr = np.array([int(fp[i:i+2], 16) for i in range(0, len(fp), 2)], dtype=np.uint8)
    return ut8arr.reshape(kp_num, 32)

def sims2repeats(sims):
    firsts = list()
    repeats = list()
    res = list()
    for sim in sims:
        first = sim[0]
        if first not in repeats:
            for repeat in sim[1]:
                if repeat not in repeats:
                    repeats.append(repeat)
                    res.append((repeat, first))
            if first not in firsts:
                firsts.append(first)
    return res

class PImQuery:
    def __init__(self, hash_k=2, sim_threshold=0.16, df_hash=None, df_fp=None):
        self.hash_k = hash_k
        self.sim_threshold = sim_threshold
        self.imsim = ImSim(k=50)

        if df_hash is None:
            self.df_hash = pd.read_csv('data/piq_imhash_k{}.csv'.format(hash_k))
        else:
            self.df_hash = df_hash
        if df_fp is None:
            self.df_fp = pd.read_csv('data/piq_imfp.csv')
        else:
            self.df_fp = df_fp
        self.piqhash = PIQHash(self.df_hash)
    def findSims(self, im, by_hamming=False):
        imsim = self.imsim
        piqhash = self.piqhash
        hash_k = self.hash_k
        sim_threshold = self.sim_threshold

        # 获取具有相同hash值的数据 以在一个较小范围内通过图片指纹匹配相似的图片
        # 可能会因为hash不能正确将所有重复的图片划分在一个集合，遗漏一些重复图片，最终导致召回率降低
        imhash = piqhash.getHash(im, hash_k)
        if by_hamming == True:
            query_idxes = piqhash.queryHamming(imhash, hash_k)
        else:
            query_idxes = piqhash.query(imhash, hash_k)
        query_df = self.df_fp.loc[query_idxes]

        # 图片指纹描述符
        imdes = imsim.getFeature(im)[1]

        query_res = list()
        for fp, i in zip(query_df['fp_long'], query_df.index):
            sim = imsim.calcSim(fp2des(fp), imdes)
            if sim > sim_threshold:
                query_res.append((i, sim))

        # 按相似度大小返回经过排序的相似图片索引数组
        return sorted(query_res, key=lambda x:x[1], reverse=True)
    def findSimsByIdx(self, idx, by_hamming=False):
        imsim = self.imsim
        piqhash = self.piqhash
        hash_k = self.hash_k
        sim_threshold = self.sim_threshold

        # 获取具有相同hash值的数据 以在一个较小范围内通过图片指纹匹配相似的图片
        # 可能会因为hash不能正确将所有重复的图片划分在一个集合，遗漏一些重复图片，最终导致召回率降低
        imhash = self.df_hash.loc[idx]['hash_k' + str(hash_k)]
        if by_hamming == True:
            query_idxes = piqhash.queryHamming(imhash, hash_k)
        else:
            query_idxes = piqhash.query(imhash, hash_k)
        query_df = self.df_fp.loc[query_idxes]

        # 图片指纹描述符
        imdes = fp2des(self.df_fp.loc[idx]['fp_long'])

        query_res = list()
        for fp, i in zip(query_df['fp_long'], query_df.index):
            sim = imsim.calcSim(fp2des(fp), imdes)
            if sim > sim_threshold:
                query_res.append((i, sim))

        # 按相似度大小返回经过排序的相似图片索引数组
        return sorted(query_res, key=lambda x:x[1], reverse=True)
    def query(self, im, by_hamming=False):
        # start = time()
        query_res = self.findSims(im, by_hamming)
        # print('query cost time: {} s'.format(time() - start))
        return [item[0] for item in query_res]
    def queryByIdx(self, idx, by_hamming=False):
        query_res = self.findSimsByIdx(idx, by_hamming)

        return [item[0] for item in query_res if item[0] != idx]
    def findRepeats(self, max=None, by_hamming=False):
        num = self.df_fp.shape[0]
        if (max is not None) and max < num:
            num = max
        sims = list()
        for i in range(num):
            idxs = self.queryByIdx(i, by_hamming=by_hamming)
            if len(idxs) > 0:
                sims.append((i, idxs))
        return sims2repeats(sims)
    def predict_test(self, train_set, test_set):
        imsim = self.imsim
        sim_threshold = self.sim_threshold

        train_fp = self.df_fp.loc[train_set]
        train_hash = self.df_hash.loc[train_set]

        predict_res = list()
        for test_index in test_set:
            test_des = fp2des(self.df_fp.loc[test_index, 'fp_long'])
            test_hash = self.df_hash.loc[test_index, 'hash_k2']
            query_fp = train_fp.loc[train_hash[train_hash.hash_k2 == test_hash].index]

            repeat = 0
            simto = 0
            sim = 0
            for fp, i in zip(query_fp['fp_long'], query_fp.index):
                sim = imsim.calcSim(fp2des(fp), test_des)
                if sim > sim_threshold:
                    repeat = 1
                    simto = i
                    break
            predict_res.append((repeat, simto, sim))
        return list(zip(*predict_res))
