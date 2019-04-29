from time import time
from imgfeature import ResizeImFeature
import pandas as pd

def imhash(gray):
    dt = gray.flatten()
    xlen = len(dt)
    avg = dt.mean()
    avg_list = ['0' if i < avg else '1' for i in dt]
    return ''.join(['%x' % int(''.join(avg_list[x: x+4]), 2) for x in range(0, xlen, 4)])

def hamming(hash1, hash2):
    len1 = len(hash1)
    len2 = len(hash2)
    hlen = 0
    if len1 < len2:
        hlen = len1
    else:
        hlen = len2
    if hlen == 0:
        return -1
    distance = 0
    for i in range(hlen):
        num1 = int(hash1[i], 16)
        num2 = int(hash2[i], 16)
        xor_num = num1 ^ num2
        for i in [1, 2, 4, 8]:
            if i & xor_num != 0:
                distance += 1
    return distance

def genhash(df, size):
    imf_resize = ResizeImFeature(k=50)
    img_paths = df['path']
    fp_hashes = list()

    start = time()

    for img_path in img_paths:
        bgr, gray = imf_resize.read(img_path, size)
        fp_hash = imhash(gray)
        fp_hashes.append(fp_hash)

    print('cost time: {}s'.format(time() - start))
    return fp_hashes

imf_resize = ResizeImFeature(k=50)

def int16hash(img_path):
    bgr, gray = imf_resize.read(img_path, 4)
    fp_hash = imhash(gray)
    return int(fp_hash, 16), fp_hash

def search_hash_by_hamming(df, fp_hash):
    hash_shorts = df['hash_short']
    distances = list()
    start = time()
    for shash in hash_shorts:
        distance = hamming(shash, fp_hash)
        distances.append(distance)
    print('search_hash_by_hamming cost time: {} s'.format(time() - start))

    dis_df = pd.DataFrame({ 'distance': distances }, index=df.index)

    return df.loc[dis_df[dis_df.distance == 0].index]

def search_hash(df, fp_hash):
    hash_shorts = df['hash_short']
    hash_indexes = list()
    start = time()
    for i, shash in enumerate(hash_shorts):
        if shash == fp_hash:
            hash_indexes.append(i)
    print('search_hash cost time: {} s'.format(time() - start))

    return df.loc[hash_indexes]
