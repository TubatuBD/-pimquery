from piq_feature import ResizeImFeature
from piq_hash import imhash_dct
from time import time
import pandas as pd

imf_resize = ResizeImFeature(k=50)
dataset = pd.read_csv('dataset.csv')

def genhash(df, k):
    size = 2 * k
    img_paths = df['path']
    fp_hashes = list()

    start = time()

    for img_path in img_paths:
        bgr, gray = imf_resize.read(img_path, size)
        fp_hash = imhash_dct(gray)
        fp_hashes.append(fp_hash)

    print('cost time: {}s'.format(time() - start))
    return fp_hashes

def genhash_batch(batches):
    for k in batches:
        print('gen hash k = {}'.format(k))

        fp_hashes = genhash(dataset, k)

        dataset['hash_k{}'.format(k)] = fp_hashes

        dataset[['path', 'hash_k{}'.format(k)]].to_csv('data/piq_imhash_k{}.csv'.format(k), index=False)

if __name__ == '__main__':
    genhash_batch([2])