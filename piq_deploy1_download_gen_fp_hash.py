import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import cv2 as cv
import numpy as np
from piq_feature import ImFeature, CropImFeature, ResizeImFeature, ImSim
from piq_hash import imhash_dct
from helper import saveJson, loadJson
from time import time
import argparse

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0, 
                        help='start index of data')
    parser.add_argument('--size', type=int, default=10000, 
                        help='num of data')
    parser.add_argument('--batch', type=int, default=10000, 
                        help='batch size')

    parser.add_argument('--id', type=str, default='tid', 
                        help='batch size')

    parser.add_argument('--file', type=str, default='case_struct', 
                        help='batch size')

    return parser.parse_args()

def download_img_by_filename(filename, timeout=3, bgr=False, gray_func=None, proc_func=None):
    img_url = 'https://pic.to8to.com/case/{}'.format(filename)
    try:
        res = requests.get(img_url, timeout=timeout)
    except:
        return None
    img_ndarray = np.asarray(Image.open(BytesIO(res.content)))
    if bgr:
        img_ndarray = img_ndarray[:, :, [2, 1, 0]]
    if gray_func is not None:
        img_ndarray = gray_func(img_ndarray)
    if proc_func is not None:
        if isinstance(proc_func, list):
            img_ndarray = [proc(img_ndarray) for proc in proc_func]
        else:
            img_ndarray = proc_func(img_ndarray)
    return img_ndarray

def download_from_df(df, start=0, size=10000, timeout=1, bgr=False, gray_func=None, proc_func=None, id_name='tid', failed_filename=''):
    index = df.index[start: start+size]
    res = list()
    failed_down = list()
    for i in index:
        cid, ID, filename = df.loc[i, ['cid', id_name, 'filename']]
        try:
            img_data = download_img_by_filename(filename,
                                        timeout=timeout,
                                        bgr=bgr,
                                        gray_func=gray_func,
                                        proc_func=proc_func)
            if img_data is None:
                failed_down.append([i, str(cid), str(ID), filename])
                continue
            res.append([i, str(cid), str(ID), img_data])
        except:
            failed_down.append([i, str(cid), str(ID), filename])
            continue
    if len(failed_down) > 0:
        saveJson(failed_down, 'data/{}_{}-{}.json'.format(failed_filename + '_down_failed', start, start+size-1))
    return res

def gen_batch(start, size):
    print('start {} to {}:\n'.format(start, start + size - 1))

    # imsim = ImSim(k=50, crop=False)
    imf = ImFeature(k=50)
    imf_crop = CropImFeature(k=50)
    imf_resize = ResizeImFeature(k=50)
    bgr2gray = lambda x: cv.cvtColor(x, cv.COLOR_BGR2GRAY)
    gen_fp = lambda im: imf.fingerprint(imf.feature(im)[1])
    gen_hash = lambda im: imhash_dct(imf_resize.resize(imf_crop.crop(im), size=4))

    print('loading csv from "data/deploy/case_struct.csv" ...')
    start_time = time()
    case_struct = pd.read_csv('data/deploy/case_struct.csv')
    print('cost time {}s'.format(time() - start_time))

    print('filtering empty filename data ...')
    df = case_struct[case_struct.filename != '\\N'].sort_values(by=['cid', 'puttime'])
    print('cost time {}s'.format(time() - start_time))

    print('downloading images ...')
    proc_func = [gen_hash, gen_fp]
    res = download_from_df(df, start=start, size=size, bgr=True, gray_func=bgr2gray, proc_func=proc_func)
    print('cost time {}s'.format(time() - start_time))

    print('save images to json ...')
    saveJson(res, 'data/case_struct_img_hash_fp_{}-{}.json'.format(start, start+size-1))
    print('cost time {}s'.format(time() - start_time))

    print('completed!')

class ImgDownLoader:
    def __init__(self, id_name='tid', csv_filename='case_struct'):
        print('init loader for {}:\n'.format(csv_filename))
        self.imf = ImFeature(k=50)
        self.imf_crop = CropImFeature(k=50)
        self.imf_resize = ResizeImFeature(k=50)
        self.bgr2gray = lambda x: cv.cvtColor(x, cv.COLOR_BGR2GRAY)
        self.gen_fp = lambda im: self.imf.fingerprint(self.imf.feature(im)[1])
        self.gen_hash = lambda im: imhash_dct(self.imf_resize.resize(self.imf_crop.crop(im), size=4))
        origin_dataset = pd.read_csv('data/deploy/{}.csv'.format(csv_filename))
        self.df = origin_dataset[origin_dataset.filename != '\\N'].sort_values(by=['cid', 'puttime'])
        self.id_name = id_name
        self.filename = csv_filename
    def run(self, start, size):
        proc_func = [self.gen_hash, self.gen_fp]
        print('download from {} to {} ...'.format(start, start + size - 1))
        start_time = time()
        res = download_from_df(self.df, start=start, size=size, bgr=True, gray_func=self.bgr2gray, proc_func=proc_func, id_name=self.id_name, failed_filename=self.filename)
        print('download cost time {} h.'.format((time() - start_time)/3600))
        res_json = 'data/{}_img_hash_fp_{}-{}.json'.format(self.filename, start, start+size-1)
        print('save to {}.'.format(res_json))
        saveJson(res, res_json)
        print('done.')
    def run_batch(self, start=0, size=100000, batch_size=10000):
        batch_num = size // batch_size
        left_num = size % batch_size
        for i in range(batch_num):
            _start = start + batch_size * i
            _size = batch_size
            self.run(_start, _size)
        if left_num > 0:
            _start = start + batch_size * batch_num
            _size = left_num
            self.run(_start, _size)

if __name__ == '__main__':
    in_arg = get_input_args()
    loader = ImgDownLoader(id_name=in_arg.id, csv_filename=in_arg.file)
    loader.run_batch(start=in_arg.start, size=in_arg.size, batch_size=in_arg.batch)