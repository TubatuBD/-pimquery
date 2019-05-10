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

def download_from_df(df, start=0, size=10000, timeout=1, bgr=False, gray_func=None, proc_func=None):
    index = df.index[start: start+size]
    res = list()
    for i in index:
        cid, tid, filename = df.loc[i, ['cid', 'tid', 'filename']]
        try:
            img_data = download_img_by_filename(filename,
                                        timeout=timeout,
                                        bgr=bgr,
                                        gray_func=gray_func,
                                        proc_func=proc_func)
            if img_data is None:
                continue
            res.append([i, str(cid), str(tid), img_data])
        except:
            continue
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
    saveJson(res, 'data/case_struct_img_hash_fp.json')
    print('cost time {}s'.format(time() - start_time))

    print('completed!')

if __name__ == '__main__':
    in_arg = get_input_args()
    gen_batch(in_arg.start, in_arg.size)