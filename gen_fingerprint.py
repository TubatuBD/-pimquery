import pandas as pd
from time import time
from imgfeature import ImFeature, CropImFeature
import argparse
import gc

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0, 
                        help='start index of data')

    return parser.parse_args()

def genlongfp(start=0):
    df = pd.read_csv('dataset.csv')
    # imf = ImFeature(k=50)
    imf = CropImFeature(k=50)

    fingerprints = list()

    time_count = 100
    time_tracker = list()

    batch_num = 0
    batch_size = 10000

    img_paths = df['path']
    last_idx = len(img_paths) - 1

    for i, img_path in enumerate(img_paths):
        if i > 0 and i % batch_size == 0:
            batch_num += 1
        if i < start:
            continue
        if i == start :
            start_time = time()
        elif i % time_count == 0:
            time_tracker.append((i, time() - start_time))
            print(*time_tracker[-1])
            start_time = time()

        if i > start and i % batch_size == 0:
            pd.DataFrame(dict(fp_long=fingerprints), index=range((batch_num-1)*batch_size, batch_num*batch_size)).to_csv('fingerprint_{}-{}.csv'.format((batch_num-1)*batch_size,  batch_num*batch_size))
            del locals()['fingerprints']
            gc.collect()
            print('Garbage Collect!', ','.join(list(locals().keys())))
            fingerprints = list()
            print('save batch to csv, cost {} s/10000pcs!'.format(pd.DataFrame(time_tracker, columns=['index', 'cost'])['cost'].sum()))
            # break
        try:
            img, kps = imf.keypoint(img_path)
            descriptor = imf.descriptor(img, kps)[1]
            fingerprint = imf.fingerprint(descriptor)
        except:
            print('excend memory size!')
            break

        if i == last_idx:
            time_tracker.append((i+1, time() - start_time))
            print(*time_tracker[-1])
            fingerprints.append(fingerprint)
            pd.DataFrame(dict(fp_long=fingerprints), index=range(batch_num*batch_size, batch_num*batch_size+((i+1)%batch_size))).to_csv('fingerprint_{}-{}.csv'.format(batch_num*batch_size, batch_num*batch_size+((i+1)%batch_size)))
            print('complete!')
            break

        fingerprints.append(fingerprint)

if __name__ == "__main__":
    in_arg = get_input_args()
    gc.enable()
    genlongfp(in_arg.start)
