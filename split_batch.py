import pandas as pd
import argparse

def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', type=int, default=10000, 
                        help='batch size')

    parser.add_argument('--file', type=str, default='case_struct', 
                        help='csv filename')

    return parser.parse_args()


class Spliter:
    def __init__(self, csv_filename='case_struct'):
        print('split {}.csv:\n'.format(csv_filename))
        origin_dataset = pd.read_csv('data/deploy/{}.csv'.format(csv_filename))
        self.df = origin_dataset[origin_dataset.filename != '\\N'].sort_values(by=['cid', 'puttime'])
        self.filename = csv_filename
    def run(self, start, size):
        index = self.df.index[start: start+size]
        self.df.loc[index].to_csv('data/deploy/{}_{}-{}.csv'.format(self.filename, start, start+size-1), index=False)
    def run_batch(self, batch_size=10000):
        size = self.df.shape[0]
        batch_num = size // batch_size
        left_num = size % batch_size
        for i in range(batch_num):
            _start = batch_size * i
            _size = batch_size
            self.run(_start, _size)
        if left_num > 0:
            _start = batch_size * batch_num
            _size = left_num
            self.run(_start, _size)

if __name__ == '__main__':
    in_arg = get_input_args()
    spliter = Spliter(csv_filename=in_arg.file)
    spliter.run_batch(batch_size=in_arg.batch)