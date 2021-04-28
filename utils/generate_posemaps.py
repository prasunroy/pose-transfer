import argparse
import numpy as np
import os
import pandas as pd


def generate_posemaps(out_dir, keypoints_data, verbose=False):
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    df = pd.read_csv(keypoints_data)
    for i in range(len(df)):
        file_id = str(df.iloc[i, 0])
        img_h, img_w = np.uint32(df.iloc[i, 1:3])
        keypoints = np.int32(df.iloc[i, 3:39]).reshape(-1, 2)
        posemap = np.zeros((img_h, img_w, keypoints.shape[0]), dtype=np.uint8)
        for k in range(keypoints.shape[0]):
            col, row = keypoints[k]
            if row == -1 or col == -1:
                continue
            posemap[row, col, k] = 1
        fp = f'{out_dir}/{file_id}.npz'
        np.savez_compressed(fp, posemap)
        if verbose:
            print(f'\r[INFO] Generating posemaps... {(i+1)*100.0/len(df):3.0f}%', end='')
    if verbose:
        print('')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', required=True, help='output directory')
    ap.add_argument('-k', required=True, help='keypoints data')
    args = vars(ap.parse_args())
    
    generate_posemaps(args['o'], args['k'], verbose=True)
