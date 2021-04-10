import argparse
import csv
import cv2
import glob
import numpy as np
import os
import pandas as pd
from openpose.body.estimator import BodyPoseEstimator


class KeypointEstimator(object):
    
    def __init__(self):
        self.estimator = BodyPoseEstimator(pretrained=True)
    
    def _write_csv(self, path, mode, data):
        with open(path, mode, newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(data)
    
    def _get_column_labels(self, num_keypoints=18):
        column_labels = ['file_id', 'img_h', 'img_w']
        for i in range(num_keypoints):
            column_labels.extend([f'p{i}_x', f'p{i}_y'])
        return column_labels
    
    def estimate_keypoints(self, fp, img_dir, images=None, verbose=False):
        out_dir = os.path.dirname(fp)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        self._write_csv(fp, 'w', self._get_column_labels(18))
        if images is None:
            images = sorted(glob.glob(f'{img_dir}/**/*.*', recursive=True))
        for i, image in enumerate(images):
            x = cv2.imread(image)
            if x is None:
                continue
            keypoints = self.estimator(x)
            keypoints = keypoints[0] if len(keypoints) > 0 else np.zeros((18, 3), dtype=np.int32)
            keypoints[np.where(keypoints[:, 2]==0), :2] = -1
            keypoints = keypoints[:, :2].reshape(-1).tolist()
            file_id = os.path.normpath(image)[len(os.path.normpath(img_dir)):]
            file_id = os.path.splitext(file_id)[0].replace('/', '').replace('\\', '')
            self._write_csv(fp, 'a', [file_id, x.shape[0], x.shape[1]] + keypoints)
            if verbose:
                print(f'\r[INFO] Estimating keypoints... {(i+1)*100.0/len(images):3.0f}%', end='')
        if verbose:
            print('\r[INFO] Estimating keypoints... 100%')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', required=True, help='output file')
    ap.add_argument('-i', required=True, help='image directory')
    ap.add_argument('-f', required=False, help='image files')
    args = vars(ap.parse_args())
    
    if args['f']:
        images = sorted(pd.read_csv(args['f'])['img'])
        images = [os.path.join(args['i'], fp) for fp in images]
    else:
        images = None
    
    estimator = KeypointEstimator()
    estimator.estimate_keypoints(args['o'], args['i'], images, verbose=True)
