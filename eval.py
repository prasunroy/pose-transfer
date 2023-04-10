# imports
import csv
import glob
import numpy as np
import os
import pandas as pd
from metrics.ssim_score import SSIMScore
from metrics.inception_score import InceptionScore
from metrics.ssd_score import SSDScore
from metrics.pckh_score import PCKhScore
from metrics.lpips_score import LPIPSScore
from utils.estimate_keypoints import KeypointEstimator


# configurations
# -----------------------------------------------------------------------------
dataset_name = 'DeepFashion'

run_id = 'pretrained'
ckpt_ids = [260500]

test_dir = f'../output/{dataset_name}/test/{run_id}'
save_dir = f'../output/{dataset_name}/eval/{run_id}'

real_A_images = sorted(glob.glob(f'{test_dir}/real_images/real_A/*.jpg'))
real_B_images = sorted(glob.glob(f'{test_dir}/real_images/real_B/*.jpg'))

fp_real_A_keypoints = f'{test_dir}/real_images/real_A_keypoints.csv'
fp_real_B_keypoints = f'{test_dir}/real_images/real_B_keypoints.csv'

db_keypoints = f'../datasets/{dataset_name}/test_img_keypoints.csv'
# -----------------------------------------------------------------------------


# create a csv file
def create_csv_file(path, data):
    with open(path, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(data)


# update a csv file
def update_csv_file(path, data):
    with open(path, 'a', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(data)


# retrieve keypoints from database
def retrieve_keypoints(db, images):
    keypoints = []
    database = pd.read_csv(db)
    for i, image in enumerate(images):
        file_id = os.path.splitext(os.path.basename(image))[0]
        if '@' in file_id:
            file_id = file_id[file_id.index('@') + 1:]
        kp = database.query('file_id==@file_id').values[0, 3:39]
        keypoints.append(kp)
        print(f'\r[INFO] Retrieving keypoints... {(i+1)*100.0/len(images):3.0f}%', end='')
    print('')
    return np.int32(keypoints)


# get keypoints
def get_keypoints(fp):
    if not os.path.isfile(fp):
        print(f'[INFO] Keypoints data not found at {fp}')
        image_dir = os.path.splitext(fp)[0].replace('_keypoints', '')
        estimator = KeypointEstimator()
        estimator.estimate_keypoints(fp, image_dir, verbose=True)
    return np.int32(pd.read_csv(fp).values[:, 3:39])


# create files for saving evaluation scores
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
for score in ['ssim_score', 'inception_score', 'ssd_score', 'pckh_score', 'lpips_score']:
    if score == 'lpips_score':
        columns = [
            'iter',
            'real_A', 'fake_A_alx', 'fake_A_vgg', 'fake_A_sqz',
            'real_B', 'fake_B_alx', 'fake_B_vgg', 'fake_B_sqz',
            'score_real', 'score_fake_alx', 'score_fake_vgg', 'score_fake_sqz'
        ]
    else:
        columns = [
            'iter',
            'real_A', 'fake_A',
            'real_B', 'fake_B',
            'score_real', 'score_fake'
        ]
    fp = f'{save_dir}/{score}.csv'
    if not os.path.isfile(fp):
        create_csv_file(fp, columns)


# initialize metrics
print('[INFO] Initializing metrics... ', end='')
ssim_score = SSIMScore()
inception_score = InceptionScore()
ssd_score = SSDScore()
pckh_score = PCKhScore()
lpips_score = LPIPSScore()
print('OK\n')


# evaluate metrics on real images
print('[INCEPTION SCORE] Evaluating real_A')
is_real_A = inception_score.eval(real_A_images, verbose=True)
print('[INCEPTION SCORE] Evaluating real_B')
is_real_B = inception_score.eval(real_B_images, verbose=True)
print('')

print('[SSD SCORE] Evaluating real_A')
ssd_real_A = ssd_score.eval(real_A_images, verbose=True)
print('[SSD SCORE] Evaluating real_B')
ssd_real_B = ssd_score.eval(real_B_images, verbose=True)
print('')

if not os.path.isfile(db_keypoints):
    real_A_keypoints = get_keypoints(fp_real_A_keypoints)
    real_B_keypoints = get_keypoints(fp_real_B_keypoints)
else:
    real_A_keypoints = retrieve_keypoints(db_keypoints, real_A_images)
    real_B_keypoints = retrieve_keypoints(db_keypoints, real_B_images)
print('')


# evaluate metrics on fake images at each checkpoint
for ckpt_id in ckpt_ids:
    print('-'*20, f'ITER {ckpt_id}', '-'*20)
    
    fake_A_images = sorted(glob.glob(f'{test_dir}/fake_images/iter_{ckpt_id}/fake_A/*.jpg'))
    fake_B_images = sorted(glob.glob(f'{test_dir}/fake_images/iter_{ckpt_id}/fake_B/*.jpg'))
    
    # 1. evaluate ssim score
    print('[SSIM SCORE] Evaluating fake_A')
    ssim_fake_A = ssim_score.eval(real_A_images, fake_A_images, verbose=True)
    print('[SSIM SCORE] Evaluating fake_B')
    ssim_fake_B = ssim_score.eval(real_B_images, fake_B_images, verbose=True)
    fp = f'{save_dir}/ssim_score.csv'
    update_csv_file(fp, [
        ckpt_id,
        1.0, ssim_fake_A[0],
        1.0, ssim_fake_B[0],
        1.0, (ssim_fake_A[0] + ssim_fake_B[0]) / 2.0
    ])
    print(f'[SSIM SCORE] Scores saved to {fp}\n')
    
    # 2. evaluate inception score
    print('[INCEPTION SCORE] Evaluating fake_A')
    is_fake_A = inception_score.eval(fake_A_images, verbose=True)
    print('[INCEPTION SCORE] Evaluating fake_B')
    is_fake_B = inception_score.eval(fake_B_images, verbose=True)
    fp = f'{save_dir}/inception_score.csv'
    update_csv_file(fp, [
        ckpt_id,
        is_real_A[0], is_fake_A[0],
        is_real_B[0], is_fake_B[0],
        (is_real_A[0] + is_real_B[0]) / 2.0, (is_fake_A[0] + is_fake_B[0]) / 2.0
    ])
    print(f'[INCEPTION SCORE] Scores saved to {fp}\n')
    
    # 3. evaluate ssd score
    print('[SSD SCORE] Evaluating fake_A')
    ssd_fake_A = ssd_score.eval(fake_A_images, verbose=True)
    print('[SSD SCORE] Evaluating fake_B')
    ssd_fake_B = ssd_score.eval(fake_B_images, verbose=True)
    fp = f'{save_dir}/ssd_score.csv'
    update_csv_file(fp, [
        ckpt_id,
        ssd_real_A[0], ssd_fake_A[0],
        ssd_real_B[0], ssd_fake_B[0],
        (ssd_real_A[0] + ssd_real_B[0]) / 2.0, (ssd_fake_A[0] + ssd_fake_B[0]) / 2.0
    ])
    print(f'[SSD SCORE] Scores saved to {fp}\n')
    
    # 4. evaluate pckh score
    fp_fake_A_keypoints = f'{test_dir}/fake_images/iter_{ckpt_id}/fake_A_keypoints.csv'
    fp_fake_B_keypoints = f'{test_dir}/fake_images/iter_{ckpt_id}/fake_B_keypoints.csv'
    fake_A_keypoints = get_keypoints(fp_fake_A_keypoints)
    fake_B_keypoints = get_keypoints(fp_fake_B_keypoints)
    print('[PCKh SCORE] Evaluating fake_A')
    pckh_fake_A = pckh_score.eval(real_A_keypoints, fake_A_keypoints, verbose=True)
    print('[PCKh SCORE] Evaluating fake_B')
    pckh_fake_B = pckh_score.eval(real_B_keypoints, fake_B_keypoints, verbose=True)
    fp = f'{save_dir}/pckh_score.csv'
    update_csv_file(fp, [
        ckpt_id,
        1.0, pckh_fake_A[0],
        1.0, pckh_fake_B[0],
        1.0, (pckh_fake_A[1] + pckh_fake_B[1]) / (pckh_fake_A[2] + pckh_fake_B[2])
    ])
    print(f'[PCKh SCORE] Scores saved to {fp}\n')
    
    # 5. evaluate lpips score
    print('[LPIPS SCORE] Evaluating fake_A')
    lpips_dict_fake_A = lpips_score.eval(real_A_images, fake_A_images, verbose=True)
    print('[LPIPS SCORE] Evaluating fake_B')
    lpips_dict_fake_B = lpips_score.eval(real_B_images, fake_B_images, verbose=True)
    fp = f'{save_dir}/lpips_score.csv'
    update_csv_file(fp, [
        ckpt_id,
        0.0, lpips_dict_fake_A['lpips_alx'][0], lpips_dict_fake_A['lpips_vgg'][0], lpips_dict_fake_A['lpips_sqz'][0],
        0.0, lpips_dict_fake_B['lpips_alx'][0], lpips_dict_fake_B['lpips_vgg'][0], lpips_dict_fake_B['lpips_sqz'][0],
        0.0,
        (lpips_dict_fake_A['lpips_alx'][0] + lpips_dict_fake_B['lpips_alx'][0]) / 2.0,
        (lpips_dict_fake_A['lpips_vgg'][0] + lpips_dict_fake_B['lpips_vgg'][0]) / 2.0,
        (lpips_dict_fake_A['lpips_sqz'][0] + lpips_dict_fake_B['lpips_sqz'][0]) / 2.0
    ])
    print(f'[LPIPS SCORE] Scores saved to {fp}\n')
