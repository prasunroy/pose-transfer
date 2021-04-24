# imports
import csv
import glob
import json
import numpy as np
import os
import pandas as pd
import pathlib
import random
import shutil
import uuid
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS


# configurations
# -----------------------------------------------------------------------------
imagedir = './static/images/test'
database = './db.json'

submissions1 = './submissions1.csv'
submissions2 = './submissions2.csv'

real_images = sorted(glob.glob('./real_images/*.jpg'))
fake_images = sorted(glob.glob('./fake_images/*.jpg'))

num_images_per_category = 10

delete_cache = False
# -----------------------------------------------------------------------------


# write a csv file
def write_csv(path, mode, data):
    with open(path, mode, newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(data)


# verify integrity of database
if not os.path.isfile(database):
    delete_cache = True
else:
    print('[INFO] Verifying integrity of database... ', end='')
    with open(database, 'r') as fp:
        db = json.load(fp)
    if len(db) < 2 * num_images_per_category:
        delete_cache = True
    else:
        for fp, label in db.items():
            if not os.path.isfile(fp) or not label in [1, 0]:
                delete_cache = True
                break
    if delete_cache:
        print('integrity failure')
    else:
        print('OK')


# delete cache
if delete_cache:
    assert len(real_images) >= num_images_per_category,\
        f'Minimum number of real images specified to be {num_images_per_category} but found {len(real_images)}'
    assert len(fake_images) >= num_images_per_category,\
        f'Minimum number of fake images specified to be {num_images_per_category} but found {len(fake_images)}'
    if os.path.isdir(imagedir):
        print('[INFO] Deleting image cache... ', end='')
        shutil.rmtree(imagedir)
        print('OK')
    if os.path.isfile(database):
        print('[INFO] Deleting database cache... ', end='')
        os.remove(database)
        print('OK')


# check database
if not os.path.isfile(database):
    print(f'[INFO] Database not found at {database}')
    if not os.path.isdir(imagedir):
        os.makedirs(imagedir)
    images = real_images + fake_images
    db = {}
    for i, image in enumerate(images):
        fp = f'{imagedir}/{str(uuid.uuid1()) + os.path.splitext(image)[1]}'
        fp = pathlib.Path(fp).as_posix()
        shutil.copy2(image, fp)
        db[fp] = 1 if i < len(real_images) else 0
        print(f'\r[INFO] Creating database... {(i+1)*100.0/len(images):3.0f}%', end='')
    print('')
    with open(database, 'w') as fp:
        json.dump(db, fp, separators=(',', ':'))
    print(f'[INFO] Database serialized to {database}')
else:
    with open(database, 'r') as fp:
        db = json.load(fp)
    print(f'[INFO] Database loaded from {database}')


# check submission records
for fp in [submissions1, submissions2]:
    if not os.path.isfile(fp):
        print(f'[INFO] Submission records not found at {fp}')
        write_csv(fp, 'w', ['submission_id', 'R2G', 'G2R'])
        print(f'[INFO] Created new submission records at {fp}')
    else:
        print(f'[INFO] Found existing submission records at {fp}')


# create shuffled lists of real and fake images
real_images = []
fake_images = []
for fp, label in db.items():
    if label == 1:
        real_images.append(fp)
    elif label == 0:
        fake_images.append(fp)

random.shuffle(real_images)
random.shuffle(fake_images)


# initialize and configure application
app = Flask(__name__)

app.config['DEBUG'] = False


# enable Cross-Origin Resource Sharing (CORS)
CORS(app)


# define routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/practice', methods=['GET'])
def practice():
    return render_template('practice.html')

@app.route('/test1', methods=['GET'])
def test1():
    return render_template('test1.html')

@app.route('/test2', methods=['GET'])
def test2():
    return render_template('test2.html')

@app.route('/stats', methods=['GET'])
def stats():
    return render_template('stats.html')

@app.route('/error', methods=['GET'])
def error():
    return render_template('error.html')

@app.route('/images', methods=['GET'])
def images():
    success = False
    urls = []
    try:
        urls.extend(random.sample(real_images, num_images_per_category))
        urls.extend(random.sample(fake_images, num_images_per_category))
        random.shuffle(urls)
        success = True
    except:
        urls = []
        success = False
    return jsonify({
        'success': success,
        'images': urls
    })

@app.route('/submit', methods=['POST'])
def submit():
    success = False
    user_r2g = 0
    user_g2r = 0
    global_r2g = 0
    global_g2r = 0
    try:
        data = request.json['data']
        mode = request.json['mode']
        for fp, user_label in data.items():
            if db[fp] == 1 and user_label == 0:
                user_r2g += 1
            elif db[fp] == 0 and user_label == 1:
                user_g2r += 1
        user_r2g /= num_images_per_category
        user_g2r /= num_images_per_category
        submission_id = str(uuid.uuid1())
        submission_fp = submissions1 if mode == 1 else submissions2
        write_csv(submission_fp, 'a', [submission_id, user_r2g, user_g2r])
        df = pd.read_csv(submission_fp)
        global_r2g = np.mean(df['R2G'].values)
        global_g2r = np.mean(df['G2R'].values)
        success = True
    except:
        user_r2g = None
        user_g2r = None
        global_r2g = None
        global_g2r = None
        success = False
    return jsonify({
        'success': success,
        'user_r2g': user_r2g,
        'user_g2r': user_g2r,
        'global_r2g': global_r2g,
        'global_g2r': global_g2r
    })

@app.route('/scores', methods=['GET'])
def scores():
    success = False
    metrics = {}
    try:
        for test_type, submission_fp in zip(['constrained', 'unconstrained'], [submissions1, submissions2]):
            df = pd.read_csv(submission_fp)
            global_r2g = np.mean(df['R2G'].values)
            global_g2r = np.mean(df['G2R'].values)
            global_acc = 1.0 - (global_r2g + global_g2r) / 2.0
            metrics[test_type] = {
                'global_r2g': global_r2g,
                'global_g2r': global_g2r,
                'global_acc': global_acc,
                'submission': len(df)
            }
        success = True
    except:
        metrics = {}
        success = False
    return jsonify({
        'success': success,
        'metrics': metrics
    })


# run application
app.run()
