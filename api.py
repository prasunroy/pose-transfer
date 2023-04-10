# -*- coding: utf-8 -*-
"""
A stand-alone inference api for `Multi-scale Attention Guided Pose Transfer`
Pattern Recognition 2023 (https://doi.org/10.1016/j.patcog.2023.109315).
Created on Mon Apr 10 19:00:00 2023
Author: Prasun Roy | University of Technology Sydney (https://prasunroy.github.io)
GitHub: https://github.com/prasunroy/pose-transfer

"""


import numpy as np
import os
import shutil
import tempfile
import torch
import torch.nn as nn
import torchvision.transforms as T
from openpose.body.estimator import BodyPoseEstimator
from tqdm import tqdm
from urllib.parse import urlparse
from urllib.request import urlopen


CHECKPOINT_URL = 'https://www.dropbox.com/s/iqrzd6lypguxw5i/pose_transfer_coco_keypoints_deepfashion_netG_iter_260500.pth?dl=1'
CHECKPOINT_DIR = os.path.join(os.path.expanduser('~'), '.cache/torch/checkpoints/')


def _download_file(url, path, progress=True):
    link = urlopen(url)
    meta = link.info()
    content_length = meta.get_all('Content-Length')
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    else:
        file_size = None
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = link.read(8192)
                if len(buffer) == 0:
                    break
                temp_file.write(buffer)
                pbar.update(len(buffer))
        temp_file.close()
        shutil.move(temp_file.name, path)
    finally:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


def _load_checkpoint(model, checkpoint_url, checkpoint_dir, ignore_cache=False, progress=True):
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    urlparts = urlparse(checkpoint_url)
    filename = os.path.basename(urlparts.path.split('/')[-1])
    cached_file = os.path.join(checkpoint_dir, filename)
    if not os.path.isfile(cached_file) or ignore_cache:
        print(f'Downloading: "{checkpoint_url}" to {cached_file}')
        _download_file(checkpoint_url, cached_file, progress)
    model.load_state_dict(torch.load(cached_file))
    return model


def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)


def downconv2x(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)


def upconv2x(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)


class ResidualBlock(nn.Module):
    
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        layers = [
            conv3x3(num_channels, num_channels),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            conv3x3(num_channels, num_channels),
            nn.BatchNorm2d(num_channels)
        ]
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        y = self.layers(x) + x
        return y


class NetG(nn.Module):
    
    def __init__(self, in1_channels, in2_channels, out_channels, ngf=64):
        super(NetG, self).__init__()
        
        self.in1_conv1 = self.inconv(in1_channels, ngf)
        self.in1_down1 = self.down2x(ngf, ngf*2)
        self.in1_down2 = self.down2x(ngf*2, ngf*4)
        self.in1_down3 = self.down2x(ngf*4, ngf*8)
        self.in1_down4 = self.down2x(ngf*8, ngf*16)
        
        self.in2_conv1 = self.inconv(in2_channels, ngf)
        self.in2_down1 = self.down2x(ngf, ngf*2)
        self.in2_down2 = self.down2x(ngf*2, ngf*4)
        self.in2_down3 = self.down2x(ngf*4, ngf*8)
        self.in2_down4 = self.down2x(ngf*8, ngf*16)
        
        self.out_up1 = self.up2x(ngf*16, ngf*8)
        self.out_up2 = self.up2x(ngf*8, ngf*4)
        self.out_up3 = self.up2x(ngf*4, ngf*2)
        self.out_up4 = self.up2x(ngf*2, ngf)
        self.out_conv1 = self.outconv(ngf, out_channels)
    
    def inconv(self, in_channels, out_channels):
        return nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def outconv(self, in_channels, out_channels):
        return nn.Sequential(
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            conv1x1(in_channels, out_channels),
            nn.Tanh()
        )
    
    def down2x(self, in_channels, out_channels):
        return nn.Sequential(
            downconv2x(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )
    
    def up2x(self, in_channels, out_channels):
        return nn.Sequential(
            upconv2x(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )
    
    def forward(self, x1, x2):
        x1_c1 = self.in1_conv1(x1)
        x1_d1 = self.in1_down1(x1_c1)
        x1_d2 = self.in1_down2(x1_d1)
        x1_d3 = self.in1_down3(x1_d2)
        x1_d4 = self.in1_down4(x1_d3)
        
        x2_c1 = self.in2_conv1(x2)
        x2_d1 = self.in2_down1(x2_c1)
        x2_d2 = self.in2_down2(x2_d1)
        x2_d3 = self.in2_down3(x2_d2)
        x2_d4 = self.in2_down4(x2_d3)
        
        y = x1_d4 * torch.sigmoid(x2_d4)
        y = self.out_up1(y)
        y = y * torch.sigmoid(x2_d3)
        y = self.out_up2(y)
        y = y * torch.sigmoid(x2_d2)
        y = self.out_up3(y)
        y = y * torch.sigmoid(x2_d1)
        y = self.out_up4(y)
        y = self.out_conv1(y)
        
        return y


class Pose2Pose(object):
    
    def __init__(self, pretrained=False, ignore_cache=False, checkpoint=None):
        self.openpose = BodyPoseEstimator(pretrained=True)
        self.renderer = NetG(3, 36, 3).eval()
        if checkpoint is not None:
            self.renderer.load_state_dict(torch.load(checkpoint))
        elif pretrained:
            self.renderer = _load_checkpoint(self.renderer, CHECKPOINT_URL, CHECKPOINT_DIR, ignore_cache)
        if torch.cuda.is_available():
            self.renderer = self.renderer.cuda()
    
    def _resize_and_pad_image(self, image, size=256):
        w = size * image.width // image.height
        w_box = min(w, size * 11 // 16)
        image = T.Resize((size, w), interpolation=T.InterpolationMode.BICUBIC)(image)
        image = T.CenterCrop((size, w_box))(image)
        image = T.Pad(size - w_box, fill=255)(image)
        image = T.CenterCrop((size, size))(image)
        return image
    
    def _estimate_keypoints(self, image):
        x = np.uint8(image.convert('RGB'))
        keypoints = self.openpose(x)
        keypoints = keypoints[0] if len(keypoints) > 0 else np.zeros((18, 3), dtype=np.int32)
        keypoints[np.where(keypoints[:, 2]==0), :2] = -1
        keypoints = keypoints[:, :2]
        return keypoints
    
    def _keypoints2heatmaps(self, keypoints, size=256):
        heatmaps = np.zeros((size, size, keypoints.shape[0]), dtype=np.float32)
        for k in range(keypoints.shape[0]):
            x, y = keypoints[k]
            if x == -1 or y == -1:
                continue
            heatmaps[y, x, k] = 1.0
        return heatmaps
    
    def _transform_input(self, x, normalize=True):
        x = T.ToTensor()(x)
        if normalize:
            x = T.Normalize((0.5,), (0.5,))(x)
        return x
    
    def _transform_output(self, x, denormalize=True):
        if denormalize:
            x = T.Normalize((-1.0,), (2.0,))(x)
        x = T.ToPILImage()(x)
        return x
    
    @torch.no_grad()
    def _transfer_pose(self, imgA, mapAB):
        if torch.cuda.is_available():
            imgA, mapAB = imgA.cuda(), mapAB.cuda()
        return self.renderer(imgA, mapAB).cpu()
    
    def transfer_as(self, condition, target_pose_reference):
        imgA = self._resize_and_pad_image(condition.convert('RGB'))
        kptA = self._estimate_keypoints(imgA)
        mapA = self._keypoints2heatmaps(kptA)
        imgB = self._resize_and_pad_image(target_pose_reference.convert('RGB'))
        kptB = self._estimate_keypoints(imgB)
        mapB = self._keypoints2heatmaps(kptB)
        imgA_t = self._transform_input(imgA, normalize=True).unsqueeze(0)
        mapA_t = self._transform_input(mapA, normalize=False).unsqueeze(0)
        mapB_t = self._transform_input(mapB, normalize=False).unsqueeze(0)
        mapAB_t = torch.cat((mapA_t, mapB_t), dim=1)
        out = self._transfer_pose(imgA_t, mapAB_t).squeeze()
        out = self._transform_output(out, denormalize=True).convert(condition.mode)
        crop_size = (256, min((256 * target_pose_reference.width // target_pose_reference.height), 176))
        return T.CenterCrop(crop_size)(out)
