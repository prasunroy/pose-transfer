import numpy as np
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PoseDataset(Dataset):
    
    def __init__(self, dataset_dir, img_pairs, pose_maps_dir,
                 img_transform=None, map_transform=None, reverse=False):
        super(PoseDataset, self).__init__()
        self._dataset_dir = dataset_dir
        self._img_pairs = pd.read_csv(img_pairs)
        self._pose_maps_dir = pose_maps_dir
        self._img_transform = img_transform or transforms.ToTensor()
        self._map_transform = map_transform or transforms.ToTensor()
        self._reverse = reverse
    
    def __len__(self):
        return len(self._img_pairs)
    
    def __getitem__(self, index):
        pthA = self._img_pairs.iloc[index].imgA
        pthB = self._img_pairs.iloc[index].imgB
        
        fidA = os.path.splitext(pthA)[0].replace('/', '').replace('\\', '')
        fidB = os.path.splitext(pthB)[0].replace('/', '').replace('\\', '')
        
        imgA = Image.open(os.path.join(self._dataset_dir, pthA))
        imgB = Image.open(os.path.join(self._dataset_dir, pthB))
        
        mapA = np.float32(np.load(os.path.join(self._pose_maps_dir, f'{fidA}.npz'))['arr_0'])
        mapB = np.float32(np.load(os.path.join(self._pose_maps_dir, f'{fidB}.npz'))['arr_0'])
        
        imgA = self._img_transform(imgA)
        imgB = self._img_transform(imgB)
        
        mapA = self._map_transform(mapA)
        mapB = self._map_transform(mapB)
        
        if not self._reverse:
            return {'imgA': imgA, 'imgB': imgB, 'mapA': mapA, 'mapB': mapB, 'fidA': fidA, 'fidB': fidB}
        else:
            return {'imgA': imgB, 'imgB': imgA, 'mapA': mapB, 'mapB': mapA, 'fidA': fidB, 'fidB': fidA}


def create_dataloader(dataset_dir, img_pairs, pose_maps_dir,
                      img_transform=None, map_transform=None, reverse=False,
                      batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
    dataset = PoseDataset(dataset_dir, img_pairs, pose_maps_dir, img_transform, map_transform, reverse)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)
