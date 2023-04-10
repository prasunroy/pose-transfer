import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.inception import inception_v3
from torchvision.transforms import transforms
from scipy.stats import entropy
from PIL import Image


class ImageDataset(Dataset):
    
    def __init__(self, images):
        super(ImageDataset, self).__init__()
        self.images = images
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = Image.open(self.images[index])
        return self.transforms(image)


class InceptionScore(object):
    
    def __init__(self, use_gpu=True):
        self.dtype = torch.cuda.FloatTensor if use_gpu and torch.cuda.is_available() else torch.FloatTensor
        self.model = inception_v3(weights='IMAGENET1K_V1', transform_input=False).type(self.dtype)
        self.model.eval()
    
    def _get_predictions(self, x, resize=False):
        if resize:
            x = F.interpolate(x, size=(299, 299), mode='bilinear')
        with torch.no_grad():
            y = self.model(x)
        return F.softmax(y, dim=1).detach().cpu().numpy()
    
    def _get_kldiv(self, predictions, splits=1):
        split_length = predictions.shape[0] // splits
        split_scores = []
        for k in range(splits):
            part = predictions[k * split_length : (k + 1) * split_length, :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
        return (np.mean(split_scores), np.std(split_scores))
    
    def eval(self, images, batch_size=32, resize=True, splits=1, verbose=False):
        n = len(images)
        dataloader = DataLoader(ImageDataset(images), batch_size=batch_size)
        predictions = np.zeros((n, 1000))
        for i, batch in enumerate(dataloader):
            batch = batch.type(self.dtype)
            batch_size_i = batch.size(0)
            predictions[i * batch_size : i * batch_size + batch_size_i] = self._get_predictions(batch, resize=resize)
            if verbose:
                print(f'\r[INCEPTION SCORE] Progress: {(i+1)*100.0/len(dataloader):3.0f}% | ', end='')
        is_mean, is_std = self._get_kldiv(predictions, splits=splits)
        if verbose:
            print(f'mean: {is_mean:.3f} | std: {is_std:.3f} |')
        return (is_mean, is_std)
