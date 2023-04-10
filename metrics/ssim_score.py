import cv2
import numpy as np
from skimage.metrics import structural_similarity


class SSIMScore(object):
    
    def eval(self, images1, images2, multichannel=True, verbose=False):
        scores = []
        for i, (a, b) in enumerate(zip(images1, images2)):
            im1 = cv2.imread(a)
            im2 = cv2.imread(b)
            ssim = structural_similarity(im1, im2, channel_axis=-1)
            scores.append(ssim)
            if verbose:
                print(f'\r[SSIM SCORE] Progress: {(i+1)*100.0/len(images1):3.0f}% |',
                      f'mean: {np.mean(scores):.3f} | std: {np.std(scores):.3f} |', end='')
        if verbose:
            print('')
        
        return (np.mean(scores), np.std(scores))
