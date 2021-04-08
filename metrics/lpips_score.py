import lpips
import numpy as np
import torch


class LPIPSScore(object):
    
    def __init__(self, use_gpu=True):
        self.cuda = True if use_gpu and torch.cuda.is_available() else False
        self.lpips_alx = lpips.LPIPS(net='alex', verbose=False)
        self.lpips_vgg = lpips.LPIPS(net='vgg', verbose=False)
        self.lpips_sqz = lpips.LPIPS(net='squeeze', verbose=False)
        if self.cuda:
            self.lpips_alx.cuda()
            self.lpips_vgg.cuda()
            self.lpips_sqz.cuda()
    
    def eval(self, images1, images2, verbose=False):
        scores_alx = []
        scores_vgg = []
        scores_sqz = []
        for i, (a, b) in enumerate(zip(images1, images2)):
            im1 = lpips.im2tensor(lpips.load_image(a))
            im2 = lpips.im2tensor(lpips.load_image(b))
            if self.cuda:
                im1 = im1.cuda()
                im2 = im2.cuda()
            with torch.no_grad():
                d_alx = self.lpips_alx.forward(im1, im2).item()
                d_vgg = self.lpips_vgg.forward(im1, im2).item()
                d_sqz = self.lpips_sqz.forward(im1, im2).item()
            scores_alx.append(d_alx)
            scores_vgg.append(d_vgg)
            scores_sqz.append(d_sqz)
            if verbose:
                print(f'\r[LPIPS SCORE] Progress: {(i+1)*100.0/len(images1):3.0f}% |',
                      f'mean_alx: {np.mean(scores_alx):.3f} | std_alx: {np.std(scores_alx):.3f} |',
                      f'mean_vgg: {np.mean(scores_vgg):.3f} | std_vgg: {np.std(scores_vgg):.3f} |',
                      f'mean_sqz: {np.mean(scores_sqz):.3f} | std_sqz: {np.std(scores_sqz):.3f} |', end='')
        if verbose:
            print('')
        
        return {
            'lpips_alx': (np.mean(scores_alx), np.std(scores_alx)),
            'lpips_vgg': (np.mean(scores_vgg), np.std(scores_vgg)),
            'lpips_sqz': (np.mean(scores_sqz), np.std(scores_sqz))
        }
