import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# -----------------------------------------------------------------------------
# GAN Loss
# -----------------------------------------------------------------------------
class GANLoss(nn.Module):
    
    def __init__(self, real_label=1.0, fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.criterion = nn.MSELoss()
    
    def forward(self, prediction, is_target_real):
        if is_target_real:
            target_tensor = self.real_label.expand_as(prediction)
        else:
            target_tensor = self.fake_label.expand_as(prediction)
        return self.criterion(prediction, target_tensor)


# -----------------------------------------------------------------------------
# Perceptual Loss
# -----------------------------------------------------------------------------
class PerceptualLoss(nn.Module):
    
    def __init__(self, model=None, mean=None, std=None, mode='L1', scaling=True):
        super(PerceptualLoss, self).__init__()
        self.model = models.vgg19(weights='IMAGENET1K_V1').features if not model else model
        mean = torch.tensor([0.485, 0.456, 0.406]).float() if not mean else mean.float()
        std = torch.tensor([0.229, 0.224, 0.225]).float() if not std else std.float()
        self.register_buffer('mean', mean.view(1, mean.size(0), 1, 1))
        self.register_buffer('std', std.view(1, std.size(0), 1, 1))
        self.mode = mode
        self.scaling = scaling
    
    def forward(self, x1, x2):
        if self.scaling:
            x1 = (x1 - x1.min()) / (x1.max() - x1.min())
            x2 = (x2 - x2.min()) / (x2.max() - x2.min())
        
        x1_out = (x1 - self.mean) / self.std
        x2_out = (x2 - self.mean) / self.std
        x1_out = self.model(x1_out)
        x2_out = self.model(x2_out)
        
        if self.mode.upper() == 'L1':
            loss = F.l1_loss(x1_out, x2_out)
        elif self.mode.upper() == 'MSE':
            loss = F.mse_loss(x1_out, x2_out)
        else:
            raise ValueError(self.mode)
        
        return loss


# -----------------------------------------------------------------------------
# Structural Similarity (SSIM) Loss
# -----------------------------------------------------------------------------
class SSIMLoss(nn.Module):
    
    def __init__(self, n_channels=3, window_size=11, size_average=True, ensure_finite=True):
        super(SSIMLoss, self).__init__()
        self._n_channels = n_channels
        self._window_size = window_size
        self._size_average = size_average
        self._ensure_finite = ensure_finite
        self.register_buffer('window', self._create_window(window_size, n_channels))
    
    def forward(self, x1, x2):
        return 1.0 - self.ssim(x1, x2)
    
    def ssim(self, x1, x2):
        mu1 = F.conv2d(x1, self.window, padding=self._window_size//2, groups=self._n_channels)
        mu2 = F.conv2d(x2, self.window, padding=self._window_size//2, groups=self._n_channels)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(x1*x1, self.window, padding=self._window_size//2, groups=self._n_channels) - mu1_sq
        sigma2_sq = F.conv2d(x2*x2, self.window, padding=self._window_size//2, groups=self._n_channels) - mu2_sq
        sigma_1_2 = F.conv2d(x1*x2, self.window, padding=self._window_size//2, groups=self._n_channels) - mu1mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1mu2+C1)*(2*sigma_1_2+C2))/((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
        
        if self._size_average:
            _ssim = ssim_map.mean()
        else:
            _ssim = ssim_map.mean(1).mean(1).mean(1)
        
        if self._ensure_finite and not math.isfinite(_ssim):
            _ssim = (x1 + x2).mean() * 0.0
        
        return _ssim
    
    def _create_window(self, window_size, n_channels):
        window_1D = self._gaussian(window_size, 1.5).unsqueeze(1)
        window_2D = window_1D.mm(window_1D.t()).float().unsqueeze(0).unsqueeze(0)
        window = window_2D.expand(n_channels, 1, window_size, window_size).contiguous()
        return window
    
    def _gaussian(self, window_size, sigma):
        g = torch.Tensor([math.exp(-(x-window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        g /= g.sum()
        return g


# -----------------------------------------------------------------------------
# Multi-scale Structural Similarity (MS-SSIM) Loss
# -----------------------------------------------------------------------------
class MSSSIMLoss(nn.Module):
    
    def __init__(self, n_channels=3, window_size=11, value_range=None, size_average=True, ensure_finite=True):
        super(MSSSIMLoss, self).__init__()
        self._n_channels = n_channels
        self._window_size = window_size
        self._value_range = value_range
        self._size_average = size_average
        self._ensure_finite = ensure_finite
        self.register_buffer('window', self._create_window(window_size, n_channels))
        self.register_buffer('weight', torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).float())
    
    def forward(self, x1, x2):
        return 1.0 - self.msssim(x1, x2)
    
    def msssim(self, x1, x2):
        ssim_list = []
        cs_list = []
        for _ in range(self.weight.size(0)):
            _ssim, cs = self.ssim(x1, x2)
            ssim_list.append(_ssim)
            cs_list.append(cs)
            x1 = F.avg_pool2d(x1, (2, 2))
            x2 = F.avg_pool2d(x2, (2, 2))
        
        t1 = torch.stack(ssim_list) ** self.weight
        t2 = torch.stack(cs_list) ** self.weight
        
        _msssim = torch.prod(t2[:-1] * t1[-1])
        
        if self._ensure_finite and not math.isfinite(_msssim):
            _msssim = (x1 + x2).mean() * 0.0
        
        return _msssim
    
    def ssim(self, x1, x2):
        if self._value_range is None:
            max_value = 255 if max(x1.max().item(), x2.max().item()) > 127 else 1
            min_value = -1 if min(x1.min().item(), x2.min().item()) < -0.5 else 0
            L = max_value - min_value
        else:
            L = self._value_range
        
        mu1 = F.conv2d(x1, self.window, padding=0, groups=self._n_channels)
        mu2 = F.conv2d(x2, self.window, padding=0, groups=self._n_channels)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(x1*x1, self.window, padding=0, groups=self._n_channels) - mu1_sq
        sigma2_sq = F.conv2d(x2*x2, self.window, padding=0, groups=self._n_channels) - mu2_sq
        sigma_1_2 = F.conv2d(x1*x2, self.window, padding=0, groups=self._n_channels) - mu1mu2
        
        C1 = (0.01*L)**2
        C2 = (0.03*L)**2
        
        v1 = 2.0 * sigma_1_2 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1/v2)
        
        ssim_map = ((2 * mu1mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
        
        if self._size_average:
            _ssim = ssim_map.mean()
        else:
            _ssim = ssim_map.mean(1).mean(1).mean(1)
        
        return (_ssim, cs)
    
    def _create_window(self, window_size, n_channels):
        window_1D = self._gaussian(window_size, 1.5).unsqueeze(1)
        window_2D = window_1D.mm(window_1D.t()).float().unsqueeze(0).unsqueeze(0)
        window = window_2D.expand(n_channels, 1, window_size, window_size).contiguous()
        return window
    
    def _gaussian(self, window_size, sigma):
        g = torch.Tensor([math.exp(-(x-window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        g /= g.sum()
        return g
