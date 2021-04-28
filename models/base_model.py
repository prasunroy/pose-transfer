import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from collections import OrderedDict


class BaseModel(ABC):
    
    def __init__(self):
        self.models = []
        self.losses = []
        self.gpuids = []
        self.device = None
        self.setup()
    
    def setup(self, verbose=False):
        assert isinstance(self.models, list) or isinstance(self.models, tuple)
        assert isinstance(self.losses, list) or isinstance(self.losses, tuple)
        assert isinstance(self.gpuids, list) or isinstance(self.gpuids, tuple)
        self.models = [name for name in self.models if isinstance(name, str)]
        self.losses = [name for name in self.losses if isinstance(name, str)]
        self.gpuids = [index for index in self.gpuids if torch.cuda.is_available() \
                       and index in range(0, torch.cuda.device_count())]
        self.device = torch.device(f'cuda:{self.gpuids[0]}') if len(self.gpuids) > 0 else torch.device('cpu')
        if verbose:
            if len(self.gpuids) > 0:
                for index in self.gpuids:
                    print(f'[INFO] Using device: GPU{index} -> {torch.cuda.get_device_name(index)}')
            else:
                print('[INFO] Using device: CPU')
    
    @abstractmethod
    def set_inputs(self, *inputs):
        pass
    
    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def backward(self):
        pass
    
    def optimize_parameters(self):
        self.forward()
        self.backward()
    
    def get_losses(self):
        loss_dict = OrderedDict()
        for name in self.losses:
            loss_dict[name] = getattr(self, name).item()
        return loss_dict
    
    def print_networks(self, verbose=False):
        print('-'*80)
        for name in self.models:
            network = getattr(self, name)
            n_params = 0
            for param in network.parameters():
                n_params += param.numel()
            if verbose:
                print(network)
            print(f'[INFO] Total parameters of network {name}: {n_params/1e6:.2f}M')
        print('-'*80)
    
    def init_networks(self, init_type='normal', init_gain=0.02, verbose=False):
        def init_params(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, init_gain)
                elif init_type == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight.data, init_gain)
                elif init_type == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight.data)
                else:
                    raise NotImplementedError(init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
                nn.init.constant_(m.bias.data, 0.0)
        
        for name in self.models:
            network = getattr(self, name)
            if len(self.gpuids) > 0 and torch.cuda.is_available():
                network.to(self.gpuids[0])
                network = nn.DataParallel(network, self.gpuids)
                setattr(self, name, network)
            network.apply(init_params)
            if verbose:
                print(f'[INFO] Network {name} initialized')
    
    def save_networks(self, root, suffix, verbose=False):
        if not os.path.isdir(root):
            os.makedirs(root)
        for name in self.models:
            network = getattr(self, name)
            filepath = os.path.join(root, f'{name}_{suffix}.pth')
            if isinstance(network, torch.nn.DataParallel):
                torch.save(network.module.cpu().state_dict(), filepath)
                network.cuda(self.gpuids[0])
            else:
                torch.save(network.cpu().state_dict(), filepath)
            if verbose:
                print(f'[INFO] Network {name} weights saved to {filepath}')
    
    def load_networks(self, root, suffix, verbose=False):
        for name in self.models:
            network = getattr(self, name)
            if isinstance(network, torch.nn.DataParallel):
                network = network.module
            filepath = os.path.join(root, f'{name}_{suffix}.pth')
            network.load_state_dict(torch.load(filepath, map_location=self.device))
            if verbose:
                print(f'[INFO] Network {name} weights loaded from {filepath}')
    
    def set_requires_grad(self, network_names, requires_grad=True):
        assert isinstance(network_names, list) or isinstance(network_names, tuple)
        network_names = [name for name in network_names if isinstance(name, str)]
        for name in network_names:
            network = getattr(self, name)
            for param in network.parameters():
                param.requires_grad = requires_grad
