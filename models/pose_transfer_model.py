import torch
import torchvision
from .base_model import BaseModel
from .netG import NetG
from .netD import NetD
from .losses import GANLoss, PerceptualLoss


class PoseTransferModel(BaseModel):
    
    def __init__(self, gpuids=None):
        super(PoseTransferModel, self).__init__()
        self.models = ['netG', 'netD']
        self.losses = ['lossG_L1', 'lossG_GAN', 'lossG_PER', 'lossG', 'lossD_fake', 'lossD_real', 'lossD']
        self.gpuids = gpuids if isinstance(gpuids, list) or isinstance(gpuids, tuple) else []
        self.device = None
        
        self.setup(verbose=True)
        
        self.netG = NetG(3, 36, 3)
        self.netD = NetD(6)
        
        self.init_networks(verbose=True)
        
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionGAN = GANLoss().to(self.device)
        self.criterionPERL4 = PerceptualLoss(self.build_vgg19_sub_model(4)).to(self.device)
        self.criterionPERL9 = PerceptualLoss(self.build_vgg19_sub_model(9)).to(self.device)
        
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.001, betas=(0.5, 0.999))
    
    def set_inputs(self, inputs):
        self.real_img_A = inputs['imgA'].to(self.device)
        self.real_img_B = inputs['imgB'].to(self.device)
        self.real_map_A = inputs['mapA'].to(self.device)
        self.real_map_B = inputs['mapB'].to(self.device)
    
    def forward(self):
        real_map_AB = torch.cat((self.real_map_A, self.real_map_B), dim=1)
        self.fake_img_B = self.netG(self.real_img_A, real_map_AB)
    
    def backward_D(self):
        fake_img_AB = torch.cat((self.real_img_A, self.fake_img_B), dim=1)
        pred_fake = self.netD(fake_img_AB.detach())
        self.lossD_fake = self.criterionGAN(pred_fake, False)
        
        real_img_AB = torch.cat((self.real_img_A, self.real_img_B), dim=1)
        pred_real = self.netD(real_img_AB)
        self.lossD_real = self.criterionGAN(pred_real, True)
        
        self.lossD = 0.5 * (self.lossD_fake + self.lossD_real)
        self.lossD.backward()
    
    def backward_G(self):
        self.lossG_L1 = self.criterionL1(self.fake_img_B, self.real_img_B)
        
        fake_img_AB = torch.cat((self.real_img_A, self.fake_img_B), dim=1)
        pred_fake = self.netD(fake_img_AB)
        self.lossG_GAN = self.criterionGAN(pred_fake, True)
        
        lossG_PER_L4 = self.criterionPERL4(self.fake_img_B, self.real_img_B)
        lossG_PER_L9 = self.criterionPERL9(self.fake_img_B, self.real_img_B)
        self.lossG_PER = lossG_PER_L4 + lossG_PER_L9
        
        self.lossG = 5.0 * self.lossG_L1 + 1.0 * self.lossG_GAN + 5.0 * self.lossG_PER
        self.lossG.backward()
    
    def backward(self):
        self.set_requires_grad(['netD'], True)
        self.optimizerD.zero_grad()
        self.backward_D()
        self.optimizerD.step()
        
        self.set_requires_grad(['netD'], False)
        self.optimizerG.zero_grad()
        self.backward_G()
        self.optimizerG.step()
    
    def compute_visuals(self):
        mode = self.netG.training
        self.netG.eval()
        real_map_AB = torch.cat((self.real_map_A, self.real_map_B), dim=1)
        with torch.no_grad():
            self.fake_img_B = self.netG(self.real_img_A, real_map_AB)
        real_img_A = torchvision.utils.make_grid(self.real_img_A.detach().cpu(), nrow=1, normalize=True)
        real_img_B = torchvision.utils.make_grid(self.real_img_B.detach().cpu(), nrow=1, normalize=True)
        fake_img_B = torchvision.utils.make_grid(self.fake_img_B.detach().cpu(), nrow=1, normalize=True)
        grid_image = torch.cat((real_img_A, real_img_B, fake_img_B), dim=2)
        self.netG.train(mode)
        return grid_image
    
    @staticmethod
    def build_vgg19_sub_model(n_layers=0):
        model = torchvision.models.vgg19(weights='IMAGENET1K_V1').features
        layers = []
        for i, layer in enumerate(model.children()):
            if i >= n_layers:
                break
            layers.append(layer)
        return torch.nn.Sequential(*layers)
