# imports
import os
import time
import torch
import torchvision
from data.dataloader import create_dataloader
from models.pose_transfer_model import PoseTransferModel


# configurations
# -----------------------------------------------------------------------------
dataset_name = 'DeepFashion'

dataset_root = f'../datasets/{dataset_name}'
img_pairs = f'{dataset_root}/test_img_pairs.csv'
pose_maps_dir = f'{dataset_root}/test_pose_maps'

gpu_ids = [0]

batch_size = 32

run_id = 'pretrained'
ckpt_ids = [260500]

ckpt_dir = f'../output/{dataset_name}/ckpt/{run_id}'
save_dir = f'../output/{dataset_name}/test/{run_id}'
# -----------------------------------------------------------------------------


# create transforms
img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
map_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
out_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage()
])

# create dataloaders
dataloader_AB = create_dataloader(dataset_root, img_pairs, pose_maps_dir,
                                  img_transform, map_transform, reverse=False,
                                  batch_size=batch_size, shuffle=False)
dataloader_BA = create_dataloader(dataset_root, img_pairs, pose_maps_dir,
                                  img_transform, map_transform, reverse=True,
                                  batch_size=batch_size, shuffle=False)

# create model
model = PoseTransferModel(gpuids=gpu_ids)
model.netG.eval()
model.netD.eval()

# create directories for real images
for subdir in ['real_A', 'real_B']:
    directory = os.path.join(save_dir, 'real_images', subdir)
    if not os.path.isdir(directory):
        os.makedirs(directory)

# test model at each checkpoint
for ckpt_id in ckpt_ids:
    
    # create directories for fake images for current checkpoint
    for subdir in ['fake_A', 'fake_B']:
        directory = os.path.join(save_dir, 'fake_images', f'iter_{ckpt_id}', subdir)
        if not os.path.isdir(directory):
            os.makedirs(directory)
    
    # load weights into model for current checkpoint
    model.load_networks(ckpt_dir, ckpt_id, verbose=True)
    
    # generate images
    n_batch = len(dataloader_AB)
    w_batch = len(str(n_batch))
    w_iters = max([len(str(i)) for i in ckpt_ids])
    
    for target, dataloader in zip(['B', 'A'], [dataloader_AB, dataloader_BA]):
        real_dir = os.path.join(save_dir, f'real_images/real_{target}')
        fake_dir = os.path.join(save_dir, f'fake_images/iter_{ckpt_id}/fake_{target}')
        
        n_images = 0
        runtimes = []
        
        for batch, data in enumerate(dataloader):
            model.set_inputs(data)
            real_map_AB = torch.cat((model.real_map_A, model.real_map_B), dim=1)
            with torch.no_grad():
                start = time.time()
                model.fake_img_B = model.netG(model.real_img_A, real_map_AB)
                runtimes.append(time.time() - start)
            
            for i in range(model.fake_img_B.size(0)):
                if ckpt_id == ckpt_ids[0]:
                    real_img_B = torchvision.utils.make_grid(model.real_img_B[i].detach().cpu(), nrow=1, padding=0, normalize=True)
                    out_transform(real_img_B).save(os.path.join(real_dir, f'{n_images}@{data["fidB"][i]}.jpg'))
                fake_img_B = torchvision.utils.make_grid(model.fake_img_B[i].detach().cpu(), nrow=1, padding=0, normalize=True)
                out_transform(fake_img_B).save(os.path.join(fake_dir, f'{n_images}@{data["fidB"][i]}.jpg'))
                
                n_images += 1
            
            outrate = n_images / sum(runtimes) if sum(runtimes) > 0 else float('inf')
            print(f'\r[TEST] Iter: {ckpt_id:{w_iters}d} | Batch: {batch+1:{w_batch}d}/{n_batch}',
                  f'@ {outrate:.2f} img/sec [{"AB".replace(target, "")} -> {target}]', end='')
        print('')
