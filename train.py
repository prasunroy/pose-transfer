# imports
import datetime
import time
import torchvision
from torch.utils.tensorboard import SummaryWriter
from data.dataloader import create_dataloader
from models.pose_transfer_model import PoseTransferModel


# configurations
# -----------------------------------------------------------------------------
dataset_name = 'DeepFashion'

dataset_root = f'../datasets/{dataset_name}'
img_pairs_train = f'{dataset_root}/train_img_pairs.csv'
img_pairs_test = f'{dataset_root}/test_img_pairs.csv'
pose_maps_dir_train = f'{dataset_root}/train_pose_maps'
pose_maps_dir_test = f'{dataset_root}/test_pose_maps'

gpu_ids = [0]

batch_size_train = 8
batch_size_test = 8
n_epoch = 100
out_freq = 500

ckpt_id = None
ckpt_dir = None

run_info = ''
out_path = f'../output/{dataset_name}'
# -----------------------------------------------------------------------------


# create timestamp and infostamp
timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
infostamp = f'_{run_info.strip()}' if run_info.strip() else ''

# create tensorboard logger
logger = SummaryWriter(f'{out_path}/runs/{timestamp}{infostamp}')

# create transforms
img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
map_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# create dataloaders
train_dataloader = create_dataloader(dataset_root, img_pairs_train, pose_maps_dir_train,
                                     img_transform, map_transform,
                                     batch_size=batch_size_train, shuffle=True)
test_dataloader = create_dataloader(dataset_root, img_pairs_test, pose_maps_dir_test,
                                    img_transform, map_transform,
                                    batch_size=batch_size_test, shuffle=False)

# create fixed batch for testing
fixed_test_batch = next(iter(test_dataloader))

# create model
model = PoseTransferModel(gpuids=gpu_ids)
model.print_networks(verbose=False)

# load pretrained weights into model
if ckpt_id and ckpt_dir:
    model.load_networks(ckpt_dir, ckpt_id, verbose=True)

# train model
n_batch = len(train_dataloader)
w_batch = len(str(n_batch))
w_epoch = len(str(n_epoch))
n_iters = 0

for epoch in range(n_epoch):
    for batch, data in enumerate(train_dataloader):
        time_0 = time.time()
        model.set_inputs(data)
        model.optimize_parameters()
        losses = model.get_losses()
        loss_G = losses['lossG']
        loss_D = losses['lossD']
        time_1 = time.time()
        print(f'[TRAIN] Epoch: {epoch+1:{w_epoch}d}/{n_epoch} | Batch: {batch+1:{w_batch}d}/{n_batch} |',
              f'LossG: {loss_G:7.4f} | LossD: {loss_D:7.4f} | Time: {round(time_1-time_0, 2):.2f} sec |')
        
        if (n_iters % out_freq == 0) or (batch+1 == n_batch and epoch+1 == n_epoch):
            model.save_networks(f'{out_path}/ckpt/{timestamp}{infostamp}', n_iters, verbose=True)
            for loss_name, loss in losses.items():
                loss_group = 'LossG' if loss_name.startswith('lossG') else 'LossD'
                logger.add_scalar(f'{loss_group}/{loss_name}', loss, n_iters)
            model.set_inputs(fixed_test_batch)
            visuals = model.compute_visuals()
            logger.add_image(f'Iteration_{n_iters}', visuals, n_iters)
        
        n_iters += 1
