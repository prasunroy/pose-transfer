# [Multi-scale Attention Guided Pose Transfer](https://arxiv.org/abs/2202.06777)

#### *-- To be updated --*

<br>

![network_architecture](https://user-images.githubusercontent.com/38404108/153903271-2a1e7faf-1bc6-4e73-811e-6fcd5c5b58a6.png)

<br>

![results](https://user-images.githubusercontent.com/38404108/153917804-2788e6d8-ffed-4aa7-b097-08bb2335a624.png)

<br>

## Getting Started
```bash
mkdir pose2pose
cd pose2pose
mkdir -p datasets/DeepFashion
mkdir -p output/DeepFashion/ckpt/pretrained
git clone https://github.com/prasunroy/pose-transfer.git
cd pose-transfer
```
* Download dataset files from [Google Drive](https://drive.google.com/drive/folders/11jM3r2kZHpO5O6TPOLsirz5W3XfPvZib) and extract into `datasets/DeepFashion` directory
* Download pretrained checkpoints from [Google Drive](https://drive.google.com/drive/folders/1SDSEfWyP5ZFR8nA-zQLhwjBsRm7ggfWj) into `output/DeepFashion/ckpt/pretrained` directory
```
pose2pose
│
├───datasets
│   └───DeepFashion
│       ├───img
│       ├───test_pose_maps
│       ├───train_pose_maps
│       ├───test_img_keypoints.csv
│       ├───test_img_list.csv
│       ├───test_img_pairs.csv
│       ├───train_img_keypoints.csv
│       ├───train_img_list.csv
│       └───train_img_pairs.csv
├───output
│   └───DeepFashion
│       └───ckpt
│           └───pretrained
│               ├───netD_257500.pth
│               ├───netD_260500.pth
│               ├───netG_257500.pth
│               └───netG_260500.pth
└───pose-transfer
```

> The precomputed keypoints and posemaps are estimated using the provided utility scripts in [pose-transfer/utils](https://github.com/prasunroy/pose-transfer/tree/main/utils).

> In the [paper](https://arxiv.org/pdf/2202.06777.pdf), all qualitative results are generated using the pretrained checkpoint at iteration **260500** and all quantitative evaluations are performed using the pretrained checkpoint at iteration **257500**.

## Citation
```
@article{roy2022multi,
  title   = {Multi-scale Attention Guided Pose Transfer},
  author  = {Roy, Prasun and Bhattacharya, Saumik and Ghosh, Subhankar and Pal, Umapada},
  journal = {arXiv preprint arXiv:2202.06777},
  year    = {2022}
}
```

##### Made with :heart: and :pizza: on Earth.
