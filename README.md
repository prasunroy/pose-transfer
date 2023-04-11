### Official code for Multi-scale Attention Guided Pose Transfer.

*Accepted in Pattern Recognition (PR) 2023.*

[![badge_torch](https://img.shields.io/badge/made_with-PyTorch_2.0-EE4C2C?style=flat-square&logo=PyTorch)](https://pytorch.org/)
[![badge_arxiv](https://img.shields.io/badge/arXiv-2202.06777-brightgreen?style=flat-square)](https://arxiv.org/abs/2202.06777)

<br>

![network_architecture](https://user-images.githubusercontent.com/38404108/153903271-2a1e7faf-1bc6-4e73-811e-6fcd5c5b58a6.png)

<br>

![results](https://user-images.githubusercontent.com/38404108/153917804-2788e6d8-ffed-4aa7-b097-08bb2335a624.png)

<br>

### :zap: Getting Started
```bash
mkdir pose2pose
cd pose2pose
mkdir -p datasets/DeepFashion
mkdir -p output/DeepFashion/ckpt/pretrained
git clone https://github.com/prasunroy/pose-transfer.git
cd pose-transfer
pip install -r requirements.txt
```

<br>

### :fire: Quick test using the inference API
```python
from api import Pose2Pose
from PIL import Image

p2p = Pose2Pose(pretrained=True)

condition = Image.open('./api-test/condition.jpg')
reference = Image.open('./api-test/target_pose_reference.jpg')
generated = p2p.transfer_as(condition, reference)
generated.show()
```

<br>

### Code organization for training, testing and evaluation
* Download dataset files from [Google Drive](https://drive.google.com/drive/folders/11jM3r2kZHpO5O6TPOLsirz5W3XfPvZib) and extract into `datasets/DeepFashion` directory.
* Download pretrained checkpoints from [Google Drive](https://drive.google.com/drive/folders/1SDSEfWyP5ZFR8nA-zQLhwjBsRm7ggfWj) into `output/DeepFashion/ckpt/pretrained` directory.
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

<br>

### External Links
<h4>
  <a href="https://arxiv.org/abs/2202.06777">arXiv</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://drive.google.com/drive/folders/11jM3r2kZHpO5O6TPOLsirz5W3XfPvZib">Dataset</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://drive.google.com/drive/folders/1SDSEfWyP5ZFR8nA-zQLhwjBsRm7ggfWj">Pretrained Models</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://drive.google.com/uc?export=download&id=1Y9MCw0liv38LcR2ShGATKVlmd4EUP3Jo">Images for User Study</a>
</h4>

<br>

### Citation
```
@article{roy2022multi,
  title   = {Multi-scale Attention Guided Pose Transfer},
  author  = {Roy, Prasun and Bhattacharya, Saumik and Ghosh, Subhankar and Pal, Umapada},
  journal = {Pattern Recognition},
  volume  = {137},
  pages   = {109315},
  year    = {2023},
  issn    = {0031-3203},
  doi     = {https://doi.org/10.1016/j.patcog.2023.109315}
}
```

<br>

### License
```
Copyright 2023 by the authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

>The pretrained models are released under Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) license.

<br>

##### Made with :heart: and :pizza: on Earth.
