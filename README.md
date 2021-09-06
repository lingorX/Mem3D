# Quality-Aware Memory Network for Interactive Volumetric Image Segmentation

<p align="center">
<img src="https://github.com/0liliulei/Mem3D/blob/main/arch.png" width="1000">
</p>

This repo contains the source code of our Quality-Aware Memory Network in MICCAI 2021. [[Paper](https://arxiv.org/abs/2106.10686)]

## Data

### Preparation
Download [MSD](http://medicaldecathlon.com/) and [KiTS](https://kits19.grand-challenge.org/data/). This repo provides dataloaders for MSD, you can some modification to adapt them to other datasets.

### Dataset Organization
To run the training and testing code, we require the following data organization format

    ${ROOT}--
            |--KiTS
            |--MSD
            │   ├── ImageSets06
            │   │   └── train.txt
            │   │   └── test.txt
            │   ├── ImageSest10
            │   ├── Task06_mask
            │   │   ├── lung_001
            │   │   │   ├── 0.png 
            │   │   │   ├── 1.png
            │   │   │   ├── 2.png
            │   │   │   ├── ...
            │   │   │   ├── ...
            │   │   │   └── 199.png
            │   │   ├── lung_002
            │   │   ├── lung_003
            │   │   ├── ...
            │   │   └── lung_060
            │   ├── Task06_origin
            │   │   ├── lung_001
            │   │   │   ├── 0.png 
            │   │   │   ├── ...
            │   │   │   └── 199.png
            │   │   ├── ...
            │   │   └── lung_060
            │   ├── ImageSets10
            │   ├── Task10_mask
            │   └── Task10_origin
            └──${DATASET3}


### Organization

To run the training script, please ensure the data is organized as following format

    ${ROOT}/${DATASET1}

## Usage
### 2D Interactive Network

    Mem3D/
    └── (train/test)_(dextr/hybrid/inter/scribble/two_point).py
    
### Memory Network

    Mem3D/
    ├── (train/test)_STM.py.  # without Quality Assessment
    └── train_SAQ.py          # with Quality Assessment

### Round Based 3D Interactive Segmentation

    Mem3D/
    ├── eval_SAQ.py               # w QA
    └── eval_IOG_refine_dextr.py  # w/o QA

### Volume-wise Dice Evaluation

    Mem3D/
    └── eval.py

### pretrained weight

- Download the [weight](https://drive.google.com/file/d/1nzhFYOJx3rzvnO8o6g-D1MMA6iQ4VYpw/) pretrained on Youtube-vos for Memory Network
- Update the `initial` attribution in `option.py`

## Acknowledgements
- This code is based on [DEXTR](https://github.com/scaelles/DEXTR-PyTorch) and [STM](https://github.com/lyxok1/STM-Training).

## Citation
```
@article{zhou2021quality,
  title={Quality-Aware Memory Network for Interactive Volumetric Image Segmentation},
  author={Zhou, Tianfei and Li, Liulei and Bredell, Gustav and Li, Jianwu and Konukoglu, Ender},
  journal={arXiv preprint arXiv:2106.10686},
  year={2021}
}
```
