# [TCSVT'25] Diffusion_Image_Enhancement
Official Pytorch implementation of **Low-Light Image Enhancement via Diffusion Models with Semantic Priors of Any Region**. [Xiangrui Zeng]() and [Lingyu Zhu](https://scholar.google.com/citations?user=IhyTEDkAAAAJ&hl=zh-CN) contribute equally.

[Xiangrui Zeng](),
[Lingyu Zhu](https://scholar.google.com/citations?user=IhyTEDkAAAAJ&hl=zh-CN),
[Wenhan Yang](https://scholar.google.com/citations?user=S8nAnakAAAAJ&hl=zh-CN),
[Howard Leung](https://scholar.google.com/citations?user=EPaaCOgAAAAJ&hl=en),
[Shiqi Wang](https://scholar.google.com.tw/citations?user=Pr7s2VUAAAAJ&hl=en),
[Sam Kwong](https://scholar.google.com.tw/citations?user=_PVI6EAAAAAJ&hl=en)

[[`Paper`](https://ieeexplore.ieee.org/document/11192523)]

## Overview
With the emergence of the diffusion model, its powerful regression capabilities have significantly boosted the performance for low-light image enhancement. However, the inherent information loss in low-light conditions calls for a deep understanding of scene semantics and structures to effectively recover missing content. Recent advances such as the Segment Anything Model (SAM) provide semantic priors for arbitrary regions through prompt-based object segmentation, which offers rich contextual cues to guide the restoration process. Motivated by this, we propose to incorporate such semantics-aware priors into a generative diffusion framework from three perspectives. This method utilizes the diffusion technique to model the distribution of images by incorporating contextually aware semantic and structural information for any region. Specifically, regional priors provided by SAM are integrated to guide the diffusion process with awareness of any object or region, enhancing the model’s capability to reason about scene content. Secondly, we design a Context Understanding Injection Encoder (CUIE) module that combines self-attention and cross-attention mechanisms to comprehensively integrate semantic and structural information into enhanced results, thus facilitating a fine-grained understanding and enhancement process. This module serves the diffusion model in generating normal-light images with richer and more semantically consistent details. Lastly, the semantic context regularization loss is introduced into the optimization process, ensuring that the recovered context better aligns with the normal-light semantic distribution. Extensive experiments on various datasets show that the proposed method attains state-of-the-art (SOTA) performance in both full-reference and no-reference evaluation measures. 

## TODO List
This repository is still under active construction:
- [x] Release testing codes
- [x] Release pretrained models
- [x] Clean the code
- [ ] Release training codes

## Installation

### Environment Setup
```bash
conda create -n lldiffsam python=3.9
source activate lldiffsam
conda install pytorch==2.3.1 torchvision==0.18.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Install SAM (Segment Anything Model)
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Dataset Preparation

### Download Datasets

**LOLv2:**
- Baidu Pan: https://pan.baidu.com/s/1qEYMvf9cZUOJbSsEgJ_Dew?pwd=yswq (Code: yswq)
- Google Drive: https://drive.google.com/file/d/1PgxhXW4vA6daCTJO0KYNlqiH49_-S6qG/view?usp=sharing

**Adobe-5K:**
- Baidu Pan: https://pan.baidu.com/s/140WiyxKnPwYyq1ANZCnBfw?pwd=fxmk (Code: fxmk)
- Google Drive: https://drive.google.com/file/d/1fmppBFGQDGMZbK28fLCIW4UkTR-_m5LY/view?usp=sharing

Download the datasets, unzip them, and place in `./dataset` directory.

### Download Pretrained Checkpoints

**Checkpoints:**
- Baidu Pan: https://pan.baidu.com/s/1X3W9V6h6vbpFO0PLmDrV_w?pwd=x375 (Code: x375)
- Google Drive: https://drive.google.com/file/d/1zIviiBx5Mm85Bp31skyqFVDFHzM-Tz2G/view?usp=sharing

Download and unzip the checkpoints.

### SAM Checkpoints

Download SAM checkpoints for generating semantic priors:
- **vit_h (default):** https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
- **vit_l:** https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
- **vit_b:** https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

## Testing

### Test on Benchmark Datasets

Specify data directory in the dataset config YAML file (e.g., `config/lolv2_real_edgesemantic.yml`) and checkpoint directory in the test config JSON file (e.g., `config/lolv2_real_test.json`).

**LOLv2-Real:**
```bash
python test.py --dataset config/lolv2_real_edgesemantic.yml --config config/lolv2_real_test.json
```

**LOLv2-Synthetic:**
```bash
python test.py --dataset config/lolv2_syn_edgesemantic.yml --config config/lolv2_syn_test.json
```

**Adobe-5K:**
```bash
python test.py --dataset config/fivek.yml --config config/fivek_test.json
```

Results will be saved in `experiments/lolv2_real_test_<timestamp>/results/`, including:
- `output/`: Enhanced images
- `gt/`: Ground truth images
- `input/`: Input low-light images

### Inference on Your Own Data

#### Step 1: Organize Your Images
Put your dark images in a folder named `Low`. For example:
```
/home/you/data/mydark/Low
```

#### Step 2: Generate SAM Priors
```bash
python make_sam_prior.py --src /home/you/data/mydark/Low --dst /home/you/data/mydark/Low_sam --model_type vit_h --sam_checkpoint <path_to_SAM_checkpoint>
```

Note: `--model_type` is optional, default is `vit_h`.

#### Step 3: Create Dataset Configuration
Create a YAML file (e.g., `config/mydark.yml`) with the following content:

```yaml
dataset: Unpaired

datasets:
  val:
    dist: False
    root: /home/you/data
    n_workers: 1
    batch_size: 1
    sub_data: mydark
```

#### Step 4: Run Inference
```bash
python test_unpaired.py --dataset config/mydark.yml --config config/lolv2_syn_test.json --name mydark
```

Results will be saved in `experiments/mydark_<timestamp>/results/output/`.

## Contact

- Xiangrui Zeng: xiazeng9-c@my.cityu.edu.hk
- Lingyu Zhu: lingyzhu-c@my.cityu.edu.hk

## Citation

If you find our work helpful, please consider citing:

```bibtex
@ARTICLE{zeng2025diffsam,
  author={Zeng, Xiangrui and Zhu, Lingyu and Yang, Wenhan and Leung, Howard and Wang, Shiqi and Kwong, Sam},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  title={Low-Light Image Enhancement via Diffusion Models with Semantic Priors of Any Region},
  year={2025},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2025.3617320}}
```


## Additional Link

We also recommend our Temporally Consistent Enhancer Network [TCE-Net](https://github.com/lingyzhu0101/low-light-video-enhancement.git). If you find our work helpful, please consider citing:

```bibtex
@article{zhu2024temporally,
  title={Temporally Consistent Enhancement of Low-Light Videos via Spatial-Temporal Compatible Learning},
  author={Zhu, Lingyu and Yang, Wenhan and Chen, Baoliang and Zhu, Hanwei and Meng, Xiandong and Wang, Shiqi},
  journal={International Journal of Computer Vision},
  pages={1--21},
  year={2024},
  publisher={Springer}
}
```

```bibtex
@inproceedings{zhu2024unrolled,
  title={Unrolled Decomposed Unpaired Learning for Controllable Low-Light Video Enhancement},
  author={Lingyu Zhu, Wenhan Yang, Baoliang Chen, Hanwei Zhu, Zhangkai Ni, Qi Mao, and Shiqi Wang},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

## Acknowledgements

This code is built on [SR3 (Image Super-Resolution via Iterative Refinement)](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement). We thank the authors for their excellent work.
