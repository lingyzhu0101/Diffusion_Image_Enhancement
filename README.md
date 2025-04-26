# [TMM'25] Diffusion_Image_Enhancement
Official Pytorch implementation of **From Understanding to Enhancement: Progressing Generative Low-Light Image Enhancement via Context-Aware Understanding Technique**. [Xiangrui Zeng]() and [Lingyu Zhu](https://scholar.google.com/citations?user=IhyTEDkAAAAJ&hl=zh-CN) contribute equally.

[Xiangrui Zeng](),
[Lingyu Zhu](https://scholar.google.com/citations?user=IhyTEDkAAAAJ&hl=zh-CN),
[Wenhan Yang](https://scholar.google.com/citations?user=S8nAnakAAAAJ&hl=zh-CN),
[Shiqi Wang](https://scholar.google.com.tw/citations?user=Pr7s2VUAAAAJ&hl=en)
[Sam Kwong](https://scholar.google.com.tw/citations?user=_PVI6EAAAAAJ&hl=en)

[[`Arxiv`](http://arxiv.org/abs/)] [[`Supplementary Material`]()]  [[`Video`]()] 

## Overview
With the emergence of the diffusion model, there have been few efforts to utilize the fine-grained understanding knowledge to enhance the low-light images in a way that aligns with human perception through generative technique. However, the challenge lies in balancing the effective injection of fine-grained understanding to capture intricate details while simultaneously leveraging the generative technique to achieve visually appealing results. To achieve this, we make the efforts from three perspectives. Firstly, we propose a novel Context-Aware Understanding Guided Diffusion model (CUGD) for low-light image enhancement. This method utilizes diffusion techniques to model normal-light image distribution by incorporating a context-aware understanding of knowledge, which includes both high-level semantic and low-level structural information. Secondly, we design a Context  Understanding Injection Encoder (CUIE) module that combines self-attention and cross-attention mechanisms to comprehensively integrate semantic and structural information into enhanced results, thus facilitating a fine-grained understanding and enhancement process. This module serves the diffusion model in generating normal-light images with richer and more semantically consistent details.  Lastly, the semantic context regularization loss is introduced into the optimization process, ensuring that the recovered context better aligns with normal-light semantic distribution. Comprehensive experiments across multiple datasets demonstrate that the proposed method achieves state-of-the-art (SOTA) performance. Additionally, more evaluations on several no-reference datasets highlight the strong generalization capability of the proposed method.

## TODO List
This repository is still under active construction:
- [ ] Release training and testing codes
- [ ] Release pretrained models
- [ ] Clean the code

## Public Dataset


## Installation

## Contact

- Xiangrui Zeng: xiazeng9-c@my.cityu.edu.hk
- Lingyu Zhu: lingyzhu-c@my.cityu.edu.hk

## Citation

If you find our work helpful, please consider citing:



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
