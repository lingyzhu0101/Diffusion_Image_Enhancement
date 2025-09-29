# [TCSVT'25] Diffusion_Image_Enhancement
Official Pytorch implementation of **Low-Light Image Enhancement via Diffusion Models with Semantic Priors of Any Region**. [Xiangrui Zeng]() and [Lingyu Zhu](https://scholar.google.com/citations?user=IhyTEDkAAAAJ&hl=zh-CN) contribute equally.

[Xiangrui Zeng](),
[Lingyu Zhu](https://scholar.google.com/citations?user=IhyTEDkAAAAJ&hl=zh-CN),
[Wenhan Yang](https://scholar.google.com/citations?user=S8nAnakAAAAJ&hl=zh-CN),
[Howard Leung](https://scholar.google.com/citations?user=EPaaCOgAAAAJ&hl=en),
[Shiqi Wang](https://scholar.google.com.tw/citations?user=Pr7s2VUAAAAJ&hl=en)
[Sam Kwong](https://scholar.google.com.tw/citations?user=_PVI6EAAAAAJ&hl=en)

[[`Arxiv`](http://arxiv.org/abs/)] [[`Supplementary Material`]()]  [[`Video`]()] 

## Overview
With the emergence of the diffusion model, its powerful regression capabilities have significantly boosted the performance for low-light image enhancement. However, the inherent information loss in low-light conditions calls for a deep understanding of scene semantics and structures to effectively recover missing content. Recent advances such as the Segment Anything Model (SAM) provide semantic priors for arbitrary regions through prompt-based object segmentation, which offers rich contextual cues to guide the restoration process. Motivated by this, we propose to incorporate such semantics-aware priors into a generative diffusion framework from three perspectives. This method utilizes the diffusion technique to model the distribution of images by incorporating contextually aware semantic and structural information for any region. Specifically, regional priors provided by SAM are integrated to guide the diffusion process with awareness of any object or region, enhancing the model’s capability to reason about scene content. Secondly, we design a \textbf{C}ontext  \textbf{U}nderstanding \textbf{I}njection \textbf{E}ncoder (\textbf{CUIE}) module that combines self-attention and cross-attention mechanisms to comprehensively integrate semantic and structural information into enhanced results, thus facilitating a fine-grained understanding and enhancement process. This module serves the diffusion model in generating normal-light images with richer and more semantically consistent details. Lastly, the semantic context regularization loss is introduced into the optimization process, ensuring that the recovered context better aligns with the normal-light semantic distribution. Extensive experiments on various datasets show that the proposed method attains state-of-the-art (SOTA) performance in both full-reference and no-reference evaluation measures. 

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
