# [TMM'25] Diffusion_Image_Enhancement
Official Pytorch implementation of **Bootstrap Low-light Image Enhancement Diffusion Model Through Context-centric Prior**. [Xiangrui Zeng]() and [Lingyu Zhu](https://scholar.google.com/citations?user=IhyTEDkAAAAJ&hl=zh-CN) contribute equally.

[Xiangrui Zeng](),
[Lingyu Zhu](https://scholar.google.com/citations?user=IhyTEDkAAAAJ&hl=zh-CN),
[Wenhan Yang](https://scholar.google.com/citations?user=S8nAnakAAAAJ&hl=zh-CN),
[Peilin Chen](https://scholar.google.com.tw/citations?user=b9k152sAAAAJ&hl=en),
[Baoliang Chen](https://scholar.google.com.tw/citations?user=w_WL27oAAAAJ&hl=en),
[Hanwei Zhu](https://scholar.google.com.tw/citations?user=-52izjkAAAAJ&hl=en),
[Shiqi Wang](https://scholar.google.com.tw/citations?user=Pr7s2VUAAAAJ&hl=en)
[Sam Kwong](https://scholar.google.com.tw/citations?user=_PVI6EAAAAAJ&hl=en)

[[`Arxiv`](http://arxiv.org/abs/)] [[`Supplementary Material`]()]  [[`Video`]()] 

## Overview
With the emergence of diffusion models, there have been few efforts to utilize the context-centric prior knowledge embedded in the Segment Anything Model (SAM) to enhance the low-light images. However, a major challenge remains in bridging the gap between context-centric prior knowledge and the generative model for low-light image enhancement. To address this challenge, we propose a novel SAM Prior-Guided Diffusion model (SPGD) for low-light image enhancement. This paradigm utilizes diffusion techniques to model normal image distributions while under the regularizing of the context-centric priors that encompass high-level semantic and low-level structural information from the SAM. Specifically, we design a SAM Prior Encoder (SPE) module that incorporates the context-centric prior, which integrates high-level and low-level details using an attention mechanism. This integration serves the diffusion model in generating normal-light images with richer and more semantically consistent details. Furthermore, the semantic context regularization loss integrates SAM semantic maps into the optimization process, ensuring that the recovered context better aligns with normal-light semantics. Comprehensive experiments across multiple datasets demonstrate that the proposed method achieves state-of-the-art (SOTA) performance.  Additionally, more evaluations on several no-reference datasets highlight the strong generalization capability of the proposed method.

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
