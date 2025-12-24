import os
from os.path import basename
import math
import argparse
import random
import logging
import cv2
import sys
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import options.options as option
from utils import util
from data import create_dataloader
from data.LoL_dataset import LOLv1_Dataset, LOLv2_Dataset, LOLv2_EdgeSemantic_Dataset,Fivek_EdgeSemantic_Dataset
import torchvision.transforms as T
import lpips
import model as Model
import core.logger as Logger
import core.metrics as Metrics
from torchvision import transforms
from metrics import *
transform = transforms.Lambda(lambda t: (t * 2) - 1)


def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='config/lolv2_real_edgesemantic.yml') 
    parser.add_argument('-c', '--config', type=str, default='config/lolv2_real_test.json')
    # parser.add_argument('--dataset', type=str, default='config/lolv2_syn_edgesemantic.yml') 
    # parser.add_argument('-c', '--config', type=str, default='config/lolv2_syn_test_ca_ddimtrain.json')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--tfboard', action='store_true')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="0")
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    opt_dataset = option.parse(args.dataset, is_train=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    opt['phase'] = 'test'

    #### distributed training settings
    opt['dist'] = False
    rank = -1
    print('Disabled distributed training.')

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### seed
    seed = opt['seed']
    if seed is None:
        seed = random.randint(1, 10000)
    # seed = random.randint(1, 1000000)
    if rank <= 0:
        logger.info('Seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    if opt_dataset['dataset'] == 'LOLv1':
        dataset_cls = LOLv1_Dataset
    elif opt_dataset['dataset'] == 'LOLv2':
        dataset_cls = LOLv2_Dataset

    elif opt_dataset['dataset'] == 'LOLv2EdgeSemantic':
        dataset_cls = LOLv2_EdgeSemantic_Dataset
    elif opt_dataset['dataset'] == 'FivekEdgeSemantic':
        dataset_cls = Fivek_EdgeSemantic_Dataset
    else:
        raise NotImplementedError()

    for phase, dataset_opt in opt_dataset['datasets'].items():
        if phase == 'val':
            val_set = dataset_cls(opt=dataset_opt, train=False, all_opt=opt_dataset)
            val_loader = create_dataloader(val_set, dataset_opt, opt_dataset, None)

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['train'], schedule_phase='train')

    result_path = '{}'.format(opt['path']['results'])
    result_path_gt = result_path + '/gt/'
    result_path_out = result_path + '/output/'
    result_path_input = result_path + '/input/'
    os.makedirs(result_path_gt, exist_ok=True)
    os.makedirs(result_path_out, exist_ok=True)
    os.makedirs(result_path_input, exist_ok=True)

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    logger_val = logging.getLogger('val')  # validation logger

    for val_data in val_loader:
        diffusion.feed_data(val_data)
        diffusion.test(continous=False)

        visuals = diffusion.get_current_visuals()

        normal_img = Metrics.tensor2img(visuals['HQ'])
        normal_img = normal_img[0:val_data['h'], 0:val_data['w'], :]
        gt_img = Metrics.tensor2img(visuals['GT'])
        ll_img = Metrics.tensor2img(visuals['LQ'])
        f_name = val_data['GT_path'][0]
        print('processing {}'.format(f_name))
        util.save_img(
            gt_img, '{}/{}.png'.format(result_path_gt, f_name))
        util.save_img(
            ll_img, '{}/{}.png'.format(result_path_input, f_name))
        util.save_img(normal_img, '{}/{}.png'.format(result_path_out, f_name))

    psnr_ssim_dists = Metric(result_path_out, result_path_gt, result_path_out)
    lpips_alex = lpips_distance_folders(result_path_out, result_path_gt, result_path_out, 'lpips_alex.txt',net ='alex')
    try:
        fid = compute_fid(result_path_gt,result_path_out)
    except:
        fid = compute_fid(result_path_gt,result_path_out,batch_size = 1)
    # log
    logger_val = logging.getLogger('val')
    logger_val.info('PSNR:{:.4f} SSIM:{:.4f} DISTS:{:.4f} LPIPS_Alex:{:.4f} FID:{:.4f}'.format(psnr_ssim_dists['psnr'],psnr_ssim_dists['ssim'],psnr_ssim_dists['dists'],lpips_alex,fid))

if __name__ == '__main__':
    main()
