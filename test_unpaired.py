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
import json

import options.options as option
from utils import util
from data import create_dataloader
from data.LoL_dataset import LOLv1_Dataset, LOLv2_Dataset, LOLv2_EdgeSemantic_Dataset,Fivek_EdgeSemantic_Dataset, Unpaired_EdgeSemantic_Dataset
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
    parser.add_argument('--dataset', type=str, default='config/unpaired.yml')
    parser.add_argument('--name', type=str, default='unpaired-test') 
    parser.add_argument('-c', '--config', type=str, default='config/lolv2_syn_test.json')
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

    if args.name is not None:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            config_data['name'] = args.name
        temp_config = args.config.replace('.json', '_temp.json')
        with open(temp_config, 'w') as f:
            json.dump(config_data, f, indent=4)
        args.config = temp_config
        cleanup_temp = True
    else:
        cleanup_temp = False

    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    opt_dataset = option.parse(args.dataset, is_train=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    opt['phase'] = 'test'
    #### distributed training settings
    opt['dist'] = False
    rank = -1
    print('Disabled distributed training.')

    if rank <= 0:
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))

    opt = option.dict_to_nonedict(opt)

    seed = opt['seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True

    if opt_dataset['dataset'] == 'LOLv1':
        dataset_cls = LOLv1_Dataset
    elif opt_dataset['dataset'] == 'LOLv2':
        dataset_cls = LOLv2_Dataset
    elif opt_dataset['dataset'] == 'LOLv2EdgeSemantic':
        dataset_cls = LOLv2_EdgeSemantic_Dataset
    elif opt_dataset['dataset'] == 'FivekEdgeSemantic':
        dataset_cls = Fivek_EdgeSemantic_Dataset
    else:
        dataset_cls = Unpaired_EdgeSemantic_Dataset

    for phase, dataset_opt in opt_dataset['datasets'].items():
        if phase == 'val':
            val_set = dataset_cls(opt=dataset_opt, train=False, all_opt=opt_dataset)
            val_loader = create_dataloader(val_set, dataset_opt, opt_dataset, None)

    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['train'], schedule_phase='train')

    result_path = '{}'.format(opt['path']['results'])
    result_path_out = result_path + '/output/'
    result_path_input = result_path + '/input/'
    os.makedirs(result_path_out, exist_ok=True)
    os.makedirs(result_path_input, exist_ok=True)

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    logger_val = logging.getLogger('val')

    for val_data in val_loader:
        diffusion.feed_data(val_data)
        diffusion.test(continous=False)

        visuals = diffusion.get_current_visuals()

        normal_img = Metrics.tensor2img(visuals['HQ'])
        normal_img = normal_img[0:val_data['h'], 0:val_data['w'], :]
        ll_img = Metrics.tensor2img(visuals['LQ'])
        f_name = val_data['GT_path'][0]
        print('processing {}'.format(f_name))
        util.save_img(
            ll_img, '{}/{}.png'.format(result_path_input, f_name))
        util.save_img(normal_img, '{}/{}.png'.format(result_path_out, f_name))

    logger_val.info('Processing completed. Results saved to {}'.format(result_path_out))

    if cleanup_temp:
        os.remove(temp_config)

if __name__ == '__main__':
    main()
