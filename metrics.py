import numpy as np
import cv2
import torch
import os
import glob
import copy
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from utils.util import imgdir_list,imgcv2tensor
from torchvision.transforms.functional import to_tensor
from DISTS_pytorch import DISTS
from pytorch_fid import fid_score

def compute_fid(real_images_path, generated_images_path, batch_size=50, device=None):
    """
    Computes the Frechet Inception Distance (FID) between two sets of images.

    Parameters:
        real_images_path (str): Path to the directory containing real images.
        generated_images_path (str): Path to the directory containing generated images.
        batch_size (int): Batch size for processing images. Default is 50.
        device (torch.device or None): Device to use for computation. Defaults to CUDA if available, otherwise CPU.

    Returns:
        float: The computed FID score.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fid_value = fid_score.calculate_fid_given_paths(
        [real_images_path, generated_images_path],
        batch_size=batch_size,
        device=device,
        dims=2048,
        num_workers=0
    )

    return fid_value


def write_dicts_to_file_withstr(m, filename):
    with open(filename, 'w') as file:
        for dictionary in m:
            line = ''
            for key, value in dictionary.items():
                if isinstance(value, str):
                    line += f"{key}: {value}, "
                elif isinstance(value, int) or isinstance(value, float):
                    line += f"{key}: {value}, "
            file.write(line.rstrip(', ') + '\n')

        # Calculate average of numeric values
        numeric_values = []
        for dictionary in m:
            for value in dictionary.values():
                if isinstance(value, int) or isinstance(value, float):
                    numeric_values.append(value)
        average = 0
        if numeric_values:
            average = sum(numeric_values) / len(numeric_values)
            file.write(f"Average: {average}")
        return average


def lpips_distance(im1, im2, loss_fn,use_gpu=True):
    if (use_gpu):
        loss_fn.cuda()
    # Load images
    img0 = lpips.im2tensor(lpips.load_image(im1))  # RGB image from [-1,1]
    img1 = lpips.im2tensor(lpips.load_image(im2))

    if (use_gpu):
        img0 = img0.cuda()
        img1 = img1.cuda()
    # Compute distance
    dist01 = loss_fn.forward(img0, img1)
    # print('Distance: %.3f' % dist01)
    return dist01


def lpips_distance_folders(pred_root, gt_root, outfolder, outname, use_gpu=True,net = 'vgg'):
    pred_dirs = imgdir_list(pred_root)
    lines = []
    loss_fn = lpips.LPIPS(net=net)
    for pred_dir in pred_dirs:
        pred_name = os.path.basename(pred_dir)
        # if 'low' in pred_name:
        #     gt_dir = os.path.join(gt_root, pred_name.replace("low", "normal"))
        # else:
        #     gt_dir = os.path.join(gt_root, pred_name)
        gt_dir = os.path.join(gt_root, pred_name)
        distance = lpips_distance(pred_dir, gt_dir, loss_fn,use_gpu).cpu().item()
        lines.append({'name': pred_name, 'distance': distance})
    # print(lines)
    average = write_dicts_to_file_withstr(lines, os.path.join(outfolder, outname))
    return average


def write_dict_list_to_file(dict_list, filename):
    with open(filename, 'w') as file:
        for dictionary in dict_list:
            line = ','.join([f'{key}:{value}' for key, value in dictionary.items()])
            file.write(line + '\n')

        avg_dict = {}
        num_dicts = len(dict_list)
        if num_dicts > 0:
            for key in dict_list[0].keys():
                values = [dictionary[key] for dictionary in dict_list]
                avg_dict[key] = sum(values) / num_dicts
            avg_line = ','.join([f'{key}:{value}' for key, value in avg_dict.items()])
            file.write(avg_line)
        return avg_dict


def Metric(pred_root, gt_goot, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    pred_dirs = imgdir_list(pred_root)
    metrics = []
    D = DISTS().cuda()
    for pred_dir in pred_dirs:
        dic1 = {}
        pred_img = cv2.imread(pred_dir)
        img_name = os.path.basename(pred_dir)
        # if 'low' in img_name:
        #     gt_dir = os.path.join(gt_goot, img_name.replace('low', 'normal'))
        # else:
        #     gt_dir = os.path.join(gt_goot, img_name)
        gt_dir = os.path.join(gt_goot, img_name)
        gt_img = cv2.imread(gt_dir)

        gray_gt = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
        gray_pred = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
        dic1['ssim'] = ssim(gray_gt, gray_pred)
        dic1['psnr'] = psnr(pred_img, gt_img)
        gt_tensor = imgcv2tensor(gt_img).unsqueeze(0).cuda()
        pred_tensor = imgcv2tensor(pred_img).unsqueeze(0).cuda()
        with torch.no_grad():
            dists_value = D(gt_tensor, pred_tensor)
            dic1['dists'] = dists_value.item()
        metrics.append(dic1)
    avg = write_dict_list_to_file(metrics, os.path.join(save_dir, 'psnr_ssim_dsits.txt'))
    return avg


if __name__ == '__main__':
    predv2_dir = 'experiments/lolv2real_train_ca_ddimtrain_lossmask-sam_consistency_q2q1_250813_032550/results/24/output'
    gtv2_dir = 'experiments/lolv2real_train_ca_ddimtrain_lossmask-sam_consistency_q2q1_250813_032550/results/24/gt'
    print(Metric(predv2_dir, gtv2_dir, predv2_dir))
    lpips_v = lpips_distance_folders(predv2_dir, gtv2_dir, predv2_dir, 'lpips_alex.txt',net ='alex')
    print(lpips_v)
