import os
import torch.utils.data as data
import numpy as np
import torch
import cv2
from torchvision.transforms import ToTensor
import torchvision.transforms as T
import torchvision
import glob

from data.data_augment import PairCompose, PairRandomCrop, PairToTensor, RandomNoise, GroupCompose, GroupRandomCrop, \
    GroupToTensor, GroupRandomHorizontalFilp, GroupRandomVerticalFlip


class LOLv1_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.crop_size = opt.get("patch_size", None)
        if train:
            self.split = 'train'
            self.root = os.path.join(self.root, 'our485')
        else:
            self.split = 'val'
            self.root = os.path.join(self.root, 'eval15')
        self.pairs = self.load_pairs(self.root)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):

        low_list = os.listdir(os.path.join(folder_path, 'low'))
        low_list = filter(lambda x: 'png' in x, low_list)

        pairs = []
        for idx, f_name in enumerate(low_list):

            if self.split == 'val':
                pairs.append(
                    [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB),
                     cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name)), cv2.COLOR_BGR2RGB),
                     f_name.split('.')[0]])
            else:
                pairs.append(
                    [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB),
                     cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name)), cv2.COLOR_BGR2RGB),
                     f_name.split('.')[0]])
        return pairs

    def __getitem__(self, item):
        lr, hr, f_name = self.pairs[item]

        if self.use_crop and self.split != 'val':
            hr, lr = random_crop(hr, lr, self.crop_size)
        elif self.split == 'val':
            lr = cv2.copyMakeBorder(lr, 8, 8, 4, 4, cv2.BORDER_REFLECT)

        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)

        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        [lr, hr] = transform_augment(
            [lr, hr], split=self.split, min_max=(-1, 1))

        return {'LQ': lr, 'GT': hr, 'LQ_path': f_name, 'GT_path': f_name}


class LOLv2_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.crop_size = opt.get("patch_size", None)
        self.sub_data = opt.get("sub_data", None)
        self.pairs = []
        self.train = train
        if train:
            self.split = 'train'
            root = os.path.join(self.root, self.sub_data, 'Train')
        else:
            self.split = 'val'
            root = os.path.join(self.root, self.sub_data, 'Test')
        self.pairs.extend(self.load_pairs(root))
        self.to_tensor = ToTensor()
        self.gamma_aug = opt['gamma_aug'] if 'gamma_aug' in opt.keys() else False

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):

        low_list = os.listdir(os.path.join(folder_path, 'Low' if self.train else 'Low'))
        low_list = sorted(list(filter(lambda x: 'png' in x, low_list)))
        high_list = os.listdir(os.path.join(folder_path, 'Normal' if self.train else 'Normal'))
        high_list = sorted(list(filter(lambda x: 'png' in x, high_list)))
        pairs = []

        for idx in range(len(low_list)):
            f_name_low = low_list[idx]
            f_name_high = high_list[idx]
            pairs.append(
                [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'Low' if self.train else 'Low', f_name_low)),
                              cv2.COLOR_BGR2RGB),
                 cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'Normal' if self.train else 'Normal', f_name_high)),
                              cv2.COLOR_BGR2RGB),
                 f_name_high.split('.')[0]])
        return pairs

    def __getitem__(self, item):

        lr, hr, f_name = self.pairs[item]
        h, w, c = lr.shape
        if self.use_crop and self.split != 'val':
            hr, lr = random_crop(hr, lr, self.crop_size)
        elif self.sub_data != 'syn1000' and self.split == 'val':  # for Real_captured
            pad_factor = 32
            bottom = int(np.ceil(h / pad_factor) * pad_factor - h)
            right = int(np.ceil(w / pad_factor) * pad_factor - w)
            lr = cv2.copyMakeBorder(lr, 0, bottom, 0, right, cv2.BORDER_REFLECT)

        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)

        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        [lr, hr] = transform_augment(
            [lr, hr], split=self.split, min_max=(-1, 1))

        return {'LQ': lr, 'GT': hr, 'LQ_path': f_name, 'GT_path': f_name, 'h': h, 'w': w}


class LOLv2_EdgeSemantic_Dataset(LOLv2_Dataset):
    def __init__(self, opt, train, all_opt):
        super(LOLv2_EdgeSemantic_Dataset, self).__init__(opt, train, all_opt)

    def load_pairs(self, folder_path):
        low_list = glob.glob(os.path.join(folder_path, 'Low', '*.png'))
        low_list.sort()

        high_list = glob.glob(os.path.join(folder_path, 'Normal', '*.png'))
        high_list.sort()

        low_edge_list = glob.glob(os.path.join(folder_path, 'Low_sam', '*edge.png'))
        low_edge_list.sort()

        low_semantic_list = glob.glob(os.path.join(folder_path, 'Low_sam', '*semantic.png'))
        low_semantic_list.sort()

        high_edge_list = glob.glob(os.path.join(folder_path, 'Normal_sam', '*edge.png'))
        high_edge_list.sort()

        high_semantic_list = glob.glob(os.path.join(folder_path, 'Normal_sam', '*semantic.png'))
        high_semantic_list.sort()

        pairs = []

        for idx in range(len(low_list)):
            name = os.path.basename(low_list[idx])
            f_name_low = low_list[idx]
            f_name_high = high_list[idx]
            f_name_low_edge = low_edge_list[idx]
            f_name_low_semantic = low_semantic_list[idx]
            f_name_high_edge = high_edge_list[idx]
            f_name_high_semantic = high_semantic_list[idx]
            pairs.append(
                [cv2.cvtColor(cv2.imread(f_name_low), cv2.COLOR_BGR2RGB),
                 cv2.cvtColor(cv2.imread(f_name_high), cv2.COLOR_BGR2RGB),
                 cv2.cvtColor(cv2.imread(f_name_low_edge), cv2.COLOR_BGR2RGB),
                 cv2.cvtColor(cv2.imread(f_name_low_semantic), cv2.COLOR_BGR2RGB),
                 cv2.cvtColor(cv2.imread(f_name_high_edge), cv2.COLOR_BGR2RGB),
                 cv2.cvtColor(cv2.imread(f_name_high_semantic), cv2.COLOR_BGR2RGB),
                 name.split('.')[0]
                 ])
        return pairs

    def __getitem__(self, item):

        lr, hr, le, ls, he, hs, f_name = self.pairs[item]
        h, w, c = lr.shape
        if self.use_crop and self.split != 'val':
            lr, hr, le, ls, he, hs = random_group_crop([lr, hr, le, ls, he, hs], self.crop_size)
        elif self.sub_data != 'syn1000' and self.split == 'val':  # for Real_captured
            # print("h, w, c",h,w)
            pad_factor = 32
            bottom = int(np.ceil(h / pad_factor) * pad_factor - h)
            right = int(np.ceil(w / pad_factor) * pad_factor - w)
            lr = cv2.copyMakeBorder(lr, 0, bottom, 0, right, cv2.BORDER_REFLECT)
            le = cv2.copyMakeBorder(le, 0, bottom, 0, right, cv2.BORDER_REFLECT)
            ls = cv2.copyMakeBorder(ls, 0, bottom, 0, right, cv2.BORDER_REFLECT)
            # print(lr.shape)

        if self.use_flip:
            lr, hr, le, ls, he, hs = random_group_flip([lr, hr, le, ls, he, hs])

        if self.use_rot:
            lr, hr, le, ls, he, hs = random_group_rotation([lr, hr, le, ls, he, hs])

        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)
        le = self.to_tensor(le)
        ls = self.to_tensor(ls)
        he = self.to_tensor(he)
        hs = self.to_tensor(hs)

        [lr, hr] = transform_augment(
            [lr, hr], split=self.split, min_max=(-1, 1))

        return {'LQ': lr, 'GT': hr, 'le': le, 'ls': ls, 'he': he, 'hs': hs, 'LQ_path': f_name, 'GT_path': f_name,'h': h, 'w': w }


class Fivek_EdgeSemantic_Dataset(LOLv2_EdgeSemantic_Dataset):
    def __init__(self, opt, train, all_opt):
        super(Fivek_EdgeSemantic_Dataset, self).__init__(opt, train, all_opt)

    def load_pairs(self, folder_path):
        low_list = glob.glob(os.path.join(folder_path, 'Low', '*.jpg'))
        low_list.sort()

        high_list = glob.glob(os.path.join(folder_path, 'Normal', '*.jpg'))
        high_list.sort()

        low_edge_list = glob.glob(os.path.join(folder_path, 'Low_sam', '*edge.png'))
        low_edge_list.sort()

        low_semantic_list = glob.glob(os.path.join(folder_path, 'Low_sam', '*semantic.png'))
        low_semantic_list.sort()

        high_edge_list = glob.glob(os.path.join(folder_path, 'Normal_sam', '*edge.png'))
        high_edge_list.sort()

        high_semantic_list = glob.glob(os.path.join(folder_path, 'Normal_sam', '*semantic.png'))
        high_semantic_list.sort()

        pairs = []

        for idx in range(len(low_list)):
            name = os.path.basename(low_list[idx])
            f_name_low = low_list[idx]
            f_name_high = high_list[idx]
            f_name_low_edge = low_edge_list[idx]
            f_name_low_semantic = low_semantic_list[idx]
            f_name_high_edge = high_edge_list[idx]
            f_name_high_semantic = high_semantic_list[idx]
            pairs.append(
                [cv2.cvtColor(cv2.imread(f_name_low), cv2.COLOR_BGR2RGB),
                 cv2.cvtColor(cv2.imread(f_name_high), cv2.COLOR_BGR2RGB),
                 cv2.cvtColor(cv2.imread(f_name_low_edge), cv2.COLOR_BGR2RGB),
                 cv2.cvtColor(cv2.imread(f_name_low_semantic), cv2.COLOR_BGR2RGB),
                 cv2.cvtColor(cv2.imread(f_name_high_edge), cv2.COLOR_BGR2RGB),
                 cv2.cvtColor(cv2.imread(f_name_high_semantic), cv2.COLOR_BGR2RGB),
                 name.split('.')[0]
                 ])
        return pairs


# class LOLEdgeSemantic(data.Dataset):
#     def __init__(self, opt, train):
#         self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
#         self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
#         self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
#         self.crop_size = opt.get("patch_size", None)
#         self.train_list = os.path.join(dir, self.file_list)
#         with open(self.train_list) as f:
#             contents = f.readlines()
#             input_names = [i.strip() for i in contents]
#             gt_names = [i.strip().replace('low', 'high') for i in input_names]
#         self.input_names = input_names
#         self.gt_names = gt_names
#         if self.train:
#             self.transforms = GroupCompose([
#                 GroupRandomCrop(self.crop_size),
#                 GroupRandomHorizontalFilp(),
#                 GroupRandomVerticalFlip(),
#                 GroupToTensor()
#             ]
#             )
#
#         else:
#             self.transforms = GroupCompose([
#                 GroupToTensor(),
#             ])


def random_flip(img, seg):
    random_choice = np.random.choice([True, False])
    img = img if random_choice else np.flip(img, 1).copy()
    seg = seg if random_choice else np.flip(seg, 1).copy()

    return img, seg


def gamma_aug(img, gamma=0):
    max_val = img.max()
    img_after_norm = img / max_val
    img_after_norm = np.power(img_after_norm, gamma)
    return img_after_norm * max_val


def random_rotation(img, seg):
    random_choice = np.random.choice([0, 1, 3])
    img = np.rot90(img, random_choice, axes=(0, 1)).copy()
    seg = np.rot90(seg, random_choice, axes=(0, 1)).copy()

    return img, seg


def random_crop(hr, lr, size_hr):
    size_lr = size_hr

    size_lr_x = lr.shape[0]
    size_lr_y = lr.shape[1]

    start_x_lr = np.random.randint(low=0, high=(size_lr_x - size_lr) + 1) if size_lr_x > size_lr else 0
    start_y_lr = np.random.randint(low=0, high=(size_lr_y - size_lr) + 1) if size_lr_y > size_lr else 0

    # LR Patch
    lr_patch = lr[start_x_lr:start_x_lr + size_lr, start_y_lr:start_y_lr + size_lr, :]

    # HR Patch
    start_x_hr = start_x_lr
    start_y_hr = start_y_lr
    hr_patch = hr[start_x_hr:start_x_hr + size_hr, start_y_hr:start_y_hr + size_hr, :]

    # HisEq Patch
    his_eq_patch = None
    return hr_patch, lr_patch,


def random_group_flip(imgs):
    random_choice = np.random.choice([True, False])
    transformed = [img if random_choice else np.flip(img, 1).copy() for img in imgs]

    return transformed


def random_group_rotation(img):
    random_choice = np.random.choice([0, 1, 3])
    transformed = [np.rot90(img, random_choice, axes=(0, 1)).copy() for img in img]
    # seg = np.rot90(seg, random_choice, axes=(0, 1)).copy()

    return transformed


def random_group_crop(imgs, size_hr):
    size_lr = size_hr

    size_lr_x = imgs[0].shape[0]
    size_lr_y = imgs[0].shape[1]

    start_x_lr = np.random.randint(low=0, high=(size_lr_x - size_lr) + 1) if size_lr_x > size_lr else 0
    start_y_lr = np.random.randint(low=0, high=(size_lr_y - size_lr) + 1) if size_lr_y > size_lr else 0

    transformed = [img[start_x_lr:start_x_lr + size_lr, start_y_lr:start_y_lr + size_lr, :] for img in imgs]
    return transformed


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()


def transform_augment(imgs, split='val', min_max=(0, 1)):
    # imgs = [totensor(img) for img in img_list]
    # if split == 'train':
    #     imgs = torch.stack(imgs, 0)
    #     imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img


class Unpaired_EdgeSemantic_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.crop_size = opt.get("patch_size", None)
        self.sub_data = opt.get("sub_data", None)
        self.pairs = []
        self.train = train
        if train:
            self.split = 'train'
            root = os.path.join(self.root, self.sub_data, 'Train')
        else:
            self.split = 'val'
            root = os.path.join(self.root, self.sub_data, 'Test')
        self.pairs.extend(self.load_pairs(root))
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):
        low_list = glob.glob(os.path.join(folder_path, 'Low', '*.png')) + glob.glob(os.path.join(folder_path, 'Low', '*.jpg'))
        low_list.sort()

        low_sam_path = os.path.join(folder_path, 'Low_sam')
        low_edge_list = glob.glob(os.path.join(low_sam_path, '*edge.png'))
        low_edge_list.sort()

        low_semantic_list = glob.glob(os.path.join(low_sam_path, '*semantic.png'))
        low_semantic_list.sort()

        pairs = []

        for idx in range(len(low_list)):
            name = os.path.basename(low_list[idx])
            f_name_low = low_list[idx]
            f_name_low_edge = low_edge_list[idx]
            f_name_low_semantic = low_semantic_list[idx]
            pairs.append(
                [cv2.cvtColor(cv2.imread(f_name_low), cv2.COLOR_BGR2RGB),
                 cv2.cvtColor(cv2.imread(f_name_low_edge), cv2.COLOR_BGR2RGB),
                 cv2.cvtColor(cv2.imread(f_name_low_semantic), cv2.COLOR_BGR2RGB),
                 name.split('.')[0]
                 ])
        return pairs

    def __getitem__(self, item):
        lr, le, ls, f_name = self.pairs[item]
        h, w, c = lr.shape
        if self.use_crop and self.split != 'val':
            lr, le, ls = random_group_crop([lr, le, ls], self.crop_size)
        elif self.split == 'val':
            pad_factor = 32
            bottom = int(np.ceil(h / pad_factor) * pad_factor - h)
            right = int(np.ceil(w / pad_factor) * pad_factor - w)
            lr = cv2.copyMakeBorder(lr, 0, bottom, 0, right, cv2.BORDER_REFLECT)
            le = cv2.copyMakeBorder(le, 0, bottom, 0, right, cv2.BORDER_REFLECT)
            ls = cv2.copyMakeBorder(ls, 0, bottom, 0, right, cv2.BORDER_REFLECT)

        if self.use_flip:
            lr, le, ls = random_group_flip([lr, le, ls])

        if self.use_rot:
            lr, le, ls = random_group_rotation([lr, le, ls])

        lr = self.to_tensor(lr)
        le = self.to_tensor(le)
        ls = self.to_tensor(ls)

        [lr] = transform_augment([lr], split=self.split, min_max=(-1, 1))

        return {'LQ': lr, 'GT': lr.clone(), 'le': le, 'ls': ls, 'he': le, 'hs': ls, 'LQ_path': f_name, 'GT_path': f_name, 'h': h, 'w': w}
