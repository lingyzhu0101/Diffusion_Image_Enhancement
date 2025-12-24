import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch
import random


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCompose(transforms.Compose):
    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


class PairRandomVerticalFlip(transforms.RandomVerticalFlip):
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.vflip(img), F.vflip(label)
        return img, label


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label)


class RandomNoise(object):
    def __init__(self, poisson_para=0.0058, gaussian_para=0.0001):
        self.poisson_para = poisson_para
        self.gaussian_para = gaussian_para

    def __call__(self, image_tensor):
        #  # Gaussian-Poisson Noise
        # noise_poisson = torch.poisson(image_tensor)
        # noise_poisson_sign = torch.randint(low=0, high=2, size=image_tensor.shape) * 2 - 1
        # noise_poisson = noise_poisson * noise_poisson_sign * random.uniform(0, self.poisson_para)
        #
        # noise_gaussian = torch.randn(image_tensor.shape)
        # noise_gaussian = noise_gaussian * random.uniform(0, self.gaussian_para)
        #
        # image_noise_tensor = image_tensor + noise_gaussian + noise_poisson

        # gaussian approximate
        poisson = random.uniform(self.poisson_para * 0.8, self.poisson_para * 1.2)
        gaussian = random.uniform(self.gaussian_para * 0.5, self.gaussian_para * 1.2)
        sigma = image_tensor * poisson + gaussian
        std = torch.sqrt(sigma)
        noise = torch.normal(0, std)

        image_noise_tensor = image_tensor + noise
        return torch.clamp(image_noise_tensor, 0, 1)


class GroupRandomCrop(transforms.RandomCrop):

    def __call__(self, images):

        if self.padding is not None:
            images = [F.pad(image, self.padding, self.fill, self.padding_mode) for image in images]

        # pad the width if needed
        size0, size1 = images[0].size[0], images[0].size[1]
        if self.pad_if_needed and size0 < self.size[1]:
            images = [F.pad(image, (self.size[1] - size0, 0), self.fill, self.padding_mode) for image in images]
        # pad the height if needed
        if self.pad_if_needed and size1 < self.size[0]:
            images = [F.pad(image, (0, self.size[0] - size1), self.fill, self.padding_mode) for image in images]

        i, j, h, w = self.get_params(images[0], self.size)

        return [F.crop(image, i, j, h, w) for image in images]


class GroupCompose(transforms.Compose):
    def __call__(self, images):
        for t in self.transforms:
            images = t(images)
        return images


class GroupRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return [F.hflip(img) for img in imgs]
        return imgs


class GroupRandomVerticalFlip(transforms.RandomVerticalFlip):
    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return [F.vflip(imgs) for imgs in imgs]
        return imgs


class GroupToTensor(transforms.ToTensor):
    def __call__(self, pics):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return [F.to_tensor(pic) for pic in pics]
