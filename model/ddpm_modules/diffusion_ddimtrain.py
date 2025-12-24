import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import cv2
from torchvision import transforms
from pytorch_msssim import ssim
import core.metrics as Metrics
from utils import util

transform = transforms.Lambda(lambda t: (t + 1) / 2)


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        # print('exist')
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            image_size,
            channels=3,
            loss_type='l1',
            conditional=True,
            schedule_opt=None,
            opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.opt = opt
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device, schedule_phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        # print(schedule_phase)
        if schedule_phase == 'train':
            self.sqrt_alphas_cumprod_prev_train = np.sqrt(np.append(1., alphas_cumprod))
            self.num_timesteps_train = int(timesteps)
            variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
            self.noise_loss_timestep_scales = (1 / (variance * 2)) * (1 - alphas) ** 2 / ((1 - alphas_cumprod) * alphas)
            self.noise_loss_timestep_scales[0] = 1.0

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
                             (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
                         x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None, q=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            if self.opt['model']['ca']:
                x_recon = self.predict_start_from_noise(
                    x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level, q)[0])
            else:
                x_recon = self.predict_start_from_noise(
                    x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)[0])
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None, q=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, q=q)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        # print(model_log_variance)
        # print(model_log_variance.shape)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    # @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False, q=None):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                          total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in reversed(range(0, self.num_timesteps)):
                img = self.p_sample(img, i, condition_x=x, q=q)
                # if i % sample_inter == 0:
                #     ret_img = torch.cat([ret_img, img], dim=0)
            return img
        # if continous:
        #     return ret_img
        # else:
        #     return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    # @torch.no_grad()
    def super_resolution(self, x_in, continous=False, q=None):
        return self.p_sample_loop(x_in, continous, q=q)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                continuous_sqrt_alpha_cumprod * x_start +
                (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )

    def draw_features(self, x, savename):
        img = x[0, 0, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imwrite(savename, img)

    def predict_start(self, x_t, continuous_sqrt_alpha_cumprod, noise):
        return (1. / continuous_sqrt_alpha_cumprod) * x_t - \
            (1. / continuous_sqrt_alpha_cumprod ** 2 - 1).sqrt() * noise

    def predict_t_minus1(self, x, t, continuous_sqrt_alpha_cumprod, noise, clip_denoised=True):

        x_recon = self.predict_start(x,
                                     continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
                                     noise=noise)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, model_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        noise_z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        return model_mean + noise_z * (0.5 * model_log_variance).exp()

    def to_patches(self, data, kernel_size):

        patches = nn.Unfold(kernel_size=kernel_size, stride=kernel_size)(torch.mean(data, dim=1, keepdim=True))
        patches = patches.transpose(2, 1)

        return patches

    def compute_loss(self, x_in, uct_model, noise=None):

        x_start = x_in['GT']
        if self.opt['model']['ca']:
            low_edge = x_in['low_edge']
            low_semantic = x_in['low_semantic']
            high_edge = x_in['high_edge']
            high_semantic = x_in['high_semantic']
            semantic_err = (low_semantic - high_semantic).abs()
            edge_err = (high_edge - low_edge).abs()
            semantic_w, edge_w = 0.9, 0.1
            loss_mask_sam = (semantic_err * semantic_w + edge_err * edge_w) + 1
            
        [b, c, h, w] = x_start.shape

        t = np.random.randint(1, self.num_timesteps_train + 1)

        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev_train[t - 1],
                self.sqrt_alphas_cumprod_prev_train[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        # noise = default(noise, lambda: torch.randn_like(x_start))
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            if self.opt['model']['ca']:
                x_recon, _ = self.denoise_fn(
                    torch.cat([x_in['LQ'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod, [low_edge, low_semantic])
                pred_x = self.super_resolution(x_in['LQ'], continous=False, q=[low_edge, low_semantic])
                
                consistence_map = 1.0
                if self.opt['loss']['use_consistence_map']:
                    with torch.no_grad():
                        pred_x2 = self.super_resolution(x_in['LQ'], continous=False, q=[low_edge, low_semantic])
                        consistence_map = (pred_x2 - pred_x.detach()).abs()+1 
 
                
            else:
                x_recon, _ = self.denoise_fn(
                    torch.cat([x_in['LQ'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        lambda_1 = self.opt['loss']['noise_loss_weight']
        if self.opt['loss']['use_noise_loss_timestep_scale']:
            noise_loss_timestep_scale = self.noise_loss_timestep_scales[t-1]
            lambda_1 = noise_loss_timestep_scale * lambda_1
        lambda_2 = self.opt['loss']['photo_loss_weight']

        loss_pix = self.loss_func(x_recon*consistence_map, noise*consistence_map) * lambda_1
        # loss_pix = self.loss_func(x_recon, noise) * lambda_1
        pred_img = transform(pred_x)
        gt_img = transform(x_start)
        ssim_loss = 1 - ssim(pred_img, gt_img, data_range=1.0).to(x_start.device)
        if self.opt['loss']['use_loss_mask_sam']:
            if not self.opt['loss']['use_consistence_map']:
                consistence_map = 0
            content_loss = self.loss_func(pred_img *(loss_mask_sam+consistence_map), gt_img * (loss_mask_sam+consistence_map))
        else:
            if not self.opt['loss']['use_consistence_map']:
                consistence_map = 1.0
            content_loss = self.loss_func(pred_img*consistence_map, gt_img*consistence_map)
        photo_loss = ssim_loss + content_loss

        return loss_pix, photo_loss * lambda_2

    def forward(self, x, uct_model, *args, **kwargs):
        return self.compute_loss(x, uct_model, *args, **kwargs)
