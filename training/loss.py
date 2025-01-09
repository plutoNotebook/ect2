import math

import torch
import torch.nn as nn
from torch_utils import persistence
from torch_utils import distributed as dist

#----------------------------------------------------------------------------
# Loss function proposed in the blog "Consistency Models Made Easy"

@persistence.persistent_class
class ECMLoss:
    def __init__(self, P_mean=-1.1, P_std=2.0, sigma_data=0.5, q=2, c=0.0, k=8.0, b=1.0, cut=4.0, adj='sigmoid'):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        
        if adj == 'const':
            self.t_to_r = self.t_to_r_const
        elif adj == 'sigmoid':
            self.t_to_r = self.t_to_r_sigmoid
        else:
            raise ValueError(f'Unknow schedule type {adj}!')

        self.q = q
        self.stage = 0
        self.ratio = 0.
        
        self.k = k
        self.b = b

        self.c = c
        dist.print0(f'P_mean: {self.P_mean}, P_std: {self.P_std}, q: {self.q}, k {self.k}, b {self.b}, c: {self.c}')

    def update_schedule(self, stage):
        self.stage = stage
        self.ratio = 1 - 1 / self.q ** (stage+1)

    def t_to_r_const(self, t):
        decay = 1 / self.q ** (self.stage+1)
        ratio = 1 - decay
        r = t * ratio
        return torch.clamp(r, min=0)

    def t_to_r_sigmoid(self, t):
        adj = 1 + self.k * torch.sigmoid(-self.b * t)
        decay = 1 / self.q ** (self.stage+1)
        ratio = 1 - decay * adj
        r = t * ratio
        return torch.clamp(r, min=0)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        # t ~ p(t) and r ~ p(r|t, iters) (Mapping fn)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        t = (rnd_normal * self.P_std + self.P_mean).exp()
        r = self.t_to_r(t)

        # Augmentation if needed
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        
        # Shared noise direction
        eps   = torch.randn_like(y)
        eps_t = eps * t
        eps_r = eps * r
        
        # Shared Dropout Mask
        rng_state = torch.cuda.get_rng_state()
        D_yt = net(y + eps_t, t, labels, augment_labels=augment_labels)
        
        if r.max() > 0:
            torch.cuda.set_rng_state(rng_state)
            with torch.no_grad():
                D_yr = net(y + eps_r, r, labels, augment_labels=augment_labels)
            
            mask = r > 0
            D_yr = torch.nan_to_num(D_yr)
            D_yr = mask * D_yr + (~mask) * y
        else:
            D_yr = y
        # L2 Loss
        loss = (D_yt - D_yr) ** 2
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        
        # Producing Adaptive Weighting (p=0.5) through Huber Loss
        if self.c > 0:
            loss = torch.sqrt(loss + self.c ** 2) - self.c
        else:
            loss = torch.sqrt(loss)
        
        # Weighting fn
        return loss / (t - r).flatten()

@persistence.persistent_class
class ECDLoss:
    def __init__(self, P_mean=-1.1, P_std=2.0, sigma_data=0.5, q=2, c=0.0, k=8.0, b=1.0, cut=4.0, adj='sigmoid', gamma=0.5, initial=0.7, mu_type='step_lr', ctm=False,):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data    

        if adj == 'const':
            self.t_to_r = self.t_to_r_const
        elif adj == 'sigmoid':
            self.t_to_r = self.t_to_r_sigmoid
        else:
            raise ValueError(f'Unknow schedule type {adj}!')

        if mu_type == 'step_lr':
            self.mu = self.inv_step_lr
        elif mu_type == 'sigmoid':
            self.mu = self.mu_sigmoid
        else:
            self.mu = self.mu_const

        self.q = q
        self.stage = 0
        self.ratio = 0.

        self.stage_mu = 0
        self.ratio_mu = 0.

        self.gamma = gamma
        self.initial = initial
        
        self.ctm = ctm
        
        self.k = k
        self.b = b

        self.c = c
        dist.print0(f'P_mean: {self.P_mean}, P_std: {self.P_std}, q: {self.q}, k {self.k}, b {self.b}, c: {self.c}')
        
        self.dpm_solver = False
        self.heun = False

    def update_schedule(self, stage):
        self.stage = stage
        self.ratio = 1 - 1 / self.q ** (stage+1)

    def t_to_r_const(self, t):
        decay = 1 / self.q ** (self.stage+1)
        ratio = 1 - decay
        r = t * ratio
        return torch.clamp(r, min=0)

    def t_to_r_sigmoid(self, t):
        adj = 1 + self.k * torch.sigmoid(-self.b * t)
        decay = 1 / self.q ** (self.stage+1)
        ratio = 1 - decay * adj
        r = t * ratio
        return torch.clamp(r, min=0)
    
    def update_schedule_mu(self, stage_mu):
        self.stage_mu = stage_mu
        self.ratio_mu = 1 - 1 / self.q ** (stage_mu+1)

    def update_schedule_step(self, stage_step):
        self.stage_step = stage_step

    def inv_step_lr(self, t):
        #ratio = 1 - self.initial * (self.gamma ** self.stage_mu)
        #return torch.tensor(ratio, dtype=t.dtype)
        adj = 1 + self.k * torch.sigmoid(-self.b * t)
        ratio = 1 - (self.initial * (self.gamma ** self.stage_mu)) * adj
        return torch.clamp(ratio, min=0)
    def sigma(self, t):
        return t.neg().exp()
    
    def eps(self, x, t, model):
        """Calculates the epsilon for the DPM-Solver."""
        sigma_val = self.sigma(t) * x.new_ones([x.shape[0]])
        eps_val = (x - model(x, sigma_val)) / self.sigma(t)
        return eps_val

    def dpm_solver_1_step(self, x, t, r, D_yt):
        """Performs a single step of the DPM-Solver."""
        h = r - t
        eps_val = (x - D_yt) / self.sigma(t)
        x_1 = x - self.sigma(r) * h.expm1() * eps_val
        return x_1
    
    def dxdt(self, x_t, x, t):
        return (x_t - x) / t
    
    def heun_solver(self, x_t, x, t, r):
        h = t - r
        x_bar = x_t - h * self.dxdt(x_t, x, t)
        x_1 = x_t - 0.5 * h * (self.dxdt(x_t, x, t) + self.dxdt(x_bar, x, r))
        return x_1
    
    
    def stf_targets(self, sigmas, perturbed_samples, ref, target):
        """
        Compute stable target using reference batch and noisy samples.

        Args:
            sigmas: noisy levels
            perturbed_samples: perturbed samples with perturbation kernel N(0, sigmas**2)
            ref: the reference batch

        Returns: stable target
        """
        with torch.no_grad():
            perturbed_samples_vec = perturbed_samples.reshape((len(perturbed_samples), -1))
            ref_vec = ref.reshape((len(ref), -1))

            gt_distance = torch.sum((perturbed_samples_vec.unsqueeze(1) - ref_vec) ** 2,
                                    dim=[-1])
            gt_distance = - gt_distance / (2 * sigmas.unsqueeze(1) ** 2)
            # adding a constant to the log-weights to prevent numerical issue
            distance = - torch.max(gt_distance, dim=1, keepdim=True)[0] + gt_distance
            distance = torch.exp(distance)[:, :, None]
            # self-normalize the per-sample weight of reference batch
            weights = distance / (torch.sum(distance, dim=1, keepdim=True))

            #target = ref_vec.unsqueeze(0).repeat(len(perturbed_samples), 1, 1)
            # calculate the stable targets with reference batch
            stable_targets = torch.sum(weights * target, dim=1)
            return stable_targets


    def mu_const(self, t):
        decay = 1 / self.q ** (self.stage_mu+1)
        ratio = 1 - decay
        return torch.clamp(torch.tensor(ratio, dtype=t.dtype), min=0)
    
    def mu_sigmoid(self, t):
        adj = 1 + self.k * torch.sigmoid(-self.b * t)
        decay = 1 / self.q ** (self.stage_mu+1)
        ratio = 1 - decay * adj
        return torch.clamp(ratio, min=0)

    def __call__(self, net, images, labels=None, augment_pipe=None, stf=False, ref_images=None):
        # t ~ p(t) and r ~ p(r|t, iters) (Mapping fn)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        t = (rnd_normal * self.P_std + self.P_mean).exp()
        r = self.t_to_r(t)

        # Augmentation if needed
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        
        # Shared noise direction

        eps   = torch.randn_like(y)
        eps_t = eps * t
        eps_r = eps * r

        # Shared Dropout Mask
        rng_state = torch.cuda.get_rng_state()
        yt = y + eps_t

        if self.ctm:
            sigma_min = torch.tensor(net.module.sigma_min, dtype=t.dtype)
            D_yt = net(yt, t, sigma_min, labels, augment_labels=augment_labels)
            D_ytr = net(yt, t, r, labels, augment_labels=augment_labels)
        else:
            D_yt = net(yt, t, None, labels, augment_labels=augment_labels)

        x_t_flat = yt.view(yt.size(0), -1)
        x_0_flat = ref_images.view(ref_images.size(0), -1)

        x_0_expanded = x_0_flat.unsqueeze(0).repeat(yt.size(0), 1, 1)  
        x_t_expanded = x_t_flat.unsqueeze(1).repeat(1, ref_images.size(0), 1)  

        target = (x_t_expanded - x_0_expanded) / t.view(-1, 1, 1)  # (ref_batch, batch_size_t, features)
        target = target.view(y.size(0), ref_images.size(0), -1)

        if stf:
            if augment_pipe is not None:
                ref_images[len(y):], augment_labels_2 = augment_pipe(ref_images[len(y):])
            else:
                # Ensures ref_images[len(y):] is filled with the same shape
                ref_images[len(y):], augment_labels_2 = ref_images[len(y):], None
            # update augmented original images
            ref_images[:len(y)] = y
            target = self.stf_targets(t.squeeze(), yt, ref_images, target)
            target = target.view_as(y)
        else:
            target = y # stable epsilon

        if r.max() > 0:
            torch.cuda.set_rng_state(rng_state)
            with torch.no_grad():
                if self.ctm:
                    sigma_min = torch.tensor(net.module.sigma_min, dtype=t.dtype)
                    D_yr = net(D_ytr, r, sigma_min, labels, augment_labels=augment_labels)
                else:
                    rt = r / t
                    mu = self.mu(t)
                    yr_hat = rt * yt + (1. - rt) * D_yt
                    #yr = target + eps_r
                    yr = yt - (t - r) * target
                    yr_hat = (1. - mu) * yr_hat + mu * yr
                    D_yr = net(yr_hat, r, None, labels, augment_labels=augment_labels)

            mask = r > 0
            D_yr = torch.nan_to_num(D_yr)
            D_yr = mask * D_yr + (~mask) * y
        else:
            D_yr = y
        # if r.max() > 0:
        #     torch.cuda.set_rng_state(rng_state)
        #     with torch.no_grad():
        #         if self.ctm:
        #             sigma_min = torch.tensor(net.module.sigma_min, dtype=t.dtype)
        #             D_yr = net(D_ytr, r, sigma_min, labels, augment_labels=augment_labels)
        #         else:
        #             rt = r / t
        #             mu = self.mu(t)
        #             if self.dpm_solver:
        #                 yr_hat = self.dpm_solver_1_step(images, t, r, D_yt)
        #             else:
        #                 yr_hat = rt * yt + (1. - rt) * D_yt
        #             if self.heun:
        #                 yr = self.heun_solver(yt, target, t, r)
        #             else:
        #                 yr = target + eps_r
        #             yr_hat = (1. - mu) * yr_hat + mu * yr
        #             D_yr = net(yr_hat, r, None, labels, augment_labels=augment_labels)

        #     mask = r > 0
        #     D_yr = torch.nan_to_num(D_yr)
        #     D_yr = mask * D_yr + (~mask) * y
        # else:
        #     D_yr = y
        
        # L2 Loss
        loss = (D_yt - D_yr) ** 2
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        
        # Producing Adaptive Weighting (p=0.5) through Huber Loss
        if self.c > 0:
            loss = torch.sqrt(loss + self.c ** 2) - self.c
        else:
            loss = torch.sqrt(loss)
        
        # Weighting fn
        return loss / (t - r).flatten()
