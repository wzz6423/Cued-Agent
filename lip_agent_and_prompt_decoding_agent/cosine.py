import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, num_epochs, iter_per_epoch):
        self.base_lrs = {
            param_group["name"]: param_group["lr"]
            for param_group in optimizer.param_groups
        }
        self.warmup_iter = warmup_epochs * iter_per_epoch
        self.total_iter = num_epochs * iter_per_epoch
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

        self.init_lr()  # so that at first step we have the correct step size

    def get_lr(self, base_lr):
        if self.iter < self.warmup_iter:
            return base_lr * self.iter / self.warmup_iter
        else:
            decay_iter = self.total_iter - self.warmup_iter
            return (
                0.5
                * base_lr
                * (1 + np.cos(np.pi * (self.iter - self.warmup_iter) / decay_iter))
            )

    def update_param_groups(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.get_lr(self.base_lrs[param_group["name"]])

    def step(self):
        self.update_param_groups()
        self.iter += 1

    def init_lr(self):
        self.update_param_groups()


class WarmupCosineRestartScheduler(_LRScheduler):
    """Cosine Annealing with Warm Restarts + Initial Warmup"""

    def __init__(self, optimizer, warmup_epochs, T_0, iter_per_epoch, T_mult=2, eta_min_ratio=0.01):
        """
        Args:
            warmup_epochs: Number of warmup epochs
            T_0: First cycle length in epochs
            iter_per_epoch: Steps per epoch
            T_mult: Cycle length multiplier (cycle lengths: T_0, T_0*T_mult, T_0*T_mult^2, ...)
            eta_min_ratio: Minimum lr ratio (eta_min = base_lr * eta_min_ratio)
        """
        self.base_lrs = {
            param_group["name"]: param_group["lr"]
            for param_group in optimizer.param_groups
        }
        self.warmup_iter = warmup_epochs * iter_per_epoch
        self.T_0_iter = T_0 * iter_per_epoch
        self.T_mult = T_mult
        self.eta_min_ratio = eta_min_ratio
        self.optimizer = optimizer
        self.iter = 0

        self.init_lr()

    def get_lr(self, base_lr):
        eta_min = base_lr * self.eta_min_ratio

        # Warmup phase
        if self.iter < self.warmup_iter:
            return base_lr * self.iter / self.warmup_iter

        # Post-warmup: cosine annealing with restarts
        t = self.iter - self.warmup_iter

        # Find current cycle
        T_cur = self.T_0_iter
        cycle_start = 0
        while t >= cycle_start + T_cur:
            cycle_start += T_cur
            T_cur = int(T_cur * self.T_mult)

        # Position within current cycle
        t_in_cycle = t - cycle_start

        return eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * t_in_cycle / T_cur)) / 2

    def update_param_groups(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.get_lr(self.base_lrs[param_group["name"]])

    def step(self):
        self.update_param_groups()
        self.iter += 1

    def init_lr(self):
        self.update_param_groups()
