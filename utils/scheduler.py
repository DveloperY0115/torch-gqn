from torch.optim.lr_scheduler import _LRScheduler


class AnnealingStepLR(_LRScheduler):

    def __init__(self, optimizer, mu_i, mu_f, n=1.6e6, last_epoch=-1):
        self.optimizer = optimizer
        self.mu_i = mu_i
        self.mu_f = mu_f
        self.n = n
        self.last_epoch = last_epoch
        super(AnnealingStepLR, self).__init__(optimizer)

    def state_dict(self):
        state_dict = {key: value for key,
                      value in self.__dict__.items() if key not in ('optimizer')}
        return state_dict

    def get_lr(self):
        return [max(self.mu_f + (self.mu_i - self.mu_f) * (1.0 - self.last_epoch / self.n), self.mu_f) for base_lr in self.base_lrs]
