import os
import logging
import numpy as np
from tensorboardX import SummaryWriter
import shutil
import datetime
import math
from torch.optim.lr_scheduler import LambdaLR

def get_settings(dataset, network, suffix):
    """get training logger, writer, checkpoint_folder"""
    ckpt_folder = os.path.join('checkpoint', dataset, network, suffix)
    log_path = os.path.join('logs', dataset, network, '{}.log'.format(suffix))

    parent_path = os.path.dirname(log_path)  # get parent path
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    
    dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_path = os.path.join('logs', dataset, network, '{}_{}.log'.format(suffix, dt_str))
    
    ckpt_folder = os.path.join("checkpoint", dataset, network, '{}_{}.log'.format(suffix, dt_str))
    
    summary_path = os.path.join("summaries", dataset, network, '{}_{}.log'.format(suffix, dt_str))
    
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    
    writer = SummaryWriter(summary_path)
    return log_path, writer, ckpt_folder


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))