import copy
import os
import logging
import sys
import re
sys.path.append("/workspace/tez")

import numpy as np
import pandas as pd
import torch
from math import ceil
from joblib import Parallel, delayed
from tqdm import tqdm
from pytorch_lightning.callbacks import Callback

from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.optim import Optimizer
from torch._six import inf
from torch.nn.utils.rnn import pad_sequence


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def biaffine_decode(logits):
    logits = torch.softmax(logits, -1)
    probs = torch.max(logits, -1).values.cpu().tolist()
    preds = torch.argmax(logits, -1).cpu().tolist()
    
    results = []
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        tmp_preds = [[0] * 15 for _ in range(len(pred))]
        for row_idx, row in enumerate(pred):
            col_idx = 0
            while col_idx < len(pred):
                val = row[col_idx]
                if val == 0:
                    col_idx += 1
                    continue
                
                next_idx = col_idx + 1
                while next_idx < len(pred):
                    next_val = row[next_idx]
                    if val == next_val:
                        next_idx += 1
                        continue
                    break
                
                next_idx -= 1
                start, end = row_idx, next_idx
                label_str = span_id_target_map[val]
                cur_prob = prob[row_idx][next_idx]
                tmp_preds[start][target_id_map[f"B-{label_str}"]] = cur_prob
                if start == end:
                    col_idx = next_idx + 1
                    continue
                for idx in range(start + 1, end + 1):
                    tmp_preds[idx][target_id_map[f"I-{label_str}"]] = cur_prob
                
                col_idx = next_idx + 1
            
        results.append(tmp_preds)
        
    results = torch.tensor(results)
    return results
                    

def span_decode(start_logits, end_logits):
    start_logits = torch.softmax(start_logits, -1)
    end_logits = torch.softmax(end_logits, -1)
    
    start_probs = torch.max(start_logits, -1)
    end_probs = torch.max(end_logits, -1)
    
    start_preds = torch.argmax(start_logits, -1)
    end_preds = torch.argmax(end_logits, -1)
    
    start_preds = start_preds.cpu().tolist()
    end_preds = end_preds.cpu().tolist()
    start_probs = start_probs.values.cpu().tolist()
    end_probs = end_probs.values.cpu().tolist()
    
    preds = []
    for start_pred, end_pred, start_prob, end_prob in zip(start_preds, end_preds, start_probs, end_probs):
        pred = [[0] * 15 for _ in range(len(start_pred))]
        #pred = [14] * len(start_pred)
        idx = 0
        next_end_idx = None
        end_idx = None
        while idx < len(start_pred):
            s_type = start_pred[idx]
            if s_type == 0 and next_end_idx is None:
                pred[idx][14] = 1
                idx += 1
                continue
            elif s_type == 0 and next_end_idx is not None:
                idx += 1
                if next_end_idx == idx:
                    next_end_idx = None
                continue
            
            cur_type_str = span_id_target_map[s_type]
            cur_start_prob = start_prob[idx]
            cur_end_prob = 0
            
            
            end_idx = idx + 1
            while end_idx < len(end_pred):
                if end_pred[end_idx] == 0:
                    end_idx += 1
                    continue
                    
                e_type = end_pred[end_idx]
                end_idx += 1
                if s_type == e_type:
                    cur_end_prob = end_prob[end_idx - 1]
                    break

            prob = (cur_start_prob + cur_end_prob) / 2

            pred[idx][target_id_map[f"B-{cur_type_str}"]] = prob
            for i in range(idx + 1, end_idx):
                pred[i][target_id_map[f"I-{cur_type_str}"]] = prob
            idx += 1
            next_end_idx = end_idx
        preds.append(pred)
        
    preds = torch.tensor(preds)
    return preds


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist


class Freeze(Callback):
    def __init__(self, epochs=1, method="hard"):
        self.thre = epochs
        self.method = method
        self.done = False
        
    def on_epoch_start(self, trainer, model):
        if self.thre == 0:
            self.done = True
            for idx, params in enumerate(model.optimizer_grouped_parameters[-2:]):
                if self.method == "hard":
                    params = params["params"]
                    for param in params:
                        param.requires_grad = False
                else:
                    params["lr"] = 7e-7
        
    def on_epoch_end(self, trainer, model):
        if self.thre == 0:
            return
        
        if self.done:
            return
        
        if trainer.current_epoch < self.thre:
            return
        
        print("freeze crf and linear layer")
        self.done = True
        
        for params in model.optimizer_grouped_parameters[-2:]:
            if self.method == "hard":
                params = params["params"]
                for param in params:
                    param.requires_grad = False
            else:
                params["lr"] = 7e-7


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, warmup_epoch, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.warmup_epoch = warmup_epoch
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch % self.total_epoch > self.warmup_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.warmup_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        self.last_epoch %= self.total_epoch
        if self.last_epoch <= self.warmup_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, self.last_epoch - self.warmup_epoch)

    def step(self, metrics=None, epoch=None):
        if not isinstance(self.after_scheduler, ReduceLROnPlateau):
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.warmup_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

            
class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown
        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    logging.info('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)

         
def process_feature_text(text):
    text = re.sub('I-year', '1-year', text)
    text = re.sub('-OR-', " or ", text)
    text = re.sub('-', ' ', text)
    return text

def clean_spaces(txt):
    txt = re.sub('\t', ' ', txt)
    txt = re.sub('\r', ' ', txt)
    return txt