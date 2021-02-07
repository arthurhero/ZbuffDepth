import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    # Initialize parameters
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def freeze_params(m):
    for param in m.parameters():
        param.requires_grad = False


def unfreeze_params(m):
    for param in m.parameters():
        param.requires_grad = True


def save_ckpt(directory, model, hyperparameters=None, optimizer=None, scheduler=None, is_best=True,
              multiple_gpu=False, epoch_num=0, step_num=-1, seed=0):
    '''
    save the current training progress
    directory - the directory to save ckpt
    model - whose state to be saved
    hyperparameters - a dictionary of hyperparameters
    optimizer - whose state to be saved
    scheduler - whose state to be saved
    is_best - a bool, whether this ckpt is the best so far
    multiple_gpu - a bool, whether the model is parallel or not
    epoch_num - current epoch number
    step_num - current step number
    seed - seed of the dataloader sampler
    '''
    ckpt = dict()
    if multiple_gpu:
        ckpt['model'] = model.module.state_dict()
    else:
        ckpt['model'] = model.state_dict()
    if hyperparameters is not None:
        ckpt['hyperparameters'] = hyperparameters
    if optimizer is not None:
        ckpt['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        ckpt['scheduler'] = scheduler.state_dict()
    ckpt['epoch_num'] = epoch_num
    ckpt['step_num'] = step_num
    ckpt['seed'] = seed
    torch.save(ckpt, os.path.join(directory, 'current.ckpt'))
    if is_best:
        torch.save(ckpt['model'], os.path.join(directory, 'best.ckpt'))


def load_ckpt(path, model, hyperparameters=None, optimizer=None, scheduler=None, eval_mode=False):
    '''
    load ckpt for model, optimizer and scheduler
    load hyperparameters from ckpt if there is one
    and return current epoch num, last step, and sampler seed
    path - path to the ckpt; dir if not eval, file if eval
    eval_mode - whether under eval mode or not
    '''
    if eval_mode:
        try:
            model.load_state_dict(torch.load(path))
        except RuntimeError:
            model.load_state_dict(torch.load(path)['model'])
        return
    ckpt = torch.load(os.path.join(path, 'current.ckpt'))
    model.load_state_dict(ckpt['model'])
    if hyperparameters is not None and ckpt['hyperparameters']:
        for _, (key, value) in enumerate(ckpt['hyperparameters'].items()):
            hyperparameters[key] = value
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt['epoch_num'], ckpt['step_num'], ckpt['seed']
