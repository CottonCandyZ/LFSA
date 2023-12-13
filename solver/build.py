import torch
import numpy as np
from .lr_scheduler import LRSchedulerWithWarmup


def build_optimizer(cfg, model):
    params = []

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        
        if "classifier" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.INIT_FACTOR
            
        # if "ct" in key:
        #     lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.CROSS_FACTOR
        
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.SGD_MOMENTUM
        )
    elif cfg.SOLVER.OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(cfg.SOLVER.ADAM_ALPHA, cfg.SOLVER.ADAM_BETA),
            eps=1e-3,
        )
    elif cfg.SOLVER.OPTIMIZER == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(cfg.SOLVER.ADAM_ALPHA, cfg.SOLVER.ADAM_BETA),
            eps=1e-8,
        )
    else:
        NotImplementedError

    return optimizer

def cosine_scheduler(cfg, niter_per_ep):
    base_value = cfg.SOLVER.BASE_LR
    start_warmup_value = cfg.SOLVER.START_LR
    final_value = cfg.SOLVER.TARGET_LR
    epochs = cfg.SOLVER.NUM_EPOCHS
    warmup_epochs = cfg.SOLVER.WARMUP_EPOCHS

    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def build_lr_scheduler(cfg, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=cfg.SOLVER.STEPS,
        gamma=cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
        total_epochs=cfg.SOLVER.NUM_EPOCHS,
        mode=cfg.SOLVER.LRSCHEDULER,
        target_lr=cfg.SOLVER.TARGET_LR,
        power=cfg.SOLVER.POWER,
    )
