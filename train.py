import time
import os
import random
import shutil

import torch
import numpy as np

from config import cfg
from datasets import build_dataloader
from model.build import build_model
from solver.build import build_optimizer, build_lr_scheduler
from engine.engine import do_train
from tools.options import get_args
from tools.logger import setup_logger
from tools.checkpoint import Checkpointer
from tools.metric_logger import TensorboardLogger
from tools.iotools import mkdir_if_missing
from server_push import sc_send

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    

    # Config file
    args = get_args(train=True)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    resume_from = args.resume_from
    name = cfg.NAME
    
    set_seed(cfg.SEED)

    # Log
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_dir = os.path.join(
        cfg.PATH.OUTPUTS, "/".join(args.config_file.split("/")[-2:])[:-5], name + "_" + cur_time)
    mkdir_if_missing(output_dir)

    logger = setup_logger('LFSA', save_dir=output_dir,
                          filename="train_log.txt")
    logger.info(str(args).replace(',', '\n'))

    logger.info("Loaded configuration file {}".format(args.config_file))
    
    # save current config
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        f.write(cfg.dump())
    
    # backup config file
    shutil.copy2(args.config_file, output_dir)
    
    logger.info("Running with config:\n{}".format(cfg))

    train_loader = [None] * 2
    # Data
    train_loader[0], *val_loaders , num_classes = build_dataloader(
        cfg, training=True, color_remove=True)
    
    # train_loader[1], *_, _ = build_dataloader(
    #     cfg, training=True, color_remove=True, gray_scale=True)

    # Model
    torch.cuda.set_device(args.device_num)
    model = build_model(cfg, num_classes)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    # Solver
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # Resume arg
    arguments = {}
    arguments["iteration"] = 0
    arguments["epoch"] = 0

    checkpointer = Checkpointer(
        model, optimizer, scheduler, output_dir, save_to_disk=True)
    if resume_from:
        if os.path.isfile(resume_from):
            extra_checkpoint_data = checkpointer.resume(resume_from)
            arguments.update(extra_checkpoint_data)
        else:
            raise IOError("{} is not a checkpoint file".format(resume_from))

    arguments["checkpoint_epoch"] = cfg.SOLVER.CHECKPOINT_EPOCH
    arguments["log_period"] = cfg.SOLVER.LOG_PERIOD
    arguments["evaluate_period"] = cfg.SOLVER.EVALUATE_PERIOD
    arguments["max_epoch"] = cfg.SOLVER.NUM_EPOCHS
    arguments["device"] = device
    arguments["embed_type"] = cfg.MODEL.EMBEDDING.EMBED_HEAD

    # Meters
    meters = TensorboardLogger(
        log_dir=os.path.join(output_dir),
        start_iter=arguments["iteration"],
        delimiter="  ",
    )
    if cfg.SC_SEND:
        sc_send(f"{name} 开始了")
    R1, mAP = do_train(arguments, model, train_loader, val_loaders, optimizer, scheduler, checkpointer, meters)
    if cfg.SC_SEND:
        sc_send(f"{name} 完成了哦", f"R1: {R1} mAP: {mAP}")
