import logging
import time
import datetime

import torch
from .evaluate import Evaluator



def do_train(arguments, model, train_loaders, val_loaders, optimizer, scheduler, checkpointer, meters):
    train_loader = train_loaders[0]
    logger = logging.getLogger("LFSA.train")
    logger.info('start training 🚀')
    
    # arg
    max_epoch = arguments["max_epoch"]
    epoch = arguments["epoch"]
    device = arguments["device"]
    log_period = arguments["log_period"]
    evaluate_period = arguments["evaluate_period"]
    checkpoint_epoch = arguments["checkpoint_epoch"]
    # compute time
    iteration = arguments["iteration"]
    embed_type = arguments["embed_type"]

    best_top1 = 0.0
    best_mAP = 0.0
    start_training_time = time.time()
    start_iter_time = time.time()
    evaluator = Evaluator(val_loaders)
    max_iter = max_epoch * len(train_loader)

    while epoch < max_epoch:
        epoch += 1
        model.train()
        start_batch_time = time.time()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            iteration += 1
            arguments["iteration"] = iteration
            # loss
            if 'two_stream' in embed_type:
                loss_dict = model(batch['images'], {'captions': batch['captions']
                                                    , 'captions_ori': batch['captions_ori']},
                              {
                                  "pids": batch["pids"],
                                  "image_ids": batch["image_ids"],
                                  "color_drop_label": batch["color_drop_label"],
                              })
            else:
                loss_dict = model(batch['images'], batch['captions'], 
                              {
                                  "pids": batch["pids"],
                                  "image_ids": batch["image_ids"],
                                  "color_drop_label": batch["color_drop_label"],
                              })
            losses = sum(loss for loss in loss_dict.values())
            meters.update(loss=losses, **loss_dict)
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            # time
            batch_time = time.time() - start_iter_time
            start_iter_time = time.time()
            meters.update(time=batch_time)
            
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            
            # print log
            if (step + 1) % log_period == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch [{epoch}][{step}/{num_iter}]",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        step=step + 1,
                        num_iter=len(train_loader),
                        meters=str(meters),
                        lr=scheduler.get_lr()[0],
                        # lr=param_group['lr'],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )

        scheduler.step()
        # it += 1
        
        # time
        end_batch_time = time.time()
        time_per_batch = (end_batch_time - start_batch_time) / (step + 1)
        logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        
        # checkpoint save
        if epoch in checkpoint_epoch:
            logger.info(f"save model at check point epoch: {epoch}.")
            checkpointer.save("epoch_{:d}".format(epoch), **arguments)
        
        # eval
        if epoch % evaluate_period == 0:
            logger.info("Validation Results - Epoch: {}".format(epoch))
            top1, mAP = evaluator.eval(model.eval())
            meters.update(top1=top1)
            torch.cuda.empty_cache()
            if best_mAP < mAP:
                best_mAP = mAP
            if best_top1 < top1:
                best_top1 = top1
                arguments['best_epoch'] = epoch
                checkpointer.save("best", **arguments)
        logger.info(f"best R1: {best_top1} at epoch {arguments['best_epoch']}")
        arguments['epoch'] = epoch

    # Total time
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    return best_top1, best_mAP

def do_inference(model, test_loaders, embed_type, save=False):
    logger = logging.getLogger("LFSA.test")
    logger.info("Enter inferencing")
    evaluator = Evaluator(test_loaders, type=embed_type)
    save_dic = evaluator.eval(model.eval(), save=save)
    if save:
        return save_dic
    else:
        return None