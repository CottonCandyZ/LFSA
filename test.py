import torch
import os

from config import cfg
from datasets import build_dataloader
from visualizer import get_local
get_local.activate() # 激活装饰器


from model.build import build_model
from engine.engine import do_inference
from tools.options import get_args
from tools.logger import setup_logger
from tools.checkpoint import Checkpointer

if __name__ == '__main__':
    
    # config file
    args = get_args(train=False)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    # log
    output_dir = "/".join(args.checkpoint_file.split("/")[:-1])

    logger = setup_logger('LFSA', save_dir=output_dir,
                          filename="test_log.txt")
    logger.info(str(args).replace(',', '\n'))

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    
    
    # Data
    *test_loaders , num_classes = build_dataloader(
        cfg, training=False)

    # Model
    torch.cuda.set_device(args.device_num)
    model = build_model(cfg, num_classes)
    
    checkpointer = Checkpointer(model)
    checkpointer.load(args.checkpoint_file)
    
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    
    # save_dic = do_inference(model, test_loaders, embed_type=cfg.MODEL.EMBEDDING.EMBED_HEAD, save=args.save)
    save_dic = do_inference(model, test_loaders, embed_type='default', save=args.save)
    # save_dic = do_inference(model, test_loaders, embed_type='local', save=args.save)
    
    if args.save:
        # save_dic.update(get_local.cache)
    
        torch.save(
            save_dic,
            os.path.join(output_dir, "inference_data.pt")
        )
