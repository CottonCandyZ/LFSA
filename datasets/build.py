import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from tools.simple_tokenizer import SimpleTokenizer


from .bases import ImageDataset, TextDataset, ImageTextDataset, ImageTextDatasetTwoCaption

from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES

__factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES}

# Same as TextReID
def build_transforms(img_size=(384, 128), aug=False, is_train=True, gray_scale=False):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # transform for training
    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    elif gray_scale:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.Grayscale,
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform


def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
             batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict

def build_dataloader(cfg, training=True, return_ds=False, color_remove=True, gray_scale=False):
    logger = logging.getLogger("LFSA.dataset")

    # Parameters
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = __factory[cfg.DATASETS.NAME](root=cfg.PATH.DATASETS)
    num_classes = len(dataset.train_id_container)
    
    image_size = cfg.INPUT.SIZE
    batch_size = cfg.SOLVER.IMS_PER_BATCH
    test_batch_size = cfg.TEST.IMS_PER_BATCH
    use_aug = cfg.INPUT.USE_AUG
    image_per_id = cfg.DATALOADER.IMS_PER_ID
    sampler = cfg.DATALOADER.SAMPLER
    train_tokenizer = SimpleTokenizer(
        remove_color=cfg.DATALOADER.TEXT_REMOVE_COLOR,
        drop_pro = cfg.DATALOADER.TEXT_REMOVE_COLOR_PRO,
        on_sentence=cfg.DATALOADER.REMOVE_COLOR_ON_SENTENCE,
        remove_color_type=cfg.DATALOADER.REMOVE_COLOR_TYPE,
        sen_drop_prob=cfg.DATALOADER.SEN_DROP_PROB,
        per_sen_max=cfg.DATALOADER.PER_SEN_MAX,
        random_remove_set=cfg.DATALOADER.RANDOM_REMOVE_SET,
        mod_percent_per_sen=cfg.DATALOADER.MOD_PERCENT_PER_SEN,
        BERT_drop_prob=cfg.DATALOADER.BERT_DROP_PROB,
        BERT_remove=cfg.DATALOADER.BERT_REMOVE,
        BERT_change_prob=cfg.DATALOADER.BERT_CHANGE_PROB,
        mask_color_type=cfg.DATALOADER.MASK_COLOR_TYPE,
        mask_color=cfg.DATALOADER.MASK_COLOR,
        mask_prob_per_sen=cfg.DATALOADER.MASK_PROB_PER_SEN,
        mask_prob=cfg.DATALOADER.MASK_PROB,
        encode_type=cfg.DATALOADER.TEXT_ENCODE_TYPE,
        drop_prob_per_sen=cfg.DATALOADER.DROP_PROB_PER_SEN,
        change_sen_prob=cfg.DATALOADER.CHANGE_SEN_PROB
    ) if color_remove else SimpleTokenizer()
    train_tokenizer_main = SimpleTokenizer(
        remove_color=cfg.DATALOADER.TEXT_REMOVE_COLOR,
        drop_pro = cfg.DATALOADER.TEXT_REMOVE_COLOR_PRO,
        on_sentence=cfg.DATALOADER.REMOVE_COLOR_ON_SENTENCE,
        remove_color_type=cfg.DATALOADER.REMOVE_COLOR_TYPE,
        sen_drop_prob=cfg.DATALOADER.SEN_DROP_PROB,
        per_sen_max=cfg.DATALOADER.PER_SEN_MAX,
        random_remove_set=cfg.DATALOADER.RANDOM_REMOVE_SET,
        mod_percent_per_sen=cfg.DATALOADER.MAIN_MOD_PERCENT_PER_SEN,
        BERT_drop_prob=cfg.DATALOADER.BERT_DROP_PROB,
        BERT_remove=cfg.DATALOADER.BERT_REMOVE,
        BERT_change_prob=cfg.DATALOADER.BERT_CHANGE_PROB,
        mask_color_type=cfg.DATALOADER.MASK_COLOR_TYPE,
        mask_color=cfg.DATALOADER.MASK_COLOR,
        mask_prob_per_sen=cfg.DATALOADER.MAIN_MASK_PROB_PER_SEN,
        mask_prob=cfg.DATALOADER.MASK_PROB,
        encode_type=cfg.DATALOADER.MAIN_TEXT_ENCODE_TYPE,
        drop_prob_per_sen=cfg.DATALOADER.MAIN_DROP_PROB_PER_SEN,
        change_sen_prob=cfg.DATALOADER.CHANGE_SEN_PROB
    ) if color_remove else SimpleTokenizer()
    if training:
        train_transforms = build_transforms(img_size=image_size,
                                            aug=use_aug,
                                            is_train=True,
                                            gray_scale=gray_scale)
        val_transforms = build_transforms(img_size=image_size,
                                          is_train=False)
        if 'two_stream' in cfg.MODEL.EMBEDDING.EMBED_HEAD:
            train_set = ImageTextDatasetTwoCaption(dataset.train,
                                        train_transforms,
                                        tokenizer_main=train_tokenizer_main,
                                        tokenizer=train_tokenizer)
        else:
            train_set = ImageTextDataset(dataset.train,
                                        train_transforms,
                                        tokenizer=train_tokenizer)
            

        if sampler == 'identity':
            logger.info(
                f'using random identity sampler: batch_size: {batch_size}, id: {batch_size // image_per_id}, instance: {image_per_id}'
            )
            train_loader = DataLoader(train_set,
                                        batch_size=batch_size,
                                        sampler=RandomIdentitySampler(
                                            dataset.train, batch_size,
                                            image_per_id),
                                        num_workers=num_workers,
                                        collate_fn=collate)
        elif sampler == 'random':
            logger.info('using random sampler')
            train_loader = DataLoader(train_set,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate)
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(sampler))

        ds = dataset.val if cfg.DATASETS.VAL == 'val' else dataset.test
        val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                   val_transforms)
        val_txt_set = TextDataset(ds['caption_pids'],
                                  ds['captions'])

        val_img_loader = DataLoader(val_img_set,
                                    batch_size=test_batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=test_batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)

        return train_loader, val_img_loader, val_txt_loader, num_classes

    else:  # for testing
        if return_ds:
            return dataset.test
        test_transforms = build_transforms(img_size=image_size, is_train=False)

        ds = dataset.test
        test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                    test_transforms)
        test_txt_set = TextDataset(ds['caption_pids'],
                                   ds['captions'])

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        return test_img_loader, test_txt_loader, num_classes
