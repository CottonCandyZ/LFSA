# Contain base and tokenize

from torch.utils.data import Dataset
import logging
import torch
from tools.iotools import read_image
from tools.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable

class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("LFSA.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])
        
        # Print
        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens, drop = tokenizer.encode(caption)
    if isinstance(drop, list):
        # assert len(tokens) == len(drop)
        drop = [0] + drop + [0]
    # 每个词和标点获得一个编码
    tokens = [sot_token] + tokens + [eot_token]
    # 填充 0
    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    if isinstance(drop, list):
        mask_labels = torch.zeros(text_length, dtype=torch.long)
        if len(drop) > text_length:
            drop = drop[:text_length]
            drop[-1] = 0
        mask_labels[:len(drop)] = torch.tensor(drop)
        return result, mask_labels
    return result, drop


class ImageTextDataset(Dataset):
    """For training set, which need to reapeat image twice or more"""
    def __init__(self,
                 dataset,
                 transform=None,
                 tokenizer=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
            
        tokens, color_drop = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'images': img,
            'captions': tokens,
            'pids': pid,
            'image_ids': image_id,
            'color_drop_label': color_drop,
        }

        return ret
    
class ImageTextDatasetTwoCaption(Dataset):
    """For training set, which need to reapeat image twice or more"""
    def __init__(self,
                 dataset,
                 transform=None,
                 tokenizer=None,
                 tokenizer_main=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = tokenizer
        self.tokenizer_main = tokenizer_main if tokenizer_main else SimpleTokenizer()
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        tokens_ori, _ = tokenize(caption, tokenizer=self.tokenizer_main, text_length=self.text_length, truncate=self.truncate)
        tokens, color_drop = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'images': img,
            'captions_ori': tokens_ori,
            'captions': tokens,
            'pids': pid,
            'image_ids': image_id,
            'color_drop_label': color_drop,
        }

        return ret


# Two sperate class for val and test.
# Image and Text do not have the same size.
class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption, _ = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption