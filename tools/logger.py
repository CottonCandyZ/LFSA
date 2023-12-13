# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys

from time import localtime, strftime


def setup_logger(name, save_dir, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Output in Terminal
    ch = logging.StreamHandler(stream=sys.stdout)
    # Specify the lowest severity that will be dispatched to the appropriate destination
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Write to file
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode="w", encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
