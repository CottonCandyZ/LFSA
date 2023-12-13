import argparse

def get_args(train=True):
    parser = None
    if train:
        parser = argparse.ArgumentParser(description="TransTextReID")
        parser.add_argument("--config-file", default="", metavar="FILE", type=str)
        parser.add_argument("--device-num", default=0, metavar="FILE", type=int)
        parser.add_argument("--resume-from", type=str)
    else:
        parser = argparse.ArgumentParser(description="TestTextReID")
        parser.add_argument("--config-file", default="", metavar="FILE", type=str)
        parser.add_argument("--checkpoint-file", default="", metavar="FILE", type=str)
        parser.add_argument("--save", default=False, action='store_true')
        parser.add_argument("--device-num", default=0, type=int)
    args = parser.parse_args()
    return args