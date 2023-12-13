from .clip_vit import build_clip_vit
from .clip_transformer import build_clip_transformer
from .resnet import build_clip_resnet


def build_visual_model(cfg):
    if cfg.MODEL.VISUAL_MODEL in ["ViT-B/32", "ViT-B/16"]:
        return build_clip_vit(cfg.MODEL.VISUAL_MODEL,
                              cfg.PATH.PRE_TRAINED,
                              cfg.INPUT.SIZE,
                              cfg.MODEL.FREEZE_CONV1)
    if cfg.MODEL.VISUAL_MODEL in ["RN50", "RN101", "RN50x4"]:
        return build_clip_resnet(cfg.MODEL.VISUAL_MODEL,
                                 cfg.PATH.PRE_TRAINED,
                                 cfg.INPUT.SIZE)
    raise NotImplementedError


def build_textual_model(cfg):
    if cfg.MODEL.TEXTUAL_MODEL in ["ViT-B/32", "ViT-B/16", "RN50", "RN101", "RN50x4"]:
        return build_clip_transformer(cfg.MODEL.TEXTUAL_MODEL,
                                      cfg.PATH.PRE_TRAINED,
                                      cfg.MODEL.TEXTUAL_MODEL_FREEZE_LAYER,
                                      cfg.MODEL.TEXT_DROPOUT)
    raise NotImplementedError
