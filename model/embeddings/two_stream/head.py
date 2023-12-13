import torch
from torch import nn
from .loss import build_loss_evaluator

from tools.download_and_load_CLIP import load

class CLIP_head(nn.Module):
    def __init__(self, cfg, visual_size, textual_size, embed_dim, num_classes) -> None:
        super().__init__()
        if "RN" not in cfg.MODEL.VISUAL_MODEL:
            self.visual_projection = nn.Parameter(torch.empty(visual_size, embed_dim))
        self.textual_projection = nn.Parameter(torch.empty(textual_size, embed_dim))
        
        self.loss_evaluator = build_loss_evaluator(cfg, num_classes, embed_dim)
        self.cfg = cfg
    
    def project_visual_only(self, visual_feature):
        if "RN" not in self.cfg.MODEL.VISUAL_MODEL:
            return visual_feature[:, 0, :] @ self.visual_projection
        return visual_feature
    
    def project_textual_only(self, textual_feature, g_index):
        return textual_feature[torch.arange(textual_feature.shape[0]), g_index] @ self.textual_projection
    
    def forward(self, visual_feature, textual_feature, g_index, labels):
        if "RN" not in self.cfg.MODEL.VISUAL_MODEL:
            visual_embed = visual_feature[:, 0, :] @ self.visual_projection
        else:
            visual_embed = visual_feature
        # textual_embed = {}
        
        textual_embed = (textual_feature[torch.arange(textual_feature.shape[0]), g_index] @ self.textual_projection)
        # textual_embed['ori'] = (textual_feature['ori'][torch.arange(textual_feature['ori'].shape[0]), g_index['ori']] @ self.textual_projection).float()
        
        losses = self.loss_evaluator(visual_embed.float(), textual_embed.float(), labels)
        return losses
    
def build_two_stream_head(cfg, visual_width, textual_width, num_classes) -> nn.Module:
    """init projection weight from clip pre-trained"""
    # test
    # state_dict = load("ViT-B/32")
    if "RN" not in cfg.MODEL.VISUAL_MODEL:
        state_dict_visual = load(cfg.MODEL.VISUAL_MODEL, download_root=cfg.PATH.PRE_TRAINED)
    state_dict_text = load(cfg.MODEL.TEXTUAL_MODEL, download_root=cfg.PATH.PRE_TRAINED)
    embed_dim = state_dict_text["text_projection"].shape[1]
    model = CLIP_head(cfg, visual_width, textual_width, embed_dim, num_classes)
    if "RN" not in cfg.MODEL.VISUAL_MODEL:
        model.state_dict()["visual_projection"].copy_(state_dict_visual["visual.proj"])
    model.state_dict()["textual_projection"].copy_(state_dict_text["text_projection"])
    
    return model
