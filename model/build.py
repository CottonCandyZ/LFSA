from torch import nn
import torch

from .backbones import build_textual_model, build_visual_model
from .embeddings import build_clip_head, build_two_stream_head

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        for name in ["visual_projection", "textual_projection"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


class Model(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.visual_model = build_visual_model(cfg)
        self.textual_model = build_textual_model(cfg)

        if cfg.MODEL.EMBEDDING.EMBED_HEAD == "default":
            self.embed_model = build_clip_head(
                cfg, self.visual_model.width, self.textual_model.width, num_classes
            )
        elif cfg.MODEL.EMBEDDING.EMBED_HEAD == "two_stream":
            self.embed_model = build_two_stream_head(
                cfg, self.visual_model.width, self.textual_model.width, num_classes
            )
        else:
            raise NotImplementedError
        
    def encode_image(self, images):
        return self.embed_model.project_visual_only(self.visual_model(images))
    
    def encode_text(self, captions):
        return self.embed_model.project_textual_only(*self.textual_model(captions))
    
    def forward(self, images, captions, labels):
        visual_feat = self.visual_model(images)
        if isinstance(captions, dict):
            textual_feat = {}
            g_index = {}
            cat_catptions = torch.cat((captions['captions'], captions['captions_ori']), dim=0)
            textual_feat, g_index = self.textual_model(cat_catptions)
        else:
            textual_feat, g_index = self.textual_model(captions)
        
        
        losses_embed = self.embed_model(
            visual_feat, textual_feat, g_index, labels
        )

        losses = {}
        losses.update(losses_embed)
        return losses


def build_model(cfg, num_classes):    
    model = Model(cfg, num_classes)
    if cfg.MODEL.USE_FP16:
        convert_weights(model)
    return model