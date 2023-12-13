import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Tuple

from .transformer import LayerNorm, Transformer
from tools.download_and_load_CLIP import load


# Same as https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py#L206
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], patch_size: int, width: int, layers: int, heads: int) -> None:
        super().__init__()
        # Input not a squre
        self.input_resulution = input_resolution  # (384, 128)
        self.grid_hight = (input_resolution[0] - patch_size) // patch_size + 1
        self.grid_width = (input_resolution[1] - patch_size) // patch_size + 1
        
        # add
        self.width = width
        
        # Use conv1 to generate patches
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(
            scale * torch.randn(width))  # shape = [width]
        # Init is diff from Transformer, which use cos and sin.
        # self.positional_embedding = nn.Parameter(
        #     scale * torch.randn((input_resulution // patch_size) ** 2 + 1, width))  # shape = [grid ** 2 + 1, width]

        # resize
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((self.grid_width * self.grid_hight) + 1, width))


        # CLIP add a LayerNorm
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
            
    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def forward(self, x: torch.Tensor):
        x = x.type(self.dtype)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grids, width]
        # Add zeros to extend `class_embdding` to [*, 1, width], then concat with `x` on dim 1
        x = torch.cat([self.class_embedding.to(x.dtype) +
                       torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # Auto extend batch dim
        x = x + self.positional_embedding.to(x.dtype)

        # CLIP: Adding an additional layer normalization to the combined patch and position embeddings before the transformer
        x = self.ln_pre(x)

        # MultiheadAttention `batch_first=False` by default, so ...
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # Class_embedding go through LayerNorm
        x = self.ln_post(x)  # shape = [*, grids, width]
        
        return x

def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb = posemb.unsqueeze(0)
    posemb_new = posemb_new.unsqueeze(0)

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb.squeeze(0)



def build_clip_vit(name: str, download_root: str, image_resolution: Tuple[int, int], freeze_conv1: bool) -> nn.Module:
    state_dict = load(name, download_root=download_root)
    # shape = [d_model(channels), 3, patch_size, path_size]
    width = state_dict["visual.conv1.weight"].shape[0]
    # MultiheadAttention include four parameters:
    # ['in_proj_weight'(3 * d_model, d_model), 'in_proj_bias'(bias == 0), 'out_proj.weight'(input_d_model, out_d_model), 'out_proj.bias'(bias == 0)]
    # Get MultiheadAttention block nums, it same as number of layers.
    layers = len([k for k in state_dict.keys() if k.startswith(
        "visual.") and k.endswith(".attn.in_proj_weight")])
    patch_size = state_dict["visual.conv1.weight"].shape[-1]
    # grid_size = round(
    #     (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    # embed_dim = state_dict["text_projection"].shape[1]
    model = VisionTransformer(
        image_resolution, patch_size, width, layers, width // 64)
    # Get paramater from load state which keys starts with "visual."
    keys = set()
    for name, param in state_dict.items():
        if (not name.startswith("visual.")) or name == "visual.proj":
            continue
        # python 3.9+
        # name = name.removeprefix("visual.")
        name = name[len("visual."):]
        keys.add(name)
        if name == 'positional_embedding' and param.shape != model.positional_embedding.shape:
            param = resize_pos_embed(param, model.positional_embedding, model.grid_hight, model.grid_width)
        model.state_dict()[name].copy_(param)

    # make sure init all weights
    assert model.state_dict().keys() == keys
    if freeze_conv1:
        for layer in [model.conv1]:
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
    return model


# test
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_clip_vit("ViT-B/32").to(device)
    x = torch.randn((1, 3, 224, 224)).to(device)
    y = model(x)
    print(y)
