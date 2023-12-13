from collections import OrderedDict


import torch
from torch import nn

# from visualizer import get_local
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype

        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGLUE(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_haed: int, attn_mask: torch.Tensor = None, dropout: float = 0.) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_haed, dropout=dropout)
        # LayerNorm in model dim
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGLUE()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
    # @get_local('attn_map')
    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """Create Multihead attention block"""

        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        # x, attn_map = self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)
        # return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, dropout: float = 0.) -> None:
        """Transformer

        Args:
            width (int): Equal to d_model.
            layers (int): Nums of ResBlock.
            heads (int): Nums of multihead attention heads.
            attn_mask (torch.Tensor, optional): Attention mask. Defaults to None.
        """
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, dropout) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)
