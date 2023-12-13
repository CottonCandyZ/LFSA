import torch
from torch import nn

from .transformer import LayerNorm, Transformer
from tools.download_and_load_CLIP import load


# width == d_model
class CLIP_transformer(nn.Module):
    def __init__(self, context_length, vocab_size, width, heads, layers, dropout = 0.):
        super().__init__()

        self.context_length = context_length  # 77
        self.vocab_size = vocab_size  # 49408
        # add
        self.width = width  # 512

        self.transformer = Transformer(
            width, layers, heads, attn_mask=self.build_attention_mask(), dropout=dropout
        )

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, width))
        self.ln_final = LayerNorm(width)
            
    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype

    def build_attention_mask(self) -> torch.Tensor:
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text) -> torch.Tensor:
        
        # text.shape = [*, context_length]
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, width]

        x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # eot_token is the end of the sequence.
        # 
        # return without projection

        return x, text.argmax(dim=-1)


def build_clip_transformer(name: str, download_root: str, freeze_layer = 0, dropout = 0.) -> nn.Module:
    state_dict = load(name, download_root=download_root)
    # text_projection.shape = [d_model(width), embed_dim]
    
    # embed_dim = state_dict["text_projection"].shape[1]
    
    # positional_embddding.shape = [context_length, width]
    context_length = state_dict["positional_embedding"].shape[0]
    # token_embedding.weight.shape = [vocab_size, width]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    width = state_dict["ln_final.weight"].shape[0]
    heads = width // 64
    layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(
        "transformer.resblocks")))

    model = CLIP_transformer(context_length, vocab_size,
                             width, heads, layers, dropout=dropout)

    # for key in ["input_resolution", "context_length", "vocab_size"]:
    #     if key in state_dict:
    #         del state_dict[key]

    keys = set()

    for name, param in state_dict.items():
        if not name in model.state_dict().keys():
            continue
        keys.add(name)
        model.state_dict()[name].copy_(param)

    # make sure init all weights
    assert model.state_dict().keys() == keys
    
    if freeze_layer != 0:
        model.positional_embedding.requires_grad = False
        for parm in model.token_embedding.parameters():
            parm.requires_grad = False
        for i in range(freeze_layer):
            m = getattr(model.transformer.resblocks, str(i))
            for parm in m.parameters():
                parm.requires_grad = False
 
    return model