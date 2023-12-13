import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import model.losses as losses


class LossComputation(nn.Module):
    def __init__(self, cfg, num_classes, embed_dim):
        super().__init__()
        # self.epsilon = cfg.MODEL.EMBEDDING.EPSILON
        self.tau = cfg.LOSS.TAU
        self.alpha = cfg.LOSS.BCL_ALPHA
        self.scale_pos = cfg.LOSS.BCL_SCALE_ALPHA
        self.beta = cfg.LOSS.BCL_BETA
        self.scale_neg = cfg.LOSS.BCL_SCALE_BETA
        self.gama = cfg.LOSS.BCL_GAMA
        self.scale_sm_pos = cfg.LOSS.BCL_SCALE_GAMA
        self.pos_margin = cfg.LOSS.MARGIN_LOSS_POS_MARGIN
        self.sm_margin = cfg.LOSS.MARGIN_LOSS_SM_MARGIN
        self.hard_scale = cfg.LOSS.MARGIN_LOSS_SCALE
        self.sm_scale = cfg.LOSS.MARGIN_LOSS_SM_SCALE

        self.num_classes = num_classes
        self.classifier = nn.Linear(embed_dim, self.num_classes)
        nn.init.normal_(self.classifier.weight.data, std=0.001)
        nn.init.constant_(self.classifier.bias.data, val=0.0)
        self.losses = cfg.LOSS.LOSSES

    def forward(
        self,
        visual_embed,
        textual_embed,
        labels,
    ):
        pids = labels['pids']
        image_ids = labels['image_ids']
        loss = {}
        visual_norm = F.normalize(visual_embed, p=2, dim=1)
        textual_norm = F.normalize(textual_embed, p=2, dim=1)
        similarity = torch.matmul(visual_norm, textual_norm.t())

        if "BCL" in self.losses:
            loss["BCL"] = losses.boundary_constraints_loss(
                similarity,
                pids,
                image_ids,
                alpha=self.alpha,
                beta=self.beta,
                gama=self.gama,
                scale_pos=self.scale_pos,
                scale_neg=self.scale_neg,
                scale_sm_pos=self.scale_sm_pos,
            )
            
        if "HSMLM" in self.losses:
            loss["HSMLM"] = losses.hard_sample_mining_loss_m(
                similarity,
                pids,
                image_ids,
                pos_margin=self.pos_margin,
                sm_margin=self.sm_margin,
                sm_scale=self.sm_scale,
            ) * self.hard_scale
            
        if "HSMLD" in self.losses:
            loss["HSMLD"] = losses.hard_sample_mining_loss_d(
                similarity,
                pids,
                image_ids,
                pos_margin=self.pos_margin,
                sm_margin=self.sm_margin,
                sm_scale=self.sm_scale,
            ) * self.hard_scale

        if "id_loss" in self.losses:
            loss["id_loss"] = losses.id_loss(
                self.classifier,
                visual_norm,
                textual_norm,
                pids,
            )

        if "infonce_loss" in self.losses:
            loss["infonce_loss"] = losses.infonce_loss(
                visual_embed,
                textual_embed,
                T=self.tau,
            )

        return loss


# TODO: Clean Code, move cfg out
def build_loss_evaluator(cfg, num_classes, embed_dim):
    return LossComputation(cfg, num_classes, embed_dim)
