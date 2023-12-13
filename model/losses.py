import torch
import torch.nn as nn
import torch.nn.functional as F

def global_align_loss(
    visual_embed,
    textual_embed,
    labels,
    alpha=0.6,
    beta=0.4,
    scale_pos=10,
    scale_neg=40,
):
    batch_size = labels.size(0)
    visual_norm = F.normalize(visual_embed, p=2, dim=1)
    textual_norm = F.normalize(textual_embed, p=2, dim=1)
    similarity = torch.matmul(visual_norm, textual_norm.t())
    labels_ = (
        labels.expand(batch_size, batch_size)
        .eq(labels.expand(batch_size, batch_size).t())
        .float()
    )

    pos_inds = labels_ == 1
    neg_inds = labels_ == 0
    loss_pos = torch.log(1 + torch.exp(-scale_pos * (similarity[pos_inds] - alpha)))
    loss_neg = torch.log(1 + torch.exp(scale_neg * (similarity[neg_inds] - beta)))
    loss = (loss_pos.sum() + loss_neg.sum()) * 2.0

    loss /= batch_size
    return loss

def boundary_constraints_loss(
    similarity,
    labels,
    img_labels,
    alpha=0.6,
    beta=0.5,
    gama=0.55,
    scale_pos=10,
    scale_neg=40,
    scale_sm_pos=5,
):
    batch_size = labels.size(0)
    labels_ = (
        labels.expand(batch_size, batch_size)
        .eq(labels.expand(batch_size, batch_size).t())
        .float()
    )
    labels_img_ = (
        img_labels.expand(batch_size, batch_size)
        .eq(img_labels.expand(batch_size, batch_size).t())
        .float()
    )
    pos_inds = labels_img_ == 1
    
    sm_pos_inds = (labels_ == 1) != pos_inds
    neg_inds = labels_ == 0
      
    loss_pos = torch.log(1 + torch.exp(-scale_pos * (similarity[pos_inds] - alpha)))
    loss_neg = torch.log(1 + torch.exp(scale_neg * (similarity[neg_inds] - beta)))
    loss_sm_pos = torch.log(1 + torch.exp(-scale_sm_pos * (similarity[sm_pos_inds] - gama)))
    
    loss = (loss_pos.sum() + loss_neg.sum() + loss_sm_pos.sum()) * 2.0

    loss /= batch_size
    return loss

def hard_sample_mining_loss_m(
    similarity,
    labels,
    img_labels,
    pos_margin,
    sm_margin,
    sm_scale=0.5,
):
    batch_size = labels.size(0)
    labels_ = (
        labels.expand(batch_size, batch_size)
        .eq(labels.expand(batch_size, batch_size).t())
        .float()
    )
    labels_img_ = (
        img_labels.expand(batch_size, batch_size)
        .eq(img_labels.expand(batch_size, batch_size).t())
        .float()
    )
    pos_inds = labels_img_ == 1
    
    sm_pos_inds = (labels_ == 1) != pos_inds
    neg_inds = labels_ == 0
    neg_sim = torch.max(similarity * neg_inds, dim=0)[0].expand(batch_size, batch_size)
    diff = neg_sim - similarity
    pos_diff = diff[pos_inds]
    sm_diff = diff[sm_pos_inds]
    
    
    neg_sim_row = torch.max(similarity * neg_inds, dim=1)[0].expand(batch_size, batch_size).T
    diff_col = neg_sim_row - similarity
    pos_diff_col = diff_col[pos_inds]
    sm_diff_col = diff_col[sm_pos_inds]
    
    # loss = torch.nn.functional.relu(neg_sim - similarity[torch.eye(batch_size, dtype=torch.bool)] + margin).sum() / batch_size
    pos_loss = torch.nn.functional.relu(pos_diff + pos_margin).sum()
    sm_loss = torch.nn.functional.relu(sm_diff + sm_margin).sum()
    pos_loss_col = torch.nn.functional.relu(pos_diff_col + pos_margin).sum()
    sm_loss_col = torch.nn.functional.relu(sm_diff_col + sm_margin).sum()
    
    loss = (pos_loss + pos_loss_col + (sm_loss + sm_loss_col) * sm_scale) / batch_size
    return loss



def hard_sample_mining_loss_d(
    similarity,
    labels,
    img_labels,
    pos_margin,
    sm_margin,
    sm_scale=0.5,
):
    batch_size = labels.size(0)
    labels_ = (
        labels.expand(batch_size, batch_size)
        .eq(labels.expand(batch_size, batch_size).t())
        .float()
    )
    labels_img_ = (
        img_labels.expand(batch_size, batch_size)
        .eq(img_labels.expand(batch_size, batch_size).t())
        .float()
    )
    pos_inds = labels_img_ == 1
    
    sm_pos_inds = (labels_ == 1) != pos_inds
    neg_inds = labels_ == 0
    neg_sim = (similarity * neg_inds).T
    neg_sim_prob = ((similarity + 1) * neg_inds).T
    hard_index = []
    for i in range(batch_size):
        indx = torch.multinomial(neg_sim_prob[i], 1).item()
        hard_index.append(indx)
    
    neg_sim_selected = neg_sim[range(batch_size), hard_index].expand(batch_size, batch_size)
    diff = neg_sim_selected - similarity
    pos_diff = diff[pos_inds]
    sm_diff = diff[sm_pos_inds]
    
    
    neg_sim_col = neg_sim.T
    neg_sim_prob_col = neg_sim_prob.T
    hard_index_col = []
    for i in range(batch_size):
        indx = torch.multinomial(neg_sim_prob_col[i], 1).item()
        hard_index_col.append(indx)

    neg_sim_selected_col = neg_sim_col[range(batch_size), hard_index_col].expand(batch_size, batch_size).T
    diff_col = neg_sim_selected_col - similarity
    pos_diff_col = diff_col[pos_inds]
    sm_diff_col = diff_col[sm_pos_inds]
    
    
    
    # loss = torch.nn.functional.relu(neg_sim - similarity[torch.eye(batch_size, dtype=torch.bool)] + margin).sum() / batch_size
    pos_loss = torch.nn.functional.relu(pos_diff + pos_margin).sum()
    sm_loss = torch.nn.functional.relu(sm_diff + sm_margin).sum()
    pos_loss_col = torch.nn.functional.relu(pos_diff_col + pos_margin).sum()
    sm_loss_col = torch.nn.functional.relu(sm_diff_col + sm_margin).sum()
    loss = (pos_loss + pos_loss_col + (sm_loss + sm_loss_col) * sm_scale) / batch_size
    return loss


def id_loss(classifier, visual_embed, textual_embed, labels):
    """Compare to instance_loss"""
    criterion = nn.CrossEntropyLoss(reduction="mean")
    image_logits = classifier(visual_embed.half()).float()
    text_logits = classifier(textual_embed.half()).float()
    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    return loss / 2

def infonce_loss(
    visual_embed,
    textual_embed,
    T=0.1,
):
    batch_size = visual_embed.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64, device=visual_embed.device)
    visual_norm = F.normalize(visual_embed, p=2, dim=1)
    textual_norm = F.normalize(textual_embed, p=2, dim=1)
    image_proj_text = torch.matmul(visual_norm, textual_norm.t()) / T
    text_proj_image = image_proj_text.t()
    loss_i = F.cross_entropy(image_proj_text, labels)
    loss_t = F.cross_entropy(text_proj_image, labels)
    loss = (loss_i +  loss_t)/2

    return loss