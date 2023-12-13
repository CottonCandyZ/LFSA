from prettytable import PrettyTable
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import numpy as np
import torchmetrics


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices

def jaccard(a_list, b_list):
        return float(len(set(a_list) & set(b_list))) / float(len(set(a_list) | set(b_list)))
    
    
def jaccard_mat(row_nn, col_nn):
    jaccard_sim = np.zeros((row_nn.shape[0], col_nn.shape[0]))
    # FIXME: need optimization
    for i in range(row_nn.shape[0]):
        for j in range(col_nn.shape[0]):
            jaccard_sim[i, j] = jaccard(row_nn[i], col_nn[j])
    return torch.from_numpy(jaccard_sim)

def k_reciprocal(q_feats, g_feats, neighbor_num=5, alpha=0.05):
    qg_sim = torch.matmul(q_feats, g_feats.t())  # q * g
    gg_sim = torch.matmul(g_feats, g_feats.t())  # g * g

    qg_indices = torch.argsort(qg_sim, dim=1, descending=True)
    gg_indices = torch.argsort(gg_sim, dim=1, descending=True)

    qg_nn = qg_indices[:, :neighbor_num]  # q * n
    gg_nn = gg_indices[:, :neighbor_num]  # g * n

    jaccard_sim = jaccard_mat(qg_nn.cpu().numpy(), gg_nn.cpu().numpy())  # q * g
    jaccard_sim = jaccard_sim.to(qg_sim.device)
    return alpha * jaccard_sim  # q * g
class Evaluator():
    def __init__(self, val_loaders, type="default"):
        self.img_loader = val_loaders[0] # gallery
        self.txt_loader = val_loaders[1] # query
        self.logger = logging.getLogger("LFSA.eval")
        self.type=type
        

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in tqdm(self.txt_loader):
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption).float()
            qids.append(pid.view(-1)) # flatten
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0).to(device)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in tqdm(self.img_loader):
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img).float()
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0).to(device)
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids
 
    
    def eval(self, model, save=False, i2t_metric=False):
        
        qfeats, gfeats, qids, gids = self._compute_embedding(model)
        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features        
        similarity = qfeats @ gfeats.t()
            
        # rvn_mat = k_reciprocal(qfeats, gfeats)
        rvn_mat = 0

        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=(similarity + rvn_mat), q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.cpu().numpy(), t2i_mAP.cpu().numpy(), t2i_mINP.cpu().numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.cpu().numpy(), i2t_mAP.cpu().numpy(), i2t_mINP.cpu().numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))
        
        save_dic = {
            'sim': similarity.cpu(),
            'cap_ids': qids.cpu(),
            'img_ids': gids.cpu(),
        }
        
        if save:
            return save_dic
        return t2i_cmc[0].item(), t2i_mAP
    
        
        
        
        