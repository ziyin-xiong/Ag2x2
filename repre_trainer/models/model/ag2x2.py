from typing import Dict
from einops import rearrange
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from models.base import MODEL
import timm
from lora import LoRA_ViT_timm


@MODEL.register()
class AG2X2(nn.Module):
    # a copy for r3m model architecture
    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        super(AG2X2, self).__init__()
        self.d_emb = cfg.d_emb
        self.backbone_type = cfg.backbone_type
        self.similarity_type = cfg.similarity_type
        self.num_negatives = cfg.num_negatives
        self.loss_weight = cfg.loss_weight

        self.mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, a=-0.05, b=0.05)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, a=-0.05, b=0.05)
        self.missing_hand_embedding = nn.Parameter(torch.randn(1, 2))
        nn.init.normal_(self.missing_hand_embedding, std=.01)

        self.normlayer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.backbone_type == 'vit':
            vit_model = timm.create_model('vit_large_patch16_224_in21k', pretrained=True)
            self.backbone = LoRA_ViT_timm(vit_model=vit_model, r=4, alpha=4, num_classes=1024)
            self.last = nn.Linear(1056, self.d_emb)
        else:
            raise NotImplementedError

    def forward(self, data: Dict) -> torch.Tensor:
        imgs = data['imgs']
        s0_ind = data['s0_ind']
        s1_ind = data['s1_ind']
        s2_ind = data['s2_ind']

        if not torch.is_tensor(imgs):
            imgs = torch.stack(imgs)
        if imgs.shape[2:] != (3, 224, 224):
            preprocess = nn.Sequential(
                transforms.Resize(224, antialias=True),
                self.normlayer,
            )
        else:
            preprocess = nn.Sequential(
                self.normlayer,
            )
        imgs = preprocess(imgs)
        B, T = imgs.shape[:2]
        imgs = imgs.reshape(B*T, *imgs.shape[2:])
        feats = self.backbone(imgs)
        hands = data['hands']
        hand_num = data['hand_num']
        if not torch.is_tensor(hands):
            hands = torch.stack(hands)
        hands = hands.reshape(B*T, *hands.shape[2:])  # [B, 2, 21, 2]
        hands = hands.mean(dim=2)  # [B, 2, 2]
        if not torch.is_tensor(hand_num):
            hand_num = torch.stack(hand_num)
        hand_num = hand_num.reshape(B*T, *hand_num.shape[2:])
        hands_flat = hands.view(B*T, 2, -1).float()
        hand_indices = torch.arange(2).unsqueeze(0).expand(B*T, -1).to(hands_flat.get_device())  # Shape: [B, 2]
        mask = (hand_indices < hand_num).unsqueeze(-1).float()  # Shape: [B, 2, 1]
        missing_hand_mask = 1. - mask  # Shape: [B, 2, 1]
        missing_hand_embedding = self.missing_hand_embedding.expand(B*T*2, 2)
        missing_hand_embedding = missing_hand_embedding.reshape(B*T, 2, 2)
        hand_repre = hands_flat * mask + missing_hand_embedding * missing_hand_mask  # Shape: [B, 2, 2]
        hand_embeds = self.mlp(hand_repre)  # Shape: [B, 2, embed_dim=32]
        feats = torch.cat((feats, hand_embeds.sum(dim=1)), dim=1)  # Shape: [B, 1024+32]
        embs = self.last(feats)
        embs = embs.reshape(B, T, *embs.shape[1:])
        emb_s0 = embs[:, 0]
        emb_s1 = embs[:, 1]
        emb_s2 = embs[:, 2]

        #* compute metrics and full loss
        full_loss = 0
        metrics = dict()

        #* 1. Embdedding Norm loss
        loss_l1 = torch.linalg.norm(embs, ord=1, dim=-1).mean()
        loss_l2 = torch.linalg.norm(embs, ord=2, dim=-1).mean()
        full_loss += self.loss_weight.l1norm * loss_l1
        full_loss += self.loss_weight.l2norm * loss_l2
        metrics['loss_l1'] = loss_l1.item()
        metrics['loss_l2'] = loss_l2.item()

        #* 2. TCN Loss
        sim_0_1 = self.similarity(emb_s0, emb_s1)
        sim_1_2 = self.similarity(emb_s1, emb_s2)
        sim_0_2 = self.similarity(emb_s0, emb_s2)

        # negative samples
        sim_s0_neg = []
        sim_s2_neg = []
        perm = [i for i in range(B)]
        for _ in range(self.num_negatives):
            perm = [(i_perm + 1) % B for i_perm in perm]
            emb_s0_shuf = emb_s0[perm]
            emb_s2_shuf = emb_s2[perm]
            sim_s0_neg.append(self.similarity(emb_s0_shuf, emb_s0))
            sim_s2_neg.append(self.similarity(emb_s2_shuf, emb_s2))
        sim_s0_neg = torch.stack(sim_s0_neg, dim=-1)
        sim_s2_neg = torch.stack(sim_s2_neg, dim=-1)

        tcn_loss_1 = -torch.log(1e-6 + (torch.exp(sim_1_2) / (1e-6 + torch.exp(sim_0_2) + torch.exp(sim_1_2) + torch.exp(sim_s2_neg).sum(-1))))
        tcn_loss_2 = -torch.log(1e-6 + (torch.exp(sim_0_1) / (1e-6 + torch.exp(sim_0_1) + torch.exp(sim_0_2) + torch.exp(sim_s0_neg).sum(-1))))
        
        tcn_loss = ((tcn_loss_1 + tcn_loss_2) / 2.0).mean()
        metrics['loss_tcn'] = tcn_loss.item()
        metrics['alignment'] = (1.0 * (sim_0_2 < sim_1_2) * (sim_0_1 > sim_0_2)).float().mean().item()

        #* compute full loss
        full_loss += self.loss_weight.tcn * tcn_loss
        metrics['full_loss'] = full_loss.item()

        return {'loss': full_loss, 'metrics': metrics}
    
    def embedding(self, imgs: torch.Tensor) -> torch.Tensor:
        """ Embedding function
        """
        if imgs.shape[1:] != (3, 224, 224):
            preprocess = nn.Sequential(
                transforms.Resize(224, antialias=True),
                self.normlayer,
            )
        else:
            preprocess = nn.Sequential(
                self.normlayer,
            )
        imgs = preprocess(imgs)
        embs = self.backbone(imgs)
        return embs

    def similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Similarity function
        """
        if self.similarity_type == 'l2':
            d = -torch.linalg.norm(x - y, dim=-1)
            return d
        elif self.similarity_type == 'cosine':
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
            d = torch.einsum('...i,...i->...', x, y)
            return d
        else:
            raise NotImplementedError
