import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
from .lora import LoRA_ViT_timm

from repres.base.base_repre import BaseRepre

class AG2X2(BaseRepre):

    def __init__(self, cfg_repre) -> None:
        super(AG2X2, self).__init__()
        self.goal_image = cfg_repre["goal_image"]
        self.goal_hand = cfg_repre["goal_hand"]
        self.device = cfg_repre["device"]
        self.batchsize = cfg_repre["batchsize"]
        self.d_emb = cfg_repre["d_emb"]
        self.backbone_type = cfg_repre['backbone_type']
        self.similarity_type = cfg_repre['similarity_type']
        if self.goal_image.dtype != torch.float32:
            raise TypeError("cfg_repre.goal_image.dtype must be torch.float32")
        
        self.normlayer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.backbone_type == 'vit':
            vit_model = timm.create_model('vit_large_patch16_224_in21k', pretrained=False)
            self.backbone = LoRA_ViT_timm(vit_model=vit_model, r=4, alpha=4, num_classes=1024)
            self.last = nn.Linear(1056, self.d_emb)
            self.mlp = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, 32)
            )
            self.missing_hand_embedding = nn.Parameter(torch.randn(1, 2))
            
        else:
            raise NotImplementedError
        
        #* load pre-trained ckpts
        if cfg_repre['ckpt_dir']:
            print(f'Require a pre-trained ckpt dir for representation model {self.__class__.__name__}')
        self.ckpt_dir = cfg_repre['ckpt_dir']
        print(f'Loading ckpt from {self.ckpt_dir}')
        checkpoint = torch.load(os.path.join(self.ckpt_dir, 'model.pth'))['model']
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v.clone().detach()  # Remove 'module.' from key (multi->single GPU)
            else:
                new_state_dict[k] = v.clone().detach()
        self.load_state_dict(new_state_dict)
        self.to(self.device)
        self.eval()
        #* compute goal image embedding
        self.goal_image = self.goal_image.to(self.device)
        self.goal_emb = self.embedding(self.goal_image.unsqueeze(0).permute(0, 3, 1, 2), self.goal_hand)  # (1, 1024)
    
    @torch.no_grad()
    def forward(self, x, hand):
        """
            x: [to torch.float32] (batch_size, 224, 224, 3)
            hand: [to torch.float32] (batch_size, 2, 2)
        """
        x = x.to(self.device)
        if x.dtype != torch.float32:
            raise TypeError("x.dtype must be torch.float32")
        x = x.permute(0, 3, 1, 2) # (batch_size, 3, 224, 224)
        embs = []
        for i in range(0, x.shape[0], self.batchsize):
            embs.append(self.embedding(x[i:i+self.batchsize], hand[i:i+self.batchsize]))
        embs = torch.cat(embs, dim=0)
        value = self.similarity(embs, self.goal_emb)
        return value, embs
    
    @torch.no_grad()
    def embedding(self, imgs: torch.Tensor, hand: torch.Tensor) -> torch.Tensor:
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
        feats = self.backbone(imgs)
        hand_embeds = self.mlp(hand)
        feats = torch.cat((feats, hand_embeds.sum(dim=1)), dim=1)  # Shape: [B, 1024+32]
        embs = self.last(feats)
        
        return embs

    def similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Similarity function #! nagative similarity 
        """
        if self.similarity_type == 'l2':
            d = -torch.linalg.norm(x - y, dim=-1)
            return -d
        elif self.similarity_type == 'cosine':
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
            d = torch.einsum('...i,...i->...', x, y)
            return -d
        else:
            raise NotImplementedError
