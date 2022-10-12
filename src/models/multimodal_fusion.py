import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.keyless_attention import KeylessAttention


class MultimodalFusion(nn.Module):
    def __init__(self, embed_size, view_modalities):

        super(MultimodalFusion, self).__init__()
        self.embed_size = embed_size
        self.view_modalities = view_modalities

        self.fusion_model = KeylessAttention(self.embed_size)

        self.fusion_model_bn = nn.BatchNorm1d(len(self.view_modalities))
        self.fusion_model.apply(self.init_weights)
    
    def forward(self, guided_embeds, mm_view_embed, verbal_embed): 

        embed_list = [mm_view_embed, verbal_embed]
        for view_modality in self.view_modalities:
            embed_list.append(guided_embeds[view_modality])

        embed_list = torch.stack(embed_list, dim=1).contiguous()
        embed_list = F.relu(self.fusion_model_bn(embed_list))

        mm_embeds = self.fusion_model(embed_list)

        return mm_embeds 

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)
