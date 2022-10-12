import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedFusion(nn.Module):
    def __init__(self, embed_size, 
                view_modalities,
                guided_fusion_nhead,
                guided_fusion_dropout):

        super(GuidedFusion, self).__init__()
        self.embed_size = embed_size
        self.view_modalities = view_modalities
        self.guided_fusion_nhead = guided_fusion_nhead
        self.guided_fusion_dropout = guided_fusion_dropout

        self.fusion_model = nn.MultiheadAttention(embed_dim=self.embed_size,
                                            num_heads=self.guided_fusion_nhead,
                                            dropout=self.guided_fusion_dropout,
                                            batch_first=True)

        self.fusion_model_bn = nn.BatchNorm1d(len(self.view_modalities))
        self.fusion_model.apply(self.init_weights)
    
    def forward(self, view_embeds, verbal_embed): 
        verbal_embed = verbal_embed.unsqueeze(dim=1).contiguous()
        view_embed_list = []
        for view_modality in self.view_modalities:
            view_embed_list.append(view_embeds[view_modality])

        view_embed_list = torch.stack(view_embed_list, dim=1).contiguous()
        view_embed_list = F.relu(self.fusion_model_bn(view_embed_list))

        guided_attn_embeds, guided_attn_weight = self.fusion_model(verbal_embed,
                                        view_embed_list,
                                        view_embed_list)
        
        guided_attn_embeds = F.relu(guided_attn_embeds)

        guided_embeds = {}
        for i in range(len(self.view_modalities)):
            guided_embeds[self.view_modalities[i]] = guided_attn_embeds[i]
        guided_attn_embeds = guided_attn_embeds.squeeze(dim=1).contiguous()
        return guided_embeds, guided_attn_embeds, guided_attn_weight

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)
