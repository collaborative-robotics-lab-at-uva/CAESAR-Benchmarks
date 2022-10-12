import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskGuidedFusion(nn.Module):
    def __init__(self, embed_size, 
                task_guided_fusion_nhead,
                task_guided_fusion_dropout):

        super(TaskGuidedFusion, self).__init__()
        self.embed_size = embed_size
        self.task_guided_fusion_nhead = task_guided_fusion_nhead
        self.task_guided_fusion_dropout = task_guided_fusion_dropout

        self.fusion_model = nn.MultiheadAttention(embed_dim=self.embed_size,
                                            num_heads=self.task_guided_fusion_nhead,
                                            dropout=self.task_guided_fusion_dropout,
                                            batch_first=True)

        self.fusion_model.apply(self.init_weights)
    
    def forward(self, embeds, task_embed): 
        task_embed = task_embed.unsqueeze(dim=1).contiguous()
        guided_embed, attn_weights = self.fusion_model(task_embed,
                                        embeds, 
                                        embeds)
        guided_embed = guided_embed.squeeze(dim=1).contiguous()
        task_embed = task_embed.squeeze(dim=1).contiguous()
        return guided_embed, attn_weights

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)
