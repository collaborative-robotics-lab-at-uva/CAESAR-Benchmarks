import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewContextGuidedFusion(nn.Module):
    def __init__(self, view_modalities, embed_size, 
                task_guided_fusion_nhead,
                task_guided_fusion_dropout):

        super(ViewContextGuidedFusion, self).__init__()
        self.view_modalities = view_modalities
        self.embed_size = embed_size
        self.task_guided_fusion_nhead = task_guided_fusion_nhead
        self.task_guided_fusion_dropout = task_guided_fusion_dropout

        self.fusion_model = nn.MultiheadAttention(embed_dim=2*self.embed_size,
                                            num_heads=self.task_guided_fusion_nhead,
                                            dropout=self.task_guided_fusion_dropout,
                                            batch_first=True)

        self.fusion_model.apply(self.init_weights)
    
    def forward(self, bbox_view_embeds, bbox_cord_embeds, view_context_embeds, bboxes_mask=None): 
        fused_bbox_view_embeds = {}
        for view_modality in self.view_modalities:
            fused_bbox_view_embed = torch.cat((bbox_view_embeds[view_modality], bbox_cord_embeds[view_modality]), -1)

            query = view_context_embeds[view_modality].unsqueeze(dim=1).contiguous()
            fused_bbox_view_embeds[view_modality], attn_weights = self.fusion_model(query,
                                            fused_bbox_view_embed, 
                                            fused_bbox_view_embed,
                                            key_padding_mask=bboxes_mask[f'{view_modality}_bboxes_seq_mask'])
            fused_bbox_view_embeds[view_modality] = fused_bbox_view_embeds[view_modality].squeeze(dim=1).contiguous()
        
        return fused_bbox_view_embeds

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)
