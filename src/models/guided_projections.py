import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedProjections(nn.Module):
    def __init__(self, embed_size, 
                view_modalities, 
                guided_projection_nhead,
                guided_projection_dropout):

        super(GuidedProjections, self).__init__()
        self.embed_size = embed_size
        self.view_modalities = view_modalities
        self.guided_projection_nhead = guided_projection_nhead
        self.guided_projection_dropout = guided_projection_dropout

        self.projection_models = nn.ModuleDict()
        for view_modality in self.view_modalities:
            self.projection_models[view_modality] = nn.MultiheadAttention(embed_dim=self.embed_size,
                                                    num_heads=self.guided_projection_nhead,
                                                    dropout=self.guided_projection_dropout,
                                                    batch_first=True)

            self.projection_models[view_modality].apply(self.init_weights)
        self.layer_norm = nn.LayerNorm(self.embed_size)

    def forward(self, view_embeds, verbal_embed): 
        verbal_embed = verbal_embed.unsqueeze(dim=1).contiguous()
        private_embeds = {}
        for view_modality in self.view_modalities:
            view_embeds[view_modality] = view_embeds[view_modality].unsqueeze(dim=1).contiguous()
            private_embeds[view_modality], _ = self.projection_models[view_modality](verbal_embed,
                                                        view_embeds[view_modality],
                                                        view_embeds[view_modality])
            private_embeds[view_modality] = F.relu(self.layer_norm(private_embeds[view_modality]))
            private_embeds[view_modality] = private_embeds[view_modality].squeeze(dim=1).contiguous()
            view_embeds[view_modality] = view_embeds[view_modality].squeeze(dim=1).contiguous()
        verbal_embed = verbal_embed.squeeze(dim=1).contiguous()
        return private_embeds

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)
