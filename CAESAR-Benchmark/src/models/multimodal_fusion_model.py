import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusionModel(nn.Module):
    def __init__(self, fusion_model_name, embed_size, 
                fusion_model_nhead,
                fusion_model_dropout):

        super(MultimodalFusionModel, self).__init__()
        self.fusion_model_name = fusion_model_name
        self.embed_size = embed_size
        self.fusion_model_nhead = fusion_model_nhead
        self.fusion_model_dropout = fusion_model_dropout

        if(('self_attention' in self.fusion_model_name) or ('cross_attention' in self.fusion_model_name)):
            self.fusion_model = nn.MultiheadAttention(embed_dim=self.embed_size,
                                            num_heads=self.fusion_model_nhead,
                                            dropout=self.fusion_model_dropout,
                                            batch_first=True)

            self.fusion_model.apply(self.init_weights)
    
    def forward(self, hidden_states, verbal_embed): 

        if(self.fusion_model_name=='concat'):
            fused_embeds = torch.cat(hidden_states, dim=-1).contiguous()
        elif(self.fusion_model_name=='sum'):
            hidden_states = torch.stack(hidden_states, dim=1).contiguous()
            fused_embeds = torch.sum(hidden_states, dim=1).squeeze(dim=1).contiguous()
        elif(self.fusion_model_name=='self_attention'):
            hidden_states = torch.stack(hidden_states, dim=1).contiguous()
            fused_embeds, attn_weights = self.fusion_model(hidden_states,
                                                    hidden_states, 
                                                    hidden_states)
            fused_embeds = torch.sum(fused_embeds, dim=1).squeeze(dim=1).contiguous()  

        elif(self.fusion_model_name=='cross_attention'):
            verbal_embed = verbal_embed.unsqueeze(dim=1).contiguous()
            hidden_states = torch.stack(hidden_states, dim=1).contiguous()
            fused_embeds, attn_weights = self.fusion_model(verbal_embed,
                                                    hidden_states, 
                                                    hidden_states)
            fused_embeds = fused_embeds.squeeze(dim=1).contiguous()     
            verbal_embed = verbal_embed.squeeze(dim=1).contiguous()    
        elif(self.fusion_model_name=='self_attention_concat'):
            hidden_states = torch.stack(hidden_states, dim=1).contiguous()
            fused_embeds, attn_weights = self.fusion_model(hidden_states,
                                                    hidden_states, 
                                                    hidden_states)
            fused_embeds = fused_embeds.contiguous().view(fused_embeds.shape[0], -1).contiguous()

        elif(self.fusion_model_name=='cross_attention_concat'):
            verbal_embed = verbal_embed.unsqueeze(dim=1).contiguous()
            hidden_states = torch.stack(hidden_states, dim=1).contiguous()
            fused_embeds, attn_weights = self.fusion_model(verbal_embed,
                                                    hidden_states, 
                                                    hidden_states)
            fused_embeds = fused_embeds.contiguous().view(fused_embeds.shape[0], -1).contiguous()   
            verbal_embed = verbal_embed.squeeze(dim=1).contiguous()                                           

        fused_embeds = F.relu(fused_embeds)

        return fused_embeds

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)
