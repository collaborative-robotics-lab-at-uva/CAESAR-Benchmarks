import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ViewContextEncoder(nn.Module):
    def __init__(self, model_name, embed_size):
        super(ViewContextEncoder, self).__init__()

        self.embed_size = embed_size

        if(model_name=='resnet50'):
            original_model = models.resnet50(pretrained=True)
            num_ftrs = 2048
        elif(model_name=='resnet34'):
            original_model = models.resnet34(pretrained=True)
            num_ftrs = 512
        elif(model_name=='resnet18'):
            original_model = models.resnet18(pretrained=True)
            num_ftrs = 512
        else:
            original_model = models.resnet50(pretrained=True)
            num_ftrs = 2048

        self.encoder = nn.Sequential(*list(original_model.children())[:-1])

        self.embed_projection = nn.Linear(num_ftrs, self.embed_size)
        self.layer_norm = nn.LayerNorm(self.embed_size)

    def forward(self, x):
        batch_size = x.size(0)
        embed = self.encoder(x)
        embed = embed.view(batch_size, -1)
        embed = self.embed_projection(embed)
        embed = F.relu(self.layer_norm(embed))
        return embed