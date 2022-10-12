import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ViewPosEncoder(nn.Module):
    def __init__(self, model_name, embed_size):
        super(ViewPosEncoder, self).__init__()

        self.embed_size = embed_size

        original_model = models.resnet18(pretrained=True)
        original_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.encoder = nn.Sequential(*list(original_model.children())[:-1])

        self.embed_projection = nn.Linear(512, self.embed_size)
        self.layer_norm = nn.LayerNorm(self.embed_size)

    def forward(self, x):
        batch_size = x.size(0)
        embed = self.encoder(x)
        embed = embed.view(batch_size, -1)
        embed = self.embed_projection(embed)
        embed = F.relu(self.layer_norm(embed))
        return embed