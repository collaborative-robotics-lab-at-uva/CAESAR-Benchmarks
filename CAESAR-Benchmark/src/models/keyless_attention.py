import torch
import torch.nn as nn
import torch.nn.functional as F

class KeylessAttention(nn.Module):
    def __init__(self, feature_embed_size):
        super(KeylessAttention, self).__init__()

        self.feature_embed_size = feature_embed_size
        self.attention_module = nn.Conv1d(self.feature_embed_size, 1, 1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        weights = self.softmax(self.attention_module(x.transpose(1,2)).squeeze(1)).unsqueeze(-1)
        weights =  weights.expand_as(x)
        # output = torch.sum(x*weights, 1).squeeze(1)
        output = x*weights
        return output, weights