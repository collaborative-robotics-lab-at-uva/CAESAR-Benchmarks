import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskPredictionModel(nn.Module):
    def __init__(self, embed_size):

        super(TaskPredictionModel, self).__init__()
        self.embed_size = embed_size

        self.model = nn.Sequential(nn.Linear(self.embed_size, self.embed_size//2),
                            # nn.ReLU(),
                            nn.Linear(self.embed_size//2, self.embed_size//2),
                            # nn.ReLU(),
                            nn.Linear(self.embed_size//2, 4))

        self.model.apply(self.init_weights)
    
    def forward(self, embed): 
        return self.model(embed)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)
