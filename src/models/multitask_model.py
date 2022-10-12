import torch
import torch.nn as nn
import torch.nn.functional as F

from src.configs import config
from src.models.task_guided_fusion import TaskGuidedFusion
from src.models.task_prediction_model import TaskPredictionModel
from src.models.classifier import Classifier
from src.configs import config

class MultitaskModel(nn.Module):
    def __init__(self, embed_size, 
                task_list, 
                multitask_modal_nhead,
                multitask_modal_dropout,
                view_modalities,
                finetune_taskname=None):
        super(MultitaskModel, self).__init__()

        self.embed_size = embed_size
        self.task_list = task_list
        self.finetune_taskname = finetune_taskname
        self.multitask_modal_nhead = multitask_modal_nhead
        self.multitask_modal_dropout = multitask_modal_dropout
        self.view_modalities = view_modalities
        
        self.task_encoder = nn.ModuleDict()
        self.task_guided_encoder = nn.ModuleDict()
        self.task_prediction_model = nn.ModuleDict()
        for task_name in self.task_list:
            self.task_encoder[task_name] = nn.Embedding(6, self.embed_size, padding_idx=0)
            self.task_guided_encoder[task_name] = TaskGuidedFusion(self.embed_size,
                                                        self.multitask_modal_nhead,
                                                        self.multitask_modal_dropout)
            if(task_name != config.instruction_valid_task_tag and task_name != config.ambiguity_recognition_task_tag):
                self.task_prediction_model[task_name] = TaskPredictionModel(self.embed_size)
            else:
                self.task_prediction_model[task_name] = Classifier(self.embed_size, 2)
        if(self.finetune_taskname is not None):
            self.task_encoder[self.finetune_taskname] = nn.Embedding(5, self.embed_size, padding_idx=0)
            self.task_guided_encoder[self.hparams.finetune_taskname] = TaskGuidedFusion(self.embed_size,
                                                                        self.multitask_modal_nhead,
                                                                        self.multitask_modal_dropout)
            if(self.finetune_taskname != config.instruction_valid_task_tag or self.finetune_taskname != config.ambiguity_recognition_task_tag):
                self.task_prediction_model[self.finetune_taskname] = TaskPredictionModel(self.embed_size)
            else:
                self.task_prediction_model[self.finetune_taskname] = Classifier(self.embed_size, 2)
        
        self.layer_norm = nn.LayerNorm(self.embed_size)
        self.mm_attn_weight = None

    def forward(self, private_embeds, guided_mm_embed, verbal_embed, task_ids):
        
        embed_list = [guided_mm_embed, verbal_embed]
        for view_modality in self.view_modalities:
            embed_list.append(private_embeds[view_modality])

        embed_list = torch.stack(embed_list, dim=1).contiguous()
        embed_list = F.relu(embed_list)

        task_embeds = {}
        task_mm_embeds = {}
        task_attn_weights = {}
        task_outputs = {}
        if(self.finetune_taskname is None):
            for task_name in self.task_list:
                task_embeds[task_name] = self.task_encoder[task_name](task_ids[f'{task_name}_id'])
                task_embeds[task_name] = task_embeds[task_name].squeeze(dim=1).contiguous()
                task_embeds[task_name] = F.relu(self.layer_norm(task_embeds[task_name]))
            
            for task_name in self.task_list:
                task_mm_embeds[task_name], task_attn_weights[task_name] = self.task_guided_encoder[task_name](embed_list, task_embeds[task_name])
                task_outputs[task_name] = self.task_prediction_model[task_name](task_mm_embeds[task_name])
        else:
            task_embeds[finetune_taskname] = self.task_encoder[finetune_taskname](task_ids[f'{task_name}_id'])
            task_embeds[finetune_taskname] = F.relu(self.layer_norm(task_embeds[finetune_taskname]))
            task_mm_embeds[finetune_taskname], task_attn_weights[task_name] = self.task_guided_encoder[finetune_taskname](embed_list, task_embeds[finetune_taskname])
            task_outputs[task_name] = self.task_prediction_model[task_name](task_mm_embeds[task_name])

        return task_outputs, task_attn_weights

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_output1.weight)
        nn.init.constant_(self.fc_output1.bias, 0.)