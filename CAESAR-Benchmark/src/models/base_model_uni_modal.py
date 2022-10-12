import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers import CLIPModel

from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import math
from src.configs import config
from src.utils.log import *
from src.models.view_encoder import ViewEncoder
from src.models.view_context_encoder import ViewContextEncoder
from src.models.view_pos_encoder import ViewPosEncoder
from src.models.base_multitask_model import *
from src.models.base_multitask_model_vanila import *
from src.models.base_multitask_transformer_model import *
from src.models.view_context_guided_fusion import *
from src.models.multimodal_fusion_model import *
from transformers import (BertModel, DistilBertModel,
                        AlbertModel, VisualBertModel, 
                        LxmertModel, VisionTextDualEncoderModel,
                        CLIPConfig)
from src.models.losses import *
from src.utils.log import TextLogger
from src.utils.model_checkpointing import ModelCheckpointing
from src.utils.training_utils import *
from src.models.task_prediction_model import TaskPredictionModel
from src.models.classifier import Classifier
from collections import Counter


class BaseModelUniModal(pl.LightningModule):

    def __init__(self,
                 hparams):

        super(BaseModelUniModal, self).__init__()

        self.hparams.update(dict(hparams))

        if('verbal' in self.hparams.model_name):
            self.embed_size = 768
        else:
            if(self.hparams.fusion_model_name=='concat'):
                self.embed_size = 512 * len(self.hparams.view_modalities)
            else:
                self.embed_size = 512

        if('verbal' in self.hparams.model_name):
            self.model = BertModel.from_pretrained("bert-base-uncased")
        
        else:
            self.view_context_encoder = ViewContextEncoder(self.hparams.view_encoder_name, 512)
            self.set_parameter_requires_grad(self.view_context_encoder, True)

            self.fusion_model = MultimodalFusionModel(self.hparams.fusion_model_name,
                                                    embed_size=512,
                                                    fusion_model_nhead=self.hparams.fusion_model_nhead,
                                                    fusion_model_dropout=self.hparams.fusion_model_dropout)

        self.task_prediction_model = nn.ModuleDict()
        for task_name in self.hparams.task_list:
            if(task_name != config.instruction_valid_task_tag and task_name != config.ambiguity_recognition_task_tag):
                self.task_prediction_model[task_name] = TaskPredictionModel(self.embed_size)
            else:
                self.task_prediction_model[task_name] = Classifier(self.embed_size, 2)

    
    def forward(self, batch):

        hidden_states = []
        if('verbal' in self.hparams.model_name):
            outputs = self.model(**batch['verbal_instruction'])
            verbal_embed = outputs.last_hidden_state
            fused_embeds = verbal_embed[:,0,:]
        else:
            for view_modality in self.hparams.view_modalities:
                hidden_state = self.view_context_encoder(batch[f'{view_modality}_context'])
                hidden_states.append(hidden_state)

            fused_embeds = self.fusion_model(hidden_states, None)
        

        task_outputs = {}
        task_attn_weights = {}
        for task_name in self.hparams.task_list:
            task_outputs[task_name] = self.task_prediction_model[task_name](fused_embeds)

        return task_outputs, task_attn_weights

    def set_parameter_requires_grad(self, model, fine_tune):
        for param in model.parameters():
            param.requires_grad = fine_tune

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)