import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import math
from src.configs import config
from src.utils.log import *
from src.models.view_encoder import ViewEncoder
from src.models.view_context_encoder import ViewContextEncoder
from src.models.base_multitask_model import *
from src.models.base_multitask_model_vanila import *
from src.models.base_multitask_transformer_model import *
from src.models.view_context_guided_fusion import *
from transformers import BertModel
from transformers import DistilBertModel
from transformers import AlbertModel
from src.models.losses import *
from src.utils.log import TextLogger
from src.utils.model_checkpointing import ModelCheckpointing
from src.utils.training_utils import *
from collections import Counter


class BaseModel(pl.LightningModule):
    def __init__(self,
                 hparams):

        super(BaseModel, self).__init__()

        self.hparams.update(dict(hparams))

        # build sub-module of the learning model
        if(self.hparams.is_bbox_embed):
            self.bbox_view_encoder = ViewEncoder(self.hparams.view_encoder_name,
                                        self.hparams.indi_modality_embedding_size)

        self.view_context_encoder = ViewContextEncoder(self.hparams.view_encoder_name,
                                    self.hparams.indi_modality_embedding_size)

        self.bbox_cord_encoder = nn.Linear(4, self.hparams.indi_modality_embedding_size) 
        self.view_context_guided_fusion = ViewContextGuidedFusion(self.hparams.view_modalities,
                                                self.hparams.indi_modality_embedding_size,
                                                self.hparams.multitask_modal_nhead,
                                                self.hparams.multitask_modal_dropout)
                                                #may need to config the nhead and dropout for this model


        # self.verbal_instruction_encoder = BertModel.from_pretrained("bert-base-uncased")
        # self.verbal_instruction_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.verbal_instruction_encoder = AlbertModel.from_pretrained("albert-base-v2")
        self.verbal_embed_projection = nn.Linear(768, self.hparams.indi_modality_embedding_size) 
                                        # 768 is the BERT model last_hidden_state embed dimension
        self.layer_norm = nn.LayerNorm(self.hparams.indi_modality_embedding_size)

        if('transformer' in self.hparams.model_name):
            self.multitask_model= BaseMultitaskTransformerModel(self.hparams.indi_modality_embedding_size,
                                        self.hparams.task_list,
                                        self.hparams.multitask_modal_nhead,
                                        self.hparams.multitask_modal_dropout,
                                        self.hparams.view_modalities,
                                        is_bbox_embed=self.hparams.is_bbox_embed)
        elif('vanila' in self.hparams.model_name):
            self.multitask_model= BaseMultitaskModelVanila(2*self.hparams.indi_modality_embedding_size,
                                        self.hparams.task_list,
                                        self.hparams.multitask_modal_nhead,
                                        self.hparams.multitask_modal_dropout,
                                        self.hparams.view_modalities)

        else:
            self.multitask_model = BaseMultitaskModel(2*self.hparams.indi_modality_embedding_size,
                                        self.hparams.task_list,
                                        self.hparams.multitask_modal_nhead,
                                        self.hparams.multitask_modal_dropout,
                                        self.hparams.view_modalities)

    
    def forward(self, batch):
        bbox_view_embeds = {}
        view_context_embeds = {}
        bbox_cord_embeds = {}
        for view_modality in self.hparams.view_modalities:
            view_context_embeds[f'{view_modality}'] = self.view_context_encoder(batch[f'{view_modality}_context'])
            view_context_embeds[f'{view_modality}'] = self.layer_norm(view_context_embeds[f'{view_modality}'])
            if(self.hparams.is_bbox_embed):
                bbox_view_embeds[view_modality] = self.bbox_view_encoder(batch[f'{view_modality}_bboxes'])
                bbox_cord_embeds[f'{view_modality}'] = self.bbox_cord_encoder(batch[f'{view_modality}_bboxes_cord'])

        tm_verbal_embed = self.verbal_instruction_encoder(**batch['verbal_instruction'])
        tm_verbal_embed = tm_verbal_embed.last_hidden_state
        verbal_embed = tm_verbal_embed[:,0,:]
        verbal_embed = self.verbal_embed_projection(verbal_embed)
        verbal_embed = F.relu(verbal_embed)
        verbal_embed = self.layer_norm(verbal_embed)
        
        if('transformer' in self.hparams.model_name):
            task_outputs, task_attn_weights = self.multitask_model(bbox_view_embeds,
                                                        bbox_cord_embeds, 
                                                        view_context_embeds,
                                                        verbal_embed, 
                                                        batch['task_ids']) 
        else:
            fused_bbox_view_embeds = self.view_context_guided_fusion(bbox_view_embeds, 
                                                            bbox_cord_embeds, 
                                                            view_context_embeds,
                                                            batch['bboxes_mask'])

            task_outputs, task_attn_weights = self.multitask_model(fused_bbox_view_embeds, verbal_embed, batch['task_ids']) 

        return task_outputs, task_attn_weights
