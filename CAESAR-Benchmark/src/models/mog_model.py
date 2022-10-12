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
from src.models.decoders import Decoders
from .classifier import Classifier
from src.models.shared_projections import *
from src.models.guided_projections import *
from src.models.guided_fusion import *
from src.models.multitask_model import *
from transformers import BertModel
from transformers import DistilBertModel
from transformers import AlbertModel
from src.models.losses import *
from src.utils.log import TextLogger
from src.utils.model_checkpointing import ModelCheckpointing
from src.utils.training_utils import *
from collections import Counter


class MOG_Model(pl.LightningModule):
    def __init__(self,
                 hparams):

        super(MOG_Model, self).__init__()

        self.hparams.update(dict(hparams))

        # build sub-module of the learning model
        self.view_encoders = nn.ModuleDict()
        for view_modality in self.hparams.view_modalities:
            self.view_encoders[view_modality] = ViewEncoder(self.hparams.view_encoder_name,
                                                        self.hparams.indi_modality_embedding_size)

        self.layer_norm = nn.LayerNorm(self.hparams.indi_modality_embedding_size)

        # self.verbal_instruction_encoder = BertModel.from_pretrained("bert-base-uncased")
        # self.verbal_instruction_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.verbal_instruction_encoder = AlbertModel.from_pretrained("albert-base-v2")
        self.verbal_embed_projection = nn.Linear(768, self.hparams.indi_modality_embedding_size) 
                                        # 768 is the BERT model last_hidden_state embed dimension

        self.shared_projection = SharedProjections(self.hparams.indi_modality_embedding_size)
        self.guided_projection = GuidedProjections(self.hparams.indi_modality_embedding_size,
                                            self.hparams.view_modalities,
                                            self.hparams.guided_projection_nhead,
                                            self.hparams.guided_projection_dropout)
        self.guided_fusion = GuidedFusion(self.hparams.indi_modality_embedding_size,
                                        self.hparams.view_modalities,
                                        self.hparams.guided_fusion_nhead,
                                        self.hparams.guided_fusion_dropout)

        if(self.hparams.is_decoders):
            self.decoders = Decoders(self.hparams.indi_modality_embedding_size,
                                self.hparams.view_modalities)
        
        self.multitask_model = MultitaskModel(self.hparams.indi_modality_embedding_size,
                                        self.hparams.task_list,
                                        self.hparams.multitask_modal_nhead,
                                        self.hparams.multitask_modal_dropout,
                                        self.hparams.view_modalities)

    
    def forward(self, batch):
        view_embeds = {}
        for view_modality in self.hparams.view_modalities:
            view_embeds[view_modality] = self.view_encoders[view_modality](batch[view_modality])
        
        tm_verbal_embed = self.verbal_instruction_encoder(**batch['verbal_instruction'])
        tm_verbal_embed = tm_verbal_embed.last_hidden_state
        verbal_embed = tm_verbal_embed[:,0,:]
        verbal_embed = self.verbal_embed_projection(verbal_embed)
        verbal_embed = self.layer_norm(verbal_embed)

        # shared_embeds = self.shared_projection(view_embeds)
        guided_embeds, guided_attn_embeds, guided_attn_weights = self.guided_fusion(view_embeds, verbal_embed)
        #guided_embeds and guided_attn_embeds are the same thing. guided_attn_embeds is a list and guided_embeds is dict where view_modalities are key
        private_embeds = self.guided_projection(view_embeds, verbal_embed)
        
        decoded_embeds = None
        if(self.hparams.is_decoders):
            decoded_embeds = self.decoders(view_embeds, guided_embeds, private_embeds)
        task_outputs, task_attn_weights = self.multitask_model(private_embeds, guided_attn_embeds, verbal_embed, batch['task_ids']) 

        return view_embeds, guided_embeds, private_embeds, decoded_embeds, task_outputs, task_attn_weights, guided_attn_weights
