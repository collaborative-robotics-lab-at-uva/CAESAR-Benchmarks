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


class BaseModelVL(pl.LightningModule):

    def get_rl_detect_model_embed_size(self):

        extra_mul = 0
        if(self.hparams.is_bbox_cord_embed):
            extra_mul += len(self.hparams.view_modalities)
        if(self.hparams.is_bbox_image_mask_encode or ('drnet' in self.hparams.model_name)):
            extra_mul += len(self.hparams.view_modalities)

        if(('clip' in self.hparams.model_name) or ('vl_dual_encoder' in self.hparams.model_name)):
            if(('concat' in self.hparams.fusion_model_name) and (not ('cross_attention' in self.hparams.fusion_model_name))):
                embed_size = 512 * 2 * len(self.hparams.view_modalities)
                embed_size += 512 * extra_mul
            else:
                embed_size = 512
        elif(('drnet' in self.hparams.model_name)):
            if(('concat' in self.hparams.fusion_model_name) and (not ('cross_attention' in self.hparams.fusion_model_name))):
                embed_size = 768 * (len(self.hparams.view_modalities)+1)
                embed_size += 768 * extra_mul
            else:
                embed_size = 768
        elif(('late_fusion' in self.hparams.model_name)):
            if(('concat' in self.hparams.fusion_model_name) and (not ('cross_attention' in self.hparams.fusion_model_name))):
                embed_size = 768 * (len(self.hparams.view_modalities)+1)
                embed_size += 768 * extra_mul
            else:
                embed_size = 768
        else:
            if(('concat' in self.hparams.fusion_model_name) and (not ('cross_attention' in self.hparams.fusion_model_name))):
                embed_size = 768 * (len(self.hparams.view_modalities)+1) 
                embed_size += 768 * extra_mul
            else:
                embed_size = 768

        return embed_size

    def __init__(self,
                 hparams):

        super(BaseModelVL, self).__init__()

        self.hparams.update(dict(hparams))

        if(('clip' in self.hparams.model_name) or ('vl_dual_encoder' in self.hparams.model_name)):
            self.embed_size = 512
        elif(('late_fusion' in self.hparams.model_name) or ('drnet' in self.hparams.model_name)):
            self.embed_size = 768
        else:
            self.embed_size = 768

        # bbox view encoder
        if(self.hparams.is_bbox_embed)  or ('lxmert' in self.hparams.model_name):
            self.bbox_view_encoder = ViewEncoder(self.hparams.view_encoder_name,
                                        2048)
            self.set_parameter_requires_grad(self.bbox_view_encoder, True)
        
        # bbox cord encoder
        if(self.hparams.is_bbox_cord_embed):
            self.bbox_cord_encoder = nn.Linear(4, self.embed_size)

        # bbox image mask encoder
        if(self.hparams.is_bbox_image_mask_encode  or ('drnet' in self.hparams.model_name)):
            self.bbox_image_mask_encoder = ViewPosEncoder(self.hparams.view_encoder_name,
                                            self.embed_size)
            self.set_parameter_requires_grad(self.bbox_image_mask_encoder, True)

        # image context encoder
        if(('visual_bert' in self.hparams.model_name)):
            self.view_context_encoder = ViewContextEncoder(self.hparams.view_encoder_name,
                                        2048)
            self.set_parameter_requires_grad(self.view_context_encoder, True)
        elif(('late_fusion' in self.hparams.model_name) or ('drnet' in self.hparams.model_name)):
            self.view_context_encoder = ViewContextEncoder(self.hparams.view_encoder_name,
                                        self.embed_size)
            self.set_parameter_requires_grad(self.view_context_encoder, True)

        # VL models
        if('lxmert' in self.hparams.model_name):
            self.model = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
            self.model.pooler.activation = nn.ReLU()
            self.model.pooler.dense.apply(self.init_weights)
        elif('visual_bert' in self.hparams.model_name):
            self.model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
            self.model.pooler.activation = nn.ReLU()
            self.model.pooler.dense.apply(self.init_weights)
        elif('clip' in self.hparams.model_name):
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            # self.model = CLIPModel(CLIPConfig())
        elif('vl_dual_encoder' in self.hparams.model_name):
            self.model = VisionTextDualEncoderModel.from_vision_text_pretrained(
                                "google/vit-base-patch16-224", "bert-base-uncased")
        elif(('late_fusion' in self.hparams.model_name) or ('drnet' in self.hparams.model_name)):
            self.model = BertModel.from_pretrained("bert-base-uncased")

        self.set_parameter_requires_grad(self.model, True)

        if(('clip' in self.hparams.model_name) or ('vl_dual_encoder' in self.hparams.model_name)):
            self.fusion_model = MultimodalFusionModel(self.hparams.fusion_model_name,
                                                    embed_size=512,
                                                    fusion_model_nhead=self.hparams.fusion_model_nhead,
                                                    fusion_model_dropout=self.hparams.fusion_model_dropout)
        else:
            self.fusion_model = MultimodalFusionModel(self.hparams.fusion_model_name,
                                                    embed_size=768,
                                                    fusion_model_nhead=self.hparams.fusion_model_nhead,
                                                    fusion_model_dropout=self.hparams.fusion_model_dropout)

        rl_detect_model_embed_size = self.get_rl_detect_model_embed_size()
        
        self.task_prediction_model = nn.ModuleDict()
        for task_name in self.hparams.task_list:
            if(task_name != config.instruction_valid_task_tag and task_name != config.ambiguity_recognition_task_tag):
                self.task_prediction_model[task_name] = TaskPredictionModel(rl_detect_model_embed_size)
            else:
                self.task_prediction_model[task_name] = Classifier(rl_detect_model_embed_size, 2)

    
    def forward(self, batch):

        hidden_states = []
        verbal_embeds = []
        verbal_embed = None
        for view_modality in self.hparams.view_modalities:
            if(not (('clip' in self.hparams.model_name) or ('vl_dual_encoder' in self.hparams.model_name))):
                view_embeds = []
                if(('visual_bert' in self.hparams.model_name)):
                    view_context_embed = self.view_context_encoder(batch[f'{view_modality}_context'])
                    view_embeds.append(F.relu(view_context_embed))
                if(('late_fusion' in self.hparams.model_name) or ('drnet' in self.hparams.model_name)):
                    view_context_embed = self.view_context_encoder(batch[f'{view_modality}_context'])
                    hidden_states.append(F.relu(view_context_embed))

                if(self.hparams.is_bbox_embed) or ('lxmert' in self.hparams.model_name):
                    bbox_view_embed = self.bbox_view_encoder(batch[f'{view_modality}_bboxes'])
                    seq_len = bbox_view_embed.shape[1]
                    for i in range(seq_len):
                        view_embeds.append(bbox_view_embed[:,i,:])

                if(('visual_bert' in self.hparams.model_name) or ('lxmert' in self.hparams.model_name)):
                    view_embeds = torch.stack(view_embeds, dim=1).contiguous()
                vl_input = batch['verbal_instruction']

                # print('verbal mask', batch['verbal_instruction']['attention_mask'].shape)
                # print('view_embeds', view_embeds.shape)
                # print('visual mask', batch['bboxes_mask'][f'{view_modality}_bboxes_seq_mask'].shape)
                # print('bbox cord shape', batch[f'{view_modality}_bboxes_cord'].shape)

                if('lxmert' in self.hparams.model_name):
                    vl_input.update({
                        'visual_feats': view_embeds,
                        'visual_pos': batch[f'{view_modality}_bboxes_cord'],
                        'visual_attention_mask': batch['bboxes_mask'][f'{view_modality}_bboxes_seq_mask'],
                    })
                elif('visual_bert' in self.hparams.model_name):
                    vl_input.update({
                        'visual_embeds': view_embeds,
                        'visual_attention_mask': batch['bboxes_mask'][f'{view_modality}_bboxes_seq_mask'],
                    })
            elif(('clip' in self.hparams.model_name) or ('vl_dual_encoder' in self.hparams.model_name)):
                vl_input = batch[f'{view_modality}_processed_input']

            outputs = self.model(**vl_input)
            if('lxmert' in self.hparams.model_name):
                hidden_states.append(outputs.pooled_output)
            elif('visual_bert' in self.hparams.model_name):
                hidden_states.append(outputs.pooler_output)
            elif(('clip' in self.hparams.model_name) or ('vl_dual_encoder' in self.hparams.model_name)):
                verbal_embed = outputs.text_embeds
                if(self.hparams.fusion_model_name=='cross_attention'):
                    verbal_embeds.append(verbal_embed)
                else:
                    hidden_states.append(verbal_embed)
                hidden_states.append(outputs.image_embeds)

        if((self.hparams.fusion_model_name=='cross_attention') and (('clip' in self.hparams.model_name) or ('vl_dual_encoder' in self.hparams.model_name))):
            verbal_embeds = torch.stack(verbal_embeds, dim=1).contiguous()
            verbal_embed = torch.sum(verbal_embeds, dim=1).squeeze(dim=1).contiguous()

        if(('late_fusion' in self.hparams.model_name) or ('drnet' in self.hparams.model_name)):
            outputs = self.model(**batch['verbal_instruction'])
            verbal_embed = outputs.last_hidden_state
            verbal_embed = verbal_embed[:,0,:]
            hidden_states.append(verbal_embed)

        if(self.hparams.is_bbox_cord_embed):
            for view_modality in self.hparams.view_modalities:
                bbox_cord_embeds = self.bbox_cord_encoder(batch[f'{view_modality}_bboxes_cord'])

                for i in range(bbox_cord_embeds.shape[1]):
                    hidden_states.append(bbox_cord_embeds[:,i,:].contiguous())

        if(self.hparams.is_bbox_image_mask_encode or ('drnet' in self.hparams.model_name)):
            for view_modality in self.hparams.view_modalities:
                tm = batch[f'{view_modality}_bboxes_image_mask'].squeeze(dim=1).contiguous()
                pos_embeds = self.bbox_image_mask_encoder(tm)
                hidden_states.append(pos_embeds)

        # fusing embeds
        fused_embeds = self.fusion_model(hidden_states, verbal_embed)

        # print('fused_embeds', fused_embeds.shape)

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