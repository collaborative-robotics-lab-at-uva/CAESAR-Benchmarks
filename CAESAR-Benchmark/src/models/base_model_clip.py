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
from src.models.task_prediction_model import TaskPredictionModel
from src.models.classifier import Classifier
from collections import Counter


class BaseModelCLIP(pl.LightningModule):
    def __init__(self,
                 hparams):

        super(BaseModelCLIP, self).__init__()

        self.hparams.update(dict(hparams))

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.set_parameter_requires_grad(self.clip_model, True)


        self.task_prediction_model = nn.ModuleDict()
        for task_name in self.hparams.task_list:
            if(task_name != config.instruction_valid_task_tag and task_name != config.ambiguity_recognition_task_tag):
                self.task_prediction_model[task_name] = TaskPredictionModel(2*self.hparams.indi_modality_embedding_size)
            else:
                self.task_prediction_model[task_name] = Classifier(2*len(self.hparams.view_modalities)*self.hparams.indi_modality_embedding_size, 2)

    
    def forward(self, batch):
        task_outputs = {}
        task_attn_weights = {}
        
        # for task_name in self.hparams.task_list:
        #     view_outputs = []
        #     for view_modality in self.hparams.view_modalities:
        #         outputs = self.clip_model(**batch[view_modality])
        #         logits = outputs.logits_per_image
        #         view_outputs.append(logits)
        #     view_outputs = torch.stack(view_outputs, dim=1).contiguous()
        #     view_outputs = torch.sum(view_outputs, dim=1).squeeze(dim=1).contiguous()
        #     view_outputs = view_outputs.softmax(dim=1)
        #     task_outputs[task_name] = view_outputs

        view_outputs = []
        for view_modality in self.hparams.view_modalities:
            outputs = self.clip_model(**batch[view_modality])
            text_embeds = outputs.text_embeds
            image_embeds = outputs.image_embeds
            view_outputs.append(text_embeds)
            view_outputs.append(image_embeds)
        view_outputs = torch.cat(view_outputs, dim=-1).contiguous()
        view_outputs = F.relu(view_outputs)

        for task_name in self.hparams.task_list:
            task_outputs[task_name] = self.task_prediction_model[task_name](view_outputs)

        return task_outputs, task_attn_weights

    def set_parameter_requires_grad(self, model, fine_tune):
        for param in model.parameters():
            param.requires_grad = fine_tune