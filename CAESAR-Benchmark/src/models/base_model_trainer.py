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
from src.models.base_model import *
from src.models.base_model_uni_modal import *
from src.models.base_model_clip import *
from src.models.base_model_vl import *
from src.models.losses import *
from src.utils.log import TextLogger
from src.utils.model_checkpointing import ModelCheckpointing
from src.utils.training_utils import *
from collections import Counter


class Base_Model_Trainer(pl.LightningModule):
    def __init__(self, hparams):

        super(Base_Model_Trainer, self).__init__()

        self.hparams.update(vars(hparams))

        # if('clip' in self.hparams.model_name):
        #     self.model = BaseModelCLIP(self.hparams)
        if('_vl_' in self.hparams.model_name):
            self.model = BaseModelVL(self.hparams)
        elif('_unimodal_' in self.hparams.model_name):
            self.model = BaseModelUniModal(self.hparams)
        else:
            self.model = BaseModel(self.hparams)

        if(self.hparams.bbox_loss_type=='l1_loss'):
            self.loss_mse_fn = nn.L1Loss()
        elif(self.hparams.bbox_loss_type is not None):
            self.loss_mse_fn = nn.MSELoss()
        self.loss_instruction_classify_fn = nn.CrossEntropyLoss()

        # define the metrics and the checkpointing mode
        self.metrics_mode_dict = {'loss': 'min'}
        train_metrics_save_ckpt_mode = {'epoch_train_loss': True}
        valid_metrics_save_ckpt_mode = {'epoch_valid_loss': True}

        train_metrics_mode_dict = {}
        valid_metrics_mode_dict = {}
        train_metrics = []
        valid_metrics = []

        self.pl_metrics_list = []

        for task_name in self.hparams.task_list:
            if(task_name != config.instruction_valid_task_tag and task_name != config.ambiguity_recognition_task_tag):
                for map_name in ['map', 'map_50', 'map_75']: 
                    self.metrics_mode_dict[f'iou_{task_name}_{map_name}'] = 'max'
                valid_metrics_save_ckpt_mode[f'epoch_valid_iou_{task_name}'] = True
                self.pl_metrics_list.append(f'iou_{task_name}')
            else:
                self.metrics_mode_dict[f'accuracy_{task_name}'] = 'max'
                valid_metrics_save_ckpt_mode[f'epoch_valid_accuracy_{task_name}'] = False
                self.pl_metrics_list.append(f'accuracy_{task_name}')

        for metric in self.metrics_mode_dict:
            train_metrics.append(f'epoch_train_{metric}')
            valid_metrics.append(f'epoch_valid_{metric}')
            train_metrics_mode_dict[f'epoch_train_{metric}'] = self.metrics_mode_dict[metric]
            valid_metrics_mode_dict[f'epoch_valid_{metric}'] = self.metrics_mode_dict[metric]
            
        stages = ['train', 'valid', 'test']
        self.pl_metrics = nn.ModuleDict()
        for metric in self.pl_metrics_list:
            for stage in stages:
                self.hparams.num_activity_types = 2 # need to change, it is for testing
                self.pl_metrics[f'{stage}_{metric}'] = get_pl_metrics(metric,
                                                            bbox_format=self.hparams.bbox_format, 
                                                            num_classes=2)

        self.txt_logger = TextLogger(self.hparams.log_base_dir, 
                                    self.hparams.log_filename,
                                    print_console=True)
        self.train_model_checkpointing = ModelCheckpointing(self.hparams.model_save_base_dir,
                                                self.hparams.model_checkpoint_filename,
                                                train_metrics,
                                                train_metrics_save_ckpt_mode,
                                                train_metrics_mode_dict,
                                                self.txt_logger)
        
        self.valid_model_checkpointing = ModelCheckpointing(self.hparams.model_save_base_dir,
                                                self.hparams.model_checkpoint_filename,
                                                valid_metrics,
                                                valid_metrics_save_ckpt_mode,
                                                valid_metrics_mode_dict,
                                                self.txt_logger)

        self.test_log = None
        self.mm_embed = None
        self.module_out = None
    
    def forward(self, batch):
        return self.model(batch)

    def set_parameter_requires_grad(self, model, is_require):
        for param in model.parameters():
            param.requires_grad = is_require

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        return self.eval_step(batch, batch_idx, pre_log_tag='train')

    def training_epoch_end(self, outputs):
        results = cal_metrics(outputs, self.pl_metrics_list, 
                            self.pl_metrics, 
                            self.hparams.task_list,
                            stage_tag='train',
                            trainer=self.trainer, device=self.device)
        self.log_metrics(results)
        self.train_model_checkpointing.update_metric_save_ckpt(results, self.current_epoch, self.trainer)
        
    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, pre_log_tag='valid')

    def validation_epoch_end(self, outputs):
        results = cal_metrics(outputs, self.pl_metrics_list, 
                            self.pl_metrics, 
                            self.hparams.task_list,
                            stage_tag='valid',
                            trainer=self.trainer, device=self.device)
        self.log_metrics(results)
        self.valid_model_checkpointing.update_metric_save_ckpt(results, self.current_epoch, self.trainer)
        

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, pre_log_tag='test')

    def test_epoch_end(self, outputs):
        results = cal_metrics(outputs, self.pl_metrics_list, 
                            self.pl_metrics, 
                            self.hparams.task_list,
                            stage_tag='test',
                            trainer=self.trainer, device=self.device)
        self.log_metrics(results)
        self.test_log = results
        self.txt_logger.log(f'{str(results)}\n')
        
    def eval_step(self, batch, batch_idx, pre_log_tag):

        task_outputs, task_attn_weights = self(batch)
        loss = self.get_multitask_loss(task_outputs, batch)
        
        metric_results = {}
        batch_size = batch['task_ids'][f'{self.hparams.task_list[0]}_id'].shape[0]

        # first_time=True
        for task_name in self.hparams.task_list:
            # if(pre_log_tag=='train' and first_time):
            #     print('predict bbox', task_outputs[task_name][0])
            #     print('truth bbox', batch[f'{task_name}_target_bbox_cord'][0])
            # first_time = False
            
            if(task_name != config.instruction_valid_task_tag and task_name != config.ambiguity_recognition_task_tag):
                metric_key = f'{pre_log_tag}_iou_{task_name}'
                preds = []
                targets = []
                for i in range(batch_size):
                    preds.append(
                        dict(
                            boxes = task_outputs[task_name][i].unsqueeze(dim=0).contiguous(),
                            scores = torch.tensor([1.0]).to(self.device),
                            labels = torch.tensor([0]).to(self.device),
                        )
                    )
                    targets.append(
                        dict(
                            boxes = batch[f'{task_name}_target_bbox_cord'][i].unsqueeze(dim=0).contiguous(),
                            labels = torch.tensor([0]).to(self.device),
                        )
                    )
                metric_results[metric_key] = self.pl_metrics[metric_key](preds, targets)
            else:
                metric_key = f'{pre_log_tag}_accuracy_{task_name}'
                metric_results[metric_key] = self.pl_metrics[metric_key](task_outputs[task_name],
                                                                        batch[f'{task_name}_labels'])

        self.log(f'{pre_log_tag}_loss', loss)

        return {'loss': loss}

    def get_multitask_loss(self, task_outputs, batch):
        loss_task = 0.0
        for task_name in self.hparams.task_list:
            if(task_name != config.instruction_valid_task_tag and task_name != config.ambiguity_recognition_task_tag):
                loss_task += self.loss_mse_fn(task_outputs[task_name], batch[f'{task_name}_target_bbox_cord'])
            else:
                loss_task += self.loss_instruction_classify_fn(task_outputs[task_name], batch[f'{task_name}_labels'])

        return loss_task

    def configure_optimizers(self):
        model_params = self.parameters()
        optimizer = torch.optim.AdamW(model_params, lr=self.hparams.learning_rate)
        # optimizer = torch.optim.SGD(model_params, lr=self.hparams.learning_rate, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                        T_0=self.hparams.cycle_length,
                                                                        T_mult=self.hparams.cycle_mul)
        return [optimizer], [lr_scheduler]
    
    def log_metrics(self, results):
        for metric in results:
            if('all' in metric):
                self.log(metric, results[metric], prog_bar = True)
            else:
                self.log(metric, results[metric], prog_bar = True)