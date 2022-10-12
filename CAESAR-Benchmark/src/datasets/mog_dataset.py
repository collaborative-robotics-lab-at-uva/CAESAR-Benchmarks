import random

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from transformers import BertTokenizer
from transformers import DistilBertTokenizer
from transformers import AlbertTokenizer
from PIL import Image
from collections import defaultdict

from src.configs import config


class MOG_Dataset(Dataset):

    def __init__(self, 
                 hparams,
                 dataset_type='train'):

        self.hparams = hparams
        self.dataset_type = dataset_type
        self.base_dir = self.hparams.data_file_dir_base_path
        self.dataset_filename = self.hparams.dataset_filename
        # self.modality_prop = self.hparams.modality_prop
        # self.transforms_modalities = self.hparams.transforms_modalities
        self.transforms = transforms.Compose([
            # transforms.Resize((hparams.resize_image_height, hparams.resize_image_width)),
            # transforms.CenterCrop((hparams.crop_image_height, hparams.crop_image_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        
        self.load_data()
        self.tokenizer = self.get_tokenizer(self.hparams.tokenizer_name)

    def load_data(self):

        self.data = pd.read_csv(f'{self.base_dir}/{self.dataset_type}_{self.dataset_filename}.csv')

        if(config.instruction_valid_task_tag not in self.hparams.task_list):
            self.data = self.data[ self.data['is_contrastive']==0 ]

        if(config.ambiguity_recognition_task_tag not in self.hparams.task_list):
            self.data = self.data[ self.data['is_instruction_ambiguous']==0 ]

        if(self.hparams.setting_names is not None):
            self.data = self.data[self.data['setting_name'].isin(self.hparams.setting_names)]

        if(self.hparams.instruction_template is not None):
            self.data = self.data[ self.data['instruction_template'].str.contains(self.hparams.instruction_template)]

        if(self.hparams.restrict_instruction_template is not None):
            self.data = self.data[ self.data['instruction_template']!=self.hparams.restrict_instruction_template]

        print('Dataset instruction_template',self.data['instruction_template'].unique())
        if(config.instruction_valid_task_tag in self.hparams.task_list):
            print('Contrastive label stat', self.data.groupby(['is_contrastive']).count())

        self.data.reset_index(drop=True, inplace=True)

    def get_tokenizer(self, tokenizer_name="bert-base-uncased"):
        # return BertTokenizer.from_pretrained(tokenizer_name)
        # return DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        return AlbertTokenizer.from_pretrained("albert-base-v2")

    def __len__(self):
        return len(self.data)

    def get_modality_image_data(self, idx, modality):
        
        data_path = f'{self.base_dir}/{self.data.loc[idx, modality]}'
        image = Image.open(data_path).convert('RGB')
        
        # image = self.transforms_modalities[modality](image)
        if(self.hparams.print_data_id):
            print(f'before scalling image size {image.size}')
        # image_width = image.size[0] * self.hparams.image_scale
        # image_height = image.size[1] * self.hparams.image_scale
        image_resized = image.resize((int(config.image_width_normalization),int(config.image_height_normalization)))
        
        all_object_bboxes = self.get_all_object_bboxes(idx)
        images = []  
        max_im_w, max_im_h = 0, 0
        # print('#################################')
        for bbox in all_object_bboxes:
            tm_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            # print('tm_image', tm_image.size)
            # print(bbox[0], bbox[1], bbox[2], bbox[3])
            tm_image = self.transforms(tm_image)
            images.append(tm_image)
            max_im_h = max(max_im_h, tm_image.shape[1])
            max_im_w = max(max_im_w, tm_image.shape[2])

        
        # The needed padding is the difference between the
        # max width/height and the image's actual width/height.
        images = [F.pad(image, [0, max_im_w - image.size(2), 0, max_im_h - image.size(1)])
                        for image in images]

        if(self.hparams.print_data_id):
            print(f'after scalling image size {image_resized.size}')
            print(f'after scalling image tensor shape {image.shape}')

        image_context = self.transforms(image_resized)
        images = torch.stack(images, dim=0)
        # print('images shape',images.shape)
        bboxes = torch.tensor(self.get_normalized_bboxes(all_object_bboxes), dtype=torch.float)
        return image_context, images, bboxes

    def get_all_object_bboxes(self, idx):
        bboxes = []
        for view_modality in self.hparams.view_modalities:

            if(self.hparams.is_only_target_box):
                x1 = self.data.loc[idx, f'{view_modality}_object_start_point_x'] * self.hparams.image_scale
                y1 = self.data.loc[idx, f'{view_modality}_object_start_point_y'] * self.hparams.image_scale
                x2 = self.data.loc[idx, f'{view_modality}_object_end_point_x'] * self.hparams.image_scale
                y2 = self.data.loc[idx, f'{view_modality}_object_end_point_y'] * self.hparams.image_scale

                bboxes.append([float(x1), float(y1), float(x2), float(y2)])
                continue

            start_point_xs = self.data.loc[idx, f'{view_modality}_all_objects_start_point_x'].split('_')
            start_point_ys = self.data.loc[idx, f'{view_modality}_all_objects_start_point_y'].split('_')
            end_point_xs = self.data.loc[idx, f'{view_modality}_all_objects_end_point_x'].split('_')
            end_point_ys = self.data.loc[idx, f'{view_modality}_all_objects_end_point_y'].split('_')

            point_len = len(start_point_xs)
            for i in range(point_len):
                if(len(start_point_xs[i])>0 or len(start_point_ys[i])>0 
                    or len(end_point_xs[i])>0 or len(end_point_ys[i])>0):

                    bboxes.append([float(start_point_xs[i]) * self.hparams.image_scale, 
                                float(start_point_ys[i]) * self.hparams.image_scale,
                                float(end_point_xs[i]) * self.hparams.image_scale, 
                                float(end_point_ys[i]) * self.hparams.image_scale])
        return bboxes

    def get_normalized_bboxes(self, all_object_bboxes):
        normalized_bboxes = []
        for bbox in all_object_bboxes:
            normalized_bboxes.append([bbox[0]/config.image_width_normalization,
                                    bbox[1]/config.image_height_normalization,
                                    bbox[2]/config.image_width_normalization,
                                    bbox[3]/config.image_height_normalization])
        return normalized_bboxes

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if(self.hparams.print_data_id):
            print(f"data id for idx={idx} is {self.data.loc[idx, 'id']}")

        data = {}
        for view_modality in self.hparams.view_modalities:
            data[f'{view_modality}_context'], data[f'{view_modality}_bboxes'], data[f'{view_modality}_bboxes_cord'] = self.get_modality_image_data(idx, view_modality)
            # print(f'data[{view_modality}] {data[view_modality].size()}')

        task_ids = {}
        for task_name in self.hparams.task_list:
            task_ids[f'{task_name}_id'] = config.task_name_to_id[task_name]

            if(task_name == config.instruction_valid_task_tag or task_name == config.ambiguity_recognition_task_tag):
                data[f'{task_name}_labels'] = self.data.loc[idx, task_name]
            else:
                x1 = (self.data.loc[idx, f'{task_name}_object_start_point_x'] * self.hparams.image_scale) / config.image_width_normalization
                y1 = (self.data.loc[idx, f'{task_name}_object_start_point_y'] * self.hparams.image_scale) / config.image_height_normalization
                x2 = (self.data.loc[idx, f'{task_name}_object_end_point_x'] * self.hparams.image_scale) / config.image_width_normalization
                y2 = (self.data.loc[idx, f'{task_name}_object_end_point_y'] * self.hparams.image_scale) / config.image_height_normalization
            
                if(self.hparams.bbox_format == config.bbox_format_xywh):
                    w = x2 - x1
                    h = y2 - y1
                    data[f'{task_name}_target_bbox_cord'] = torch.tensor([x1, y1, w, h], dtype=torch.float)
                else:
                    data[f'{task_name}_target_bbox_cord'] = torch.tensor([x1, y1, x2, y2], dtype=torch.float)

        data['task_ids'] = task_ids
        data['scores'] = 1.0
        data['box_labels'] = 0

        #verbal input processing
        if(self.hparams.model_name=='mog_model' or ('base' in self.hparams.model_name)):
            data['verbal_instruction'] = self.tokenizer(self.data.loc[idx, 'verbal_instruction'], return_tensors="pt")
        
        return data

class MOG_Collator:

    def __init__(self, view_modalities, task_list):
        self.view_modalities = view_modalities
        self.task_list = task_list

    def gen_mask(self, seq_len, max_len):
        return torch.arange(max_len) > seq_len

    def __call__(self, batch):
        batch_size = len(batch)
        data = {}

        max_im_h = max([batch[bin][f'{view_modality}_bboxes'].shape[2] for bin in range(batch_size) for view_modality in self.view_modalities])
        max_im_w = max([batch[bin][f'{view_modality}_bboxes'].shape[3] for bin in range(batch_size) for view_modality in self.view_modalities])

        bboxes = {}
        for bin in range(batch_size):
            bboxes[bin] = {}
            for view_modality in self.view_modalities:
                bboxes[bin][f'{view_modality}_bboxes'] = F.pad(batch[bin][f'{view_modality}_bboxes'], [0, max_im_w - batch[bin][f'{view_modality}_bboxes'].shape[3], 0, max_im_h - batch[bin][f'{view_modality}_bboxes'].shape[2]])
        
        bboxes_mask = {}
        for view_modality in self.view_modalities:
            bboxes_seq_len = [batch[bin][f'{view_modality}_bboxes'].shape[0] for bin in range(batch_size)]
            max_bbox_seq_len = max(bboxes_seq_len)
            data[f'{view_modality}_bboxes'] = pad_sequence([bboxes[bin][f'{view_modality}_bboxes'] for bin in range(batch_size)], batch_first=True)
            bboxes_mask[f'{view_modality}_bboxes_seq_mask'] = torch.stack(
                                                        [self.gen_mask(seq_len, max_bbox_seq_len) 
                                                            for seq_len in bboxes_seq_len],
                                                            dim=0)
            
            data[f'{view_modality}_context'] = pad_sequence([batch[bin][f'{view_modality}_context'] for bin in range(batch_size)], batch_first=True)
            # data[view_modality] = torch.stack([batch[bin][view_modality] for bin in range(batch_size)], dim=0)
            # data[f'{view_modality}_context'] = torch.stack([batch[bin][f'{view_modality}_context'] for bin in range(batch_size)], dim=0)
            data[f'{view_modality}_bboxes_cord'] = pad_sequence([batch[bin][f'{view_modality}_bboxes_cord'] for bin in range(batch_size)], batch_first=True)
        data['bboxes_mask'] = bboxes_mask

        task_ids = {}
        for task_name in self.task_list:
            task_ids[f'{task_name}_id'] = torch.tensor([batch[bin]['task_ids'][f'{task_name}_id'] for bin in range(batch_size)],
                                    dtype=torch.long)
                                    
            if(task_name != config.instruction_valid_task_tag and task_name != config.ambiguity_recognition_task_tag):
                data[f'{task_name}_target_bbox_cord'] = torch.stack([batch[bin][f'{task_name}_target_bbox_cord'] for bin in range(batch_size)], dim=0)
                # data[f'{task_name}_all_bboxes'] = torch.stack([batch[bin][f'{task_name}_all_bboxes'] for bin in range(batch_size)], dim=0)
            else:
                data[f'{task_name}_labels'] = torch.tensor([batch[bin][f'{task_name}_labels'] for bin in range(batch_size)],
                                    dtype=torch.long)
        data['task_ids'] = task_ids
        
        for bin in range(batch_size):
            batch[bin]['verbal_instruction']['input_ids'] = batch[bin]['verbal_instruction']['input_ids'].squeeze(dim=0).contiguous()
            batch[bin]['verbal_instruction']['token_type_ids'] = batch[bin]['verbal_instruction']['token_type_ids'].squeeze(dim=0).contiguous()
            batch[bin]['verbal_instruction']['attention_mask'] = batch[bin]['verbal_instruction']['attention_mask'].squeeze(dim=0).contiguous()

        verbal_data = {}
        verbal_data['input_ids'] = pad_sequence([batch[bin]['verbal_instruction']['input_ids'] for bin in range(batch_size)], batch_first=True).long()
        verbal_data['token_type_ids'] = pad_sequence([batch[bin]['verbal_instruction']['token_type_ids'] for bin in range(batch_size)], batch_first=True).long()
        verbal_data['attention_mask'] = pad_sequence([batch[bin]['verbal_instruction']['attention_mask'] for bin in range(batch_size)], batch_first=True).long()
        data['verbal_instruction'] = verbal_data

        data[f'scores'] = torch.tensor([batch[bin][f'scores'] for bin in range(batch_size)],
                                    dtype=torch.float)
        data[f'box_labels'] = torch.tensor([batch[bin][f'box_labels'] for bin in range(batch_size)],
                                    dtype=torch.long)

        return data
