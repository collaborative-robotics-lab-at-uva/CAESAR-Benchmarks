import random

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from transformers import (AlbertTokenizer, DistilBertTokenizer, 
                        LxmertTokenizer, BertTokenizer, 
                        VisionTextDualEncoderProcessor,
                        ViTFeatureExtractor)
from transformers import CLIPProcessor
from PIL import Image
from collections import defaultdict

from src.configs import config


class MOG_Dataset_VL(Dataset):

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
        
        if(('clip' in self.hparams.model_name) or ('vl_dual_encoder' in self.hparams.model_name)):
            self.processor = self.get_processor()
        else:
            self.tokenizer = self.get_tokenizer(self.hparams.tokenizer_name)

    def load_data(self):
        if(self.dataset_filename=='none'):
            self.data = pd.read_csv(f'{self.base_dir}/{self.dataset_type}.csv')
        else:
            self.data = pd.read_csv(f'{self.base_dir}/{self.dataset_type}_{self.dataset_filename}.csv')

        # print('before Contrastive label stat', self.dataset_type, self.data.groupby(['is_contrastive']).count())

        # if(config.instruction_valid_task_tag not in self.hparams.task_list):
        #     self.data = self.data[ self.data['is_contrastive']==0 ]

        # if(config.ambiguity_recognition_task_tag not in self.hparams.task_list):
        #     self.data = self.data[ self.data['is_instruction_ambiguous']==0 ]

        if(self.hparams.setting_names is not None):
            self.data = self.data[self.data['setting_name'].isin(self.hparams.setting_names)]

        if(self.hparams.instruction_template is not None):
            self.data = self.data[ self.data['instruction_template'].str.contains(self.hparams.instruction_template)]

        if(self.hparams.restrict_instruction_template is not None):
            self.data = self.data[ self.data['instruction_template']!=self.hparams.restrict_instruction_template]

        # print('Dataset instruction_template',self.data['instruction_template'].unique())
        # if(config.instruction_valid_task_tag in self.hparams.task_list):
        #     print('Contrastive label stat', self.dataset_type, self.data.groupby(['is_contrastive']).count())

        self.data.reset_index(drop=True, inplace=True)
        self.data_len = len(self.data)

    def get_tokenizer(self, tokenizer_name="bert-base-uncased"):
        if('lxmert' in self.hparams.model_name):
            return LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        elif(('visual_bert' in self.hparams.model_name) or ('late_fusion' in self.hparams.model_name)):
            return BertTokenizer.from_pretrained("bert-base-uncased")
        return BertTokenizer.from_pretrained("bert-base-uncased")

    def get_processor(self, processor_name="openai/clip-vit-base-patch32"):
        if(('clip' in self.hparams.model_name)):
            return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        elif(('vl_dual_encoder' in self.hparams.model_name)):
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
            processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)
            return processor

    def __len__(self):
        return len(self.data)

    def get_modality_image_data(self, idx, modality):
        
        data_path = f'{self.base_dir}/{self.data.loc[idx, modality]}'
        image = Image.open(data_path).convert('RGB')
        image_resized = image.resize((int(config.image_width_normalization),int(config.image_height_normalization)))
        image_context = self.transforms(image_resized)

        all_object_bboxes = self.get_all_object_bboxes(idx, modality)
        images = []  
        max_im_w, max_im_h = 0, 0
        
        bbox_image_masks = []
        for bbox in all_object_bboxes:
            tm_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            # print(modality, 'tm_image', tm_image.size)
            # print(bbox[0], bbox[1], bbox[2], bbox[3])
            tm_image = self.transforms(tm_image)
            images.append(tm_image)
            max_im_h = max(max_im_h, tm_image.shape[1])
            max_im_w = max(max_im_w, tm_image.shape[2])

            bbox_image_masks.append(self.get_bbox_image_mask(bbox))
        
        bbox_image_masks = torch.stack(bbox_image_masks, dim=0).contiguous()

        
        # The needed padding is the difference between the
        # max width/height and the image's actual width/height.
        images = [F.pad(image, [0, max_im_w - image.size(2), 0, max_im_h - image.size(1)])
                        for image in images]

        if(self.hparams.print_data_id):
            print(f'after scalling image size {image_resized.size}')
            print(f'after scalling image tensor shape {image.shape}')

        images = torch.stack(images, dim=0)
        bboxes = torch.tensor(self.get_normalized_bboxes(all_object_bboxes), dtype=torch.float)
        return image_context, image_resized, images, bboxes, bbox_image_masks
        # else:
        #     return image_context, image_resized

    def get_all_object_bboxes(self, idx, modality):
        bboxes = []
        modality = self.convert_modality_name(modality) # convert skeletal to image modality to get the bbox as image and skeletal has the same bbox
        for view_modality in self.hparams.view_modalities:

            if(view_modality!=modality):
                continue

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

    def convert_modality_name(self, modality):
        if('skeletal' in modality):
            modality = modality.split('_')
            modality = f'{modality[0]}_{modality[1]}_image'
            return modality
        return modality

    def get_normalized_bboxes(self, all_object_bboxes):
        normalized_bboxes = []
        for bbox in all_object_bboxes:
            normalized_bboxes.append([bbox[0]/config.image_width_normalization,
                                    bbox[1]/config.image_height_normalization,
                                    bbox[2]/config.image_width_normalization,
                                    bbox[3]/config.image_height_normalization])
        return normalized_bboxes

    def get_bbox_image_mask(self, bbox):
        mask = np.zeros((int(config.image_width_normalization),int(config.image_height_normalization)),dtype=np.uint8)
        mask[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])] = 255
        mask = torch.tensor(mask, dtype=torch.float)
        mask = mask.unsqueeze(dim=0).contiguous()
        return mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if(int(self.data.loc[idx, 'is_contrastive'])==0):
            verbal_des = self.data.loc[idx, 'verbal_instruction']
        else:
            if(self.hparams.random_contrastive_data):
                
                while(True):
                    con_data_index = random.randint(0, self.data_len-1)
                    verbal_des_pos = self.data.loc[idx, 'verbal_instruction']
                    verbal_des = self.data.loc[con_data_index, 'verbal_instruction']

                    if(verbal_des!=verbal_des_pos and con_data_index!=idx):
                        break
            else:
                con_data_index = idx
            verbal_des = self.data.loc[con_data_index, 'verbal_instruction']

        data = {}
        for view_modality in self.hparams.view_modalities:
            data[f'{view_modality}_context'], raw_image, data[f'{view_modality}_bboxes'], data[f'{view_modality}_bboxes_cord'], data[f'{view_modality}_bboxes_image_mask'] = self.get_modality_image_data(idx, view_modality)
            # else:
            #     data[f'{view_modality}_context'], raw_image = self.get_modality_image_data(idx, view_modality)
            if(('clip' in self.hparams.model_name) or ('vl_dual_encoder' in self.hparams.model_name)):
                processed_input = self.processor(text=[verbal_des], 
                                    images=raw_image, 
                                    return_tensors="pt", padding=True)
                data[f'{view_modality}_processed_input'] = processed_input
        
        if(not (('clip' in self.hparams.model_name) or ('vl_dual_encoder' in self.hparams.model_name))):
            data['verbal_instruction'] = self.tokenizer(verbal_des, return_tensors="pt")

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

        return data

class MOG_Collator_VL:

    def __init__(self, model_name, view_modalities, 
                task_list, is_bbox_embed, is_bbox_cord_embed, 
                is_bbox_image_mask_encode, combine_view_context_bbox):
        self.model_name = model_name
        self.view_modalities = view_modalities
        self.task_list = task_list
        self.is_bbox_embed = is_bbox_embed
        self.is_bbox_cord_embed = is_bbox_cord_embed
        self.is_bbox_image_mask_encode = is_bbox_image_mask_encode
        self.combine_view_context_bbox = combine_view_context_bbox

    def gen_mask(self, seq_len, max_len):
        if(self.combine_view_context_bbox):
            seq_len += 1
            max_len += 1 
        return torch.arange(max_len) > seq_len

    def __call__(self, batch):
        batch_size = len(batch)
        data = {}
        
        if(self.is_bbox_embed or self.is_bbox_cord_embed or self.is_bbox_image_mask_encode  or ('drnet' in self.model_name)):
            max_im_h = max([batch[bin][f'{view_modality}_bboxes'].shape[2] for bin in range(batch_size) for view_modality in self.view_modalities])
            max_im_w = max([batch[bin][f'{view_modality}_bboxes'].shape[3] for bin in range(batch_size) for view_modality in self.view_modalities])

            bboxes = {}
            for bin in range(batch_size):
                bboxes[bin] = {}
                for view_modality in self.view_modalities:
                    bboxes[bin][f'{view_modality}_bboxes'] = F.pad(batch[bin][f'{view_modality}_bboxes'], [0, max_im_w - batch[bin][f'{view_modality}_bboxes'].shape[3], 0, max_im_h - batch[bin][f'{view_modality}_bboxes'].shape[2]])
            
            bboxes_mask = {}
            for view_modality in self.view_modalities:
                data[f'{view_modality}_context'] = pad_sequence([batch[bin][f'{view_modality}_context'] for bin in range(batch_size)], batch_first=True)
                
                bboxes_seq_len = [batch[bin][f'{view_modality}_bboxes'].shape[0] for bin in range(batch_size)]
                max_bbox_seq_len = max(bboxes_seq_len)
                data[f'{view_modality}_bboxes'] = pad_sequence([bboxes[bin][f'{view_modality}_bboxes'] for bin in range(batch_size)], batch_first=True)
                bboxes_mask[f'{view_modality}_bboxes_seq_mask'] = torch.stack(
                                                            [self.gen_mask(seq_len, max_bbox_seq_len) 
                                                                for seq_len in bboxes_seq_len],
                                                                dim=0)

                bboxes_seq_len = [batch[bin][f'{view_modality}_bboxes_image_mask'].shape[0] for bin in range(batch_size)]
                max_bbox_seq_len = max(bboxes_seq_len)
                data[f'{view_modality}_bboxes_image_mask'] = pad_sequence([batch[bin][f'{view_modality}_bboxes_image_mask'] for bin in range(batch_size)], batch_first=True)
                bboxes_mask[f'{view_modality}_bboxes_image_mask_seq_mask'] = torch.stack(
                                                            [self.gen_mask(seq_len, max_bbox_seq_len) 
                                                                for seq_len in bboxes_seq_len],
                                                                dim=0)


                # data[view_modality] = torch.stack([batch[bin][view_modality] for bin in range(batch_size)], dim=0)
                # data[f'{view_modality}_context'] = torch.stack([batch[bin][f'{view_modality}_context'] for bin in range(batch_size)], dim=0)
                data[f'{view_modality}_bboxes_cord'] = pad_sequence([batch[bin][f'{view_modality}_bboxes_cord'] for bin in range(batch_size)], batch_first=True)
        else:
            bboxes_mask = {}
            for view_modality in self.view_modalities:
                data[f'{view_modality}_context'] = pad_sequence([batch[bin][f'{view_modality}_context'] for bin in range(batch_size)], batch_first=True)
                bboxes_seq_len = [1 for bin in range(batch_size)]
                max_bbox_seq_len = max(bboxes_seq_len)
                bboxes_mask[f'{view_modality}_bboxes_seq_mask'] = torch.stack(
                                                            [self.gen_mask(seq_len, max_bbox_seq_len) 
                                                                for seq_len in bboxes_seq_len],
                                                                dim=0)
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

        # Process input for clip and vl-dual-encoder
        if(('clip' in self.model_name) or ('vl_dual_encoder' in self.model_name)):
            for view_modality in self.view_modalities:
                input_key = f'{view_modality}_processed_input'
                for bin in range(batch_size):
                    batch[bin][input_key]['input_ids'] = batch[bin][input_key]['input_ids'].squeeze(dim=0).contiguous()
                    batch[bin][input_key]['attention_mask'] = batch[bin][input_key]['attention_mask'].squeeze(dim=0).contiguous()
                    batch[bin][input_key]['pixel_values'] = batch[bin][input_key]['pixel_values'].squeeze(dim=0).contiguous()

                processed_data = {}
                processed_data['input_ids'] = pad_sequence([batch[bin][input_key]['input_ids'] for bin in range(batch_size)], batch_first=True).long()
                processed_data['attention_mask'] = pad_sequence([batch[bin][input_key]['attention_mask'] for bin in range(batch_size)], batch_first=True).long()
                processed_data['pixel_values'] = pad_sequence([batch[bin][input_key]['pixel_values'] for bin in range(batch_size)], batch_first=True)
                data[input_key] = processed_data
        else:
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