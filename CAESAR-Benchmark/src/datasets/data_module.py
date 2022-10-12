import pytorch_lightning as pl
from src.datasets.mog_dataset import *
from src.datasets.mog_dataset_clip import *
from src.datasets.mog_dataset_vl import *
from src.utils.log import *
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import math

class MOGDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(vars(hparams))

        # if(self.hparams.model_name=='base_model_clip'):
        #     self.Dataset = MOG_Dataset_CLIP
        #     self.collate_fn = MOG_Collator_CLIP(self.hparams.view_modalities,
        #                                 self.hparams.task_list)
        if(('_vl_' in self.hparams.model_name) or ('_unimodal_' in self.hparams.model_name)):
            self.Dataset = MOG_Dataset_VL
            self.collate_fn = MOG_Collator_VL(self.hparams.model_name,
                                        self.hparams.view_modalities,
                                        self.hparams.task_list,
                                        self.hparams.is_bbox_embed,
                                        self.hparams.is_bbox_cord_embed,
                                        self.hparams.is_bbox_image_mask_encode,
                                        self.hparams.combine_view_context_bbox)
        elif self.hparams.dataset_name=='mog':
            self.Dataset = MOG_Dataset
            self.collate_fn = MOG_Collator(self.hparams.view_modalities,
                                        self.hparams.task_list)
        
        self.txt_logger = TextLogger(self.hparams.log_base_dir, 
                                    self.hparams.log_filename,
                                    print_console=True)


    # def prepare_data(self):
    def setup(self, stage=None):

        if self.hparams.share_train_dataset:
            self.full_dataset = self.Dataset(hparams=self.hparams,
                                            dataset_type='train')

            dataset_len = len(self.full_dataset)
            self.hparams.dataset_len = dataset_len

            valid_len = math.floor(dataset_len*self.hparams.valid_split_pct)
            train_len = dataset_len - valid_len

            self.train_dataset, self.valid_dataset = random_split(self.full_dataset,
                                                                [train_len, valid_len])

            self.test_dataset = self.Dataset(hparams=self.hparams,
                                            dataset_type='test')
        
        else:
            self.train_dataset = self.Dataset(hparams=self.hparams,
                                        dataset_type='train')

            self.valid_dataset = self.Dataset(hparams=self.hparams,
                                            dataset_type='valid')

            self.test_dataset = self.Dataset(hparams=self.hparams,
                                            dataset_type='test')
                                            
        self.txt_logger.log(f'train dataset len: {len(self.train_dataset)}\n')
        self.txt_logger.log(f'valid dataset len: {len(self.valid_dataset)}\n')
        self.txt_logger.log(f'test dataset len: {len(self.test_dataset)}\n')

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
                            batch_size=self.hparams.batch_size,
                            collate_fn=self.collate_fn,
                            num_workers=self.hparams.num_workers,
                            shuffle=True,
                            drop_last=True,
                            persistent_workers=True)
        return loader

    def val_dataloader(self):
        if self.hparams.no_validation:
            return None
            
        loader = DataLoader(self.valid_dataset,
                            batch_size=self.hparams.batch_size,
                            collate_fn=self.collate_fn,
                            num_workers=self.hparams.num_workers,
                            shuffle=False,
                            drop_last=True,
                            persistent_workers=True)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset,
                            batch_size=self.hparams.batch_size,
                            collate_fn=self.collate_fn,
                            num_workers=self.hparams.num_workers,
                            shuffle=False,
                            drop_last=True,
                            persistent_workers=True)
        return loader