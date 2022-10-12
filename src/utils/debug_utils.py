
from torch.utils.data import DataLoader
from src.configs import config
from src.datasets.mog_dataset import *
from src.datasets.mog_dataset_vl import *

def debug_dataloader(hparams, dataset_type='test', last_batch=-1):

    Dataset = MOG_Dataset_VL
    collate_fn = MOG_Collator_VL(hparams.model_name,
                                hparams.view_modalities,
                                hparams.task_list,
                                hparams.is_bbox_embed,
                                hparams.combine_view_context_bbox)

    dataset = Dataset(hparams, dataset_type)
    
    dataloader = DataLoader(dataset,
                                  batch_size=hparams.batch_size,
                                  shuffle=True, 
                                  collate_fn=collate_fn,
                                  num_workers=hparams.num_workers)
    print(len(dataloader))
    for batch_id, batch in enumerate(dataloader, 0):
        print('batch_id', batch_id)
        print(batch.keys())

        # if('clip' not in hparams.model_name):
        for view_modality in hparams.view_modalities:
            print(f'{view_modality}_context shape', batch[f'{view_modality}_context'].size())
            print(f'{view_modality}_bboxes shape', batch[f'{view_modality}_bboxes'].size())
            print(f'{view_modality}_bboxes_cord shape', batch[f'{view_modality}_bboxes_cord'].size())
            print(f'{view_modality}_bboxes_seq_mask size:', batch['bboxes_mask'][f'{view_modality}_bboxes_seq_mask'].size())
            print(f'{view_modality}_bboxes_image_mask size:', batch[f'{view_modality}_bboxes_image_mask'].size())
            print(f'{view_modality}_bboxes_image_mask_seq_mask size:', batch['bboxes_mask'][f'{view_modality}_bboxes_image_mask_seq_mask'].size())
            # print(f'{view_modality}_bboxes_seq_mask:', batch['bboxes_mask'][f'{view_modality}_bboxes_seq_mask'])
        
        for task_name in hparams.task_list:
            print(f'task_ids size:', batch['task_ids'][f'{task_name}_id'].size())
            print(f'task_ids:', batch['task_ids'][f'{task_name}_id'])

            if(task_name != config.instruction_valid_task_tag and task_name != config.ambiguity_recognition_task_tag):
                print(f'{task_name}_target_bbox_cord size:', batch[f'{task_name}_target_bbox_cord'].size())
            else:
                print(f'{task_name} size:', batch[f'{task_name}_labels'].size())
                print(f'{task_name}:', batch[f'{task_name}_labels'])

            # print(batch['verbal_instruction'])

        if (batch_id > last_batch) and last_batch!=-1:
            break