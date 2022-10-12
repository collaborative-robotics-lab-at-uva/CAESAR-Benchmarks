# !/usr/bin/env python
# coding: utf-8
from argparse import ArgumentParser
from datetime import datetime
import torch
import wandb
from pytorch_lightning import loggers, Trainer, seed_everything
from sklearn.model_selection import LeaveOneOut

from src.models.mog_model_trainer import *
from src.models.base_model_trainer import *
from src.datasets.data_module import *
from src.utils.model_saving import *
from src.utils.debug_utils import *
from src.utils.log import TextLogger
from src.configs import config

from collections import defaultdict
import numpy as np
import torch
import json
import random

test_metrics = {}

def main(args):

    if(args.is_random_seed):
        seed_everything(random.randint(0,1000000))
    else:
        seed_everything(33)

    txt_logger = TextLogger(args.log_base_dir, 
                            args.log_filename,
                            print_console=True)

    if args.model_checkpoint_filename is None:
        args.model_checkpoint_filename = f'{args.model_checkpoint_prefix}_{datetime.utcnow().timestamp()}.pth'
    
    args.setting_names = args.setting_names.strip().split(',')
    args.test_models = args.test_models.strip().split(',')
    args.test_metrics = args.test_metrics.strip().split(',')

    for test_model in args.test_models:
        test_metrics[f'{test_model}'] = defaultdict(list)
    
    txt_logger.log(f'model_checkpoint_prefix:{args.model_checkpoint_prefix}\n')
    txt_logger.log(f'model_checkpoint_filename:{args.model_checkpoint_filename}, resume_checkpoint_filename:{args.resume_checkpoint_filename}\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    txt_logger.log(f'pytorch version: {torch.__version__}\n')
    txt_logger.log(f'GPU Availability: {device}, gpus: {args.gpus}\n')

    # Set dataloader class and prop 
    args.view_modalities = args.view_modalities.strip().split(',')
    args.task_list = args.task_list.strip().split(',')

    if args.exe_mode=='debug_dl':
        debug_dataloader(args, last_batch=3)
        return

    if args.dataset_name=='mog':

        if args.resume_checkpoint_filename is not None:
            args.resume_checkpoint_filepath = f'{args.model_save_base_dir}/{args.resume_checkpoint_filename}'
            if os.path.exists(args.resume_checkpoint_filepath):
                args.resume_training = True
            else:
                txt_logger.log(f'Checkpoint is not exists: {args.resume_checkpoint_filename}\n')
                args.resume_training = False

        loggers_list = []
        if (args.tb_writer_name is not None) and (args.exe_mode=='train'):
            loggers_list.append(loggers.TensorBoardLogger(save_dir=args.log_base_dir, 
                                name=f'{args.tb_writer_name}'))
        if (args.wandb_log_name is not None) and (args.exe_mode=='train'):
            loggers_list.append(loggers.WandbLogger(save_dir=args.log_base_dir, 
                            name=f'{args.wandb_log_name}',
                            entity=f'{args.wandb_entity}',
                            project=f'{args.wandb_project_name}'))

        txt_logger.log(str(args), print_console=args.log_model_archi)

        start_training(args, txt_logger, loggers_list)        

def start_training(args, txt_logger, loggers=None):

    txt_logger.log(f"\n\n$$$$$$$$$ Start training $$$$$$$$$\n\n")
    dataModule = MOGDataModule(args)
    if(args.model_name=='mog_model'):
        ModelTrainer = MOG_Model_Trainer
    elif('base' in args.model_name):
        ModelTrainer = Base_Model_Trainer
    model = ModelTrainer(hparams=args)
    if args.log_model_archi:
        txt_logger.log(str(model))

    # if(loggers is not None and len(loggers)>0):
    #     loggers[0].watch(model, log_graph=False)

    if args.resume_training:
        # model, _ = load_model(model, args.resume_checkpoint_filepath, strict_load=False)
        model = ModelTrainer.load_from_checkpoint(args.resume_checkpoint_filepath, hparams=args)
        txt_logger.log(f'Reload model from chekpoint: {args.resume_checkpoint_filename}\n model_checkpoint_filename: {args.model_checkpoint_filename}\n')
        
    
    trainer = Trainer.from_argparse_args(args,gpus=args.gpus, 
                accelerator=args.compute_mode,
                strategy=args.strategy,
                max_epochs=args.epochs,
                logger=loggers,
                enable_checkpointing=False,
                precision=args.float_precision,
                limit_train_batches=args.train_percent_check,
                num_sanity_val_steps=args.num_sanity_val_steps,
                limit_val_batches=args.val_percent_check,
                fast_dev_run=args.fast_dev_run,
                detect_anomaly=False)

    if args.only_testing:
        model.eval()
        trainer.test(model, datamodule=dataModule)
    else:
        if args.lr_find:
            print('Start Learning rate finder')
            lr_trainer = Trainer()
            # lr_trainer = Trainer()
            lr_finder = lr_trainer.tuner.lr_find(model, datamodule=dataModule)
            fig = lr_finder.plot(suggest=True)
            fig.show()
            new_lr = lr_finder.suggestion()
            txt_logger.log(str(new_lr))
            model.hparams.learning_rate = new_lr

        trainer.fit(model, datamodule=dataModule)
        if args.is_test:
            txt_logger.log(f"\n\n$$$$$$$$$ Start testing $$$$$$$$$\n\n")
            for test_model in args.test_models:
                trainer = None
                model = None 
                trainer = Trainer.from_argparse_args(args,gpus=args.gpus, 
                            strategy=None,
                            max_epochs=args.epochs,
                            logger=None,
                            enable_checkpointing=False,
                            precision=args.float_precision,
                            limit_test_batches=args.limit_test_batches)

                model = ModelTrainer(hparams=args)
                ckpt_filename = f'best_epoch_{test_model}_{args.model_checkpoint_filename}'
                ckpt_filepath = f'{args.model_save_base_dir}/{ckpt_filename}'
                if not os.path.exists(ckpt_filepath):
                    txt_logger.log(f'Skip testing model for chekpoint({ckpt_filepath}) is not found\n')
                    continue 
                #model, _ = load_model(model, ckpt_filepath, strict_load=False)
                model = ModelTrainer.load_from_checkpoint(ckpt_filepath, hparams=args)
                model.eval()
                txt_logger.log(f'Reload testing model from chekpoint: {ckpt_filepath}\n')
                txt_logger.log(f'{test_model}')
                trainer.test(model, datamodule=dataModule)
                
                # test_log = model.test_log 
                # for test_metric in args.test_metrics:
                #     test_metrics[f'{test_model}'][f'test_{test_metric}'].append(test_log[f'test_{test_metric}'])
                
                trainer = None
                model = None
                torch.cuda.empty_cache()

    trainer = None
    model = None 
    torch.cuda.empty_cache()

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser()

    parser.add_argument("-compute_mode", "--compute_mode", help="compute_mode",
                        default='gpu')
    parser.add_argument("--fast_dev_run", help="fast_dev_run",
                        action="store_true", default=False)    
    parser.add_argument("-num_nodes", "--num_nodes", help="num_nodes",
                        type=int, default=1)
    parser.add_argument("--strategy", help="strategy",
                        default='ddp_spawn')
    parser.add_argument("--gpus", help="number of gpus or gpus list",
                        default="-1")
    parser.add_argument("--float_precision", help="float precision",
                        type=int, default=32)
    parser.add_argument("--dataset_name", help="dataset_name",
                        default=None)
    parser.add_argument("--dataset_filename", help="dataset_name",
                        default='none')
    parser.add_argument("-ws", "--window_size", help="windows size",
                        type=int, default=5)
    parser.add_argument("-wst", "--window_stride", help="windows stride",
                        type=int, default=5)
    parser.add_argument("-bs", "--batch_size", help="batch size",
                        type=int, default=2)
    parser.add_argument("-nw", "--num_workers", help="num_workers",
                        type=int, default=2)
    parser.add_argument("-ep", "--epochs", help="epoch per validation cycle",
                        type=int, default=200)
    parser.add_argument("-lr", "--learning_rate", help="learning rate",
                        type=float, default=3e-4)
    parser.add_argument("-sml", "--seq_max_len", help="maximum sequence length",
                        type=int, default=200)
    parser.add_argument("-rt", "--resume_training", help="resume training",
                        action="store_true", default=False)
    parser.add_argument("-sl", "--strict_load", help="partially or strictly load the saved model",
                        action="store_true", default=False)
    parser.add_argument("-dfp", "--data_file_dir_base_path", help="data_file_dir_base_path",
                        default=None)

    parser.add_argument("-enl", "--encoder_num_layers", help="LSTM encoder layer",
                        type=int, default=2)
    parser.add_argument("-lstm_bi", "--lstm_bidirectional", help="LSTM bidirectional [True/False]",
                        action="store_true", default=False)

    parser.add_argument("-mcp", "--model_checkpoint_prefix", help="model checkpoint filename prefix",
                        default='uva_dar')
    parser.add_argument("-mcf", "--model_checkpoint_filename", help="model checkpoint filename",
                        default=None)
    parser.add_argument("-rcf", "--resume_checkpoint_filename", help="resume checkpoint filename",
                        default=None)

    parser.add_argument("-logf", "--log_filename", help="execution log filename",
                        default='exe_uva_dar.log')
    parser.add_argument("-logbd", "--log_base_dir", help="execution log base dir",
                        default='log/uva_dar')
    parser.add_argument("-tb_wn", "--tb_writer_name", help="tensorboard writer name",
                        default=None)
    parser.add_argument("-wdbln", "--wandb_log_name", help="wandb_log_name",
                        default=None)
    parser.add_argument("--wandb_entity", help="wandb_entity",
                        default='crg')
    parser.add_argument("--wandb_project_name", help="wandb_project_name",
                        default='MAME')
    parser.add_argument("--log_model_archi", help="log model",
                        action="store_true", default=False)
    parser.add_argument("--print_data_id", help="print_data_id",
                        action="store_true", default=False)

    parser.add_argument("-ipf", "--is_pretrained_fe", help="is_pretrained_fe",
                        action="store_true", default=False)
    parser.add_argument("-edbp", "--embed_dir_base_path", help="embed_dir_base_path",
                        default=None)
    parser.add_argument("--pt_vis_encoder_archi_type", help="pt_vis_encoder_archi_type",
                        default='resnet18')
                
    parser.add_argument("-msbd", "--model_save_base_dir", help="model_save_base_dir",
                        default="trained_model")
    parser.add_argument("-exe_mode", "--exe_mode", help="exe_mode[dl_test/train]",
                        default='train')
    parser.add_argument("--train_percent_check", help="train_percent_check",
                        type=float, default=1.0)
    parser.add_argument("--num_sanity_val_steps", help="num_sanity_val_steps",
                        type=int, default=5)
    parser.add_argument("--val_percent_check", help="val_percent_check",
                        type=float, default=1.0)
    parser.add_argument("--limit_test_batches", help="limit_test_batches",
                        type=float, default=1.0)
    parser.add_argument("--no_validation", help="no_validation",
                        action="store_true", default=False)
    parser.add_argument("--slurm_job_id", help="slurm_job_id",
                        default=None)
    
    # Model
    parser.add_argument("--model_name", help="model_name",
                        default=None)
    parser.add_argument("--is_decoders", help="is_decoders",
                        action="store_true", default=False)
    parser.add_argument("--is_bbox_embed", help="is_bbox_embed",
                        action="store_true", default=False)
    parser.add_argument("--is_bbox_cord_embed", help="is_bbox_cord_embed",
                        action="store_true", default=False)
    parser.add_argument("--is_bbox_image_mask_encode", help="is_bbox_image_mask_encode",
                        action="store_true", default=False)
    parser.add_argument("--combine_view_context_bbox", help="combine_view_context_bbox",
                        action="store_true", default=False)
    parser.add_argument("--is_only_target_box", help="is_only_target_box",
                        action="store_true", default=False)

    # Dataset Config
    parser.add_argument("--setting_names", help="setting_names",
                        default=None)
    parser.add_argument("--view_modalities", help="view_modalities",
                        default=None)
    parser.add_argument("--random_contrastive_data", help="random_contrastive_data",
                        action="store_true", default=False)
    parser.add_argument("--indi_modality_embedding_size", help="indi_modality_embedding_size",
                        type=int, default=None)

    # Verbal Instruction Encoder Prop
    parser.add_argument("--tokenizer_name", help="tokenizer_name",
                        default='bert-base-uncased')

    # View Encoders Prop
    parser.add_argument("--view_encoder_name", help="tokenizer_name",
                        default='resnet34')
    
    # Multimodal Fusion Prop
    parser.add_argument("--fusion_model_name", help="fusion_model_name",
                        default='concat')
    parser.add_argument("--fusion_model_nhead", help="fusion_model_nhead",
                        type=int, default=4)
    parser.add_argument("--fusion_model_dropout", help="fusion_model_dropout",
                        type=float, default=0.1)

    # Guided Projection Prop
    parser.add_argument("--guided_projection_nhead", help="guided_projection_nhead",
                        type=int, default=1)
    parser.add_argument("--guided_projection_dropout", help="guided_projection_dropout",
                        type=float, default=0.1)
    
    # Guided Fusion Prop
    parser.add_argument("--guided_fusion_nhead", help="guided_fusion_nhead",
                        type=int, default=1)
    parser.add_argument("--guided_fusion_dropout", help="guided_fusion_dropout",
                        type=float, default=0.1)

    # Multi-task Config
    parser.add_argument("--task_list", help="task_list",
                        default=None)
    parser.add_argument("--bbox_format", help="bbox_format",
                        default='xyxy')
    parser.add_argument("--multitask_modal_nhead", help="multitask_modal_nhead",
                        type=int, default=1)
    parser.add_argument("--multitask_modal_dropout", help="multitask_modal_dropout",
                        type=float, default=0.1)
    parser.add_argument("--instruction_template", help="instruction_template",
                        default=None)
    parser.add_argument("--restrict_instruction_template", help="instruction_template",
                        default=None)
    # possible value for instruction_template
    # ['template_null', 'template_1_1', 'ego_template_2_2', 'ego_template_3_1',
    # 'template_1_2', 'exo_template_2_1', 'exo_template_3_2',
    # 'ego_template_3_2', 'exo_template_2_2',
    # 'ego_template_2_1', 'exo_template_3_1']

    # Data preprocessing
    parser.add_argument("--data_split_type", help="data_split_type",
                        default=None)
    parser.add_argument("--valid_split_pct", help="valid_split_pct",
                        type=float, default=0.15)
    parser.add_argument("--test_split_pct", help="test_split_pct",
                        type=float, default=0.2)
    parser.add_argument("--share_train_dataset", help="share_train_dataset",
                        action="store_true", default=False)
    parser.add_argument("--skip_frame_len", help="skip_frame_len",
                        type=int, default=1)
    parser.add_argument("-rimg_w", "--resize_image_width", help="resize to image width",
                        type=int, default=config.image_width)
    parser.add_argument("-rimg_h", "--resize_image_height", help="resize to image height",
                        type=int, default=config.image_height)
    parser.add_argument("-cimg_w", "--crop_image_width", help="crop to image width",
                        type=int, default=config.image_width)
    parser.add_argument("-cimg_h", "--crop_image_height", help="crop to image height",
                        type=int, default=config.image_height)
    parser.add_argument("--image_scale", help="image scale ration [0.1-1.0]",
                        type=float, default=0.35)

    # Optimization
    parser.add_argument("--lr_find", help="learning rate finder",
                        action="store_true", default=False)
    parser.add_argument("--lr_scheduler", help="lr_scheduler",
                        default=None)
    parser.add_argument("-cl", "--cycle_length", help="total number of executed iteration",
                        type=int, default=100)
    parser.add_argument("-cm", "--cycle_mul", help="total number of executed iteration",
                        type=int, default=2)
    parser.add_argument("--is_random_seed", help="is_random_seed",
                        action="store_true", default=False)

    # Losses
    parser.add_argument("--loss_align_weight", help="loss_align_weight",
                        type=float, default=0.3)
    parser.add_argument("--loss_diff_weight", help="loss_diff_weight",
                        type=float, default=0.3)
    parser.add_argument("--loss_decoder_weight", help="loss_decoder_weight",
                        type=float, default=0.3)
    parser.add_argument("--loss_multitask_weight", help="loss_multitask_weight",
                        type=float, default=1.0)
    parser.add_argument("--bbox_loss_type", help="bbox_loss_type",
                        default=None)

    # Testing Config
    parser.add_argument("--test_models", help="test_models",
                        default='valid_loss,valid_accuracy,train_loss')
    parser.add_argument("--test_metrics", help="test_metrics",
                        default='loss,accuracy,f1_scores,precision,recall_scores')
    parser.add_argument("--is_test", help="evaluate on test dataset",
                        action="store_true", default=False)
    parser.add_argument("--only_testing", help="Perform only test on the pretrained model",
                        action="store_true", default=False)

    args = parser.parse_args()
    main(args=args)
