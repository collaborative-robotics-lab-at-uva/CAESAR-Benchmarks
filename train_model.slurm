#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0-16
#SBATCH --ntasks=1
#SBATCH -t 72:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH -N 1
#SBATCH --cpus-per-task=28
#SBATCH --mem=240GB
#SBATCH --output=log/slurm/base-fusion-all-clip/fa-%A-%a.log
#SBATCH -J fusions
#SBATCH --exclude=udc-ba25-[23,27,28],udc-ba26-[23,24,25,26],udc-ba27-[23,24],udc-an28-[1,7]

model_names=( 'clip' 'clip' 'clip' 'clip' 'clip' 'clip' 'clip' 'clip' 'clip' 'clip' 'clip' 'clip' 'clip' 'clip' 'clip' 'clip')
modality=( 'exo_view_image' 'exo_view_image' 'exo_view_image' 'exo_view_image' 'ego_view_image' 'ego_view_image' 'ego_view_image' 'ego_view_image' 'top_view_image' 'top_view_image' 'top_view_image' 'top_view_image' 'exo_view_image,ego_view_image,top_view_image' 'exo_view_image,ego_view_image,top_view_image' 'exo_view_image,ego_view_image,top_view_image' 'exo_view_image,ego_view_image,top_view_image')
sn=( 'both_gaze_gesture,wrong_gaze_gesture' 'both_gaze_gesture,wrong_gaze_gesture' 'both_gaze_gesture,wrong_gaze_gesture' 'both_gaze_gesture,wrong_gaze_gesture' 'both_gaze_gesture,wrong_gaze_gesture' 'both_gaze_gesture,wrong_gaze_gesture' 'both_gaze_gesture,wrong_gaze_gesture' 'both_gaze_gesture,wrong_gaze_gesture' 'both_gaze_gesture,wrong_gaze_gesture' 'both_gaze_gesture,wrong_gaze_gesture' 'both_gaze_gesture,wrong_gaze_gesture' 'both_gaze_gesture,wrong_gaze_gesture' 'both_gaze_gesture,wrong_gaze_gesture' 'both_gaze_gesture,wrong_gaze_gesture' 'both_gaze_gesture,wrong_gaze_gesture' 'both_gaze_gesture,wrong_gaze_gesture')
fusion=( 'self_attention' 'cross_attention' 'sum' 'concat' 'self_attention' 'cross_attention' 'sum' 'concat' 'self_attention' 'cross_attention' 'sum' 'concat' 'self_attention' 'cross_attention' 'sum' 'concat')

module purge
module load singularity

srun singularity exec --nv pytorch_1-11_pl_1-6-1.sif python train_model.py \
--dataset_name 'mog' \
--dataset_filename 'except_tiny_object_other_con' \
--valid_split_pct 0.2 \
--view_modalities ${modality[${SLURM_ARRAY_TASK_ID}]} \
--setting_names ${sn[${SLURM_ARRAY_TASK_ID}]} \
--task_list 'is_contrastive' \
--model_name 'base_model_vl_'${model_names[${SLURM_ARRAY_TASK_ID}]}'' \
--fusion_model_name ${fusion[${SLURM_ARRAY_TASK_ID}]} \
--exe_mode 'train' \
--val_percent_check 1.0 \
--num_sanity_val_steps 1 \
--limit_test_batches 1.0 \
--train_percent_check 1.0 \
--compute_mode 'gpu' \
--strategy 'ddp_spawn' \
--num_workers 8 \
--gpus "-1" \
-bs 8 \
-ep 4 \
-lr 0.000003 \
-cm 2 \
-cl 2 \
--image_scale 0.5 \
--view_encoder_name 'resnet50' \
--indi_modality_embedding_size 512 \
--bbox_format 'xyxy' \
--is_only_target_box \
--restrict_instruction_template 'template_null' \
-dfp '/project/CollabRoboGroup/datasets/official_data' \
-msbd 'trained_model/vl' \
-mcp 'base_model_vl' \
-logbd 'log/base/base-fusion-all-clip' \
--log_model_archi \
-logf 'rc_cd_pt_'${sn[${SLURM_ARRAY_TASK_ID}]}'_base_model_'${model_names[${SLURM_ARRAY_TASK_ID}]}'_'${fusion[${SLURM_ARRAY_TASK_ID}]}'_modality-'${modality[${SLURM_ARRAY_TASK_ID}]}'_'${SLURM_ARRAY_JOB_ID}'_'${SLURM_ARRAY_TASK_ID}'.log' \
-wdbln 'rc_cd_pt_'${sn[${SLURM_ARRAY_TASK_ID}]}'_base_model_'${model_names[${SLURM_ARRAY_TASK_ID}]}'_'${fusion[${SLURM_ARRAY_TASK_ID}]}'_modality-'${modality[${SLURM_ARRAY_TASK_ID}]}'_'${SLURM_ARRAY_JOB_ID}'_'${SLURM_ARRAY_TASK_ID}'' \
--wandb_entity 'crg' \
--wandb_project_name 'MAME-VL-Final-Fusions-All' \
--is_test \
--is_random_seed \
#--random_contrastive_data \