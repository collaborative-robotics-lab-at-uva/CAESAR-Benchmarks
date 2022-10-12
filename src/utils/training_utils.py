import torch
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
from collections import defaultdict

from src.configs import config

def get_pl_metrics(metric, bbox_format='xyxy', num_classes=2):
    if 'accuracy' in metric:
        return torchmetrics.Accuracy()
        # return Accuracy()
    elif 'f1_scores' in metric:
        return torchmetrics.F1(num_classes)
    elif 'precision' in metric:
        return torchmetrics.Precision(num_classes)
    elif 'recall_scores' in metric:
        return torchmetrics.Recall(num_classes)
    elif 'iou' in metric:
        return MeanAveragePrecision(bbox_format)


def cal_metrics(outputs, metrics, pl_metrics, task_list, stage_tag, trainer, device):
    results = {}
    map_list = ['map', 'map_50', 'map_75']
    is_map_metric = False
    for metric in metrics:
        metric_key = f'{stage_tag}_{metric}'
        if('accuracy' in metric_key):
            results[f'epoch_{metric_key}'] = pl_metrics[metric_key].compute()
        elif('iou' in metric_key):
            is_map_metric = True
            tm_results = pl_metrics[metric_key].compute()
            for map_name in map_list:
                results[f'epoch_{metric_key}_{map_name}'] = tm_results[map_name]

    task_ious = defaultdict(list)
    # print('results metric keys', results.keys())
    # print('### task list', task_list)
    for task_name in task_list:
        if(task_name != config.instruction_valid_task_tag and task_name != config.ambiguity_recognition_task_tag):
            metric_key = f'epoch_{stage_tag}_iou_{task_name}'
            for map_name in map_list:
                task_ious[map_name].append(results[f'{metric_key}_{map_name}'])
    if(is_map_metric):
        for map_name in map_list:
            results[f'epoch_{stage_tag}_iou_all_{map_name}'] = np.mean(task_ious[map_name])
    
    losses = [output['loss'] for output in outputs]
    loss = torch.mean(torch.tensor(losses, device=device))
    torch.distributed.all_reduce(loss, op = torch.distributed.ReduceOp.SUM)
    loss = loss / trainer.world_size
    results[f'epoch_{stage_tag}_loss'] = loss
    return results
