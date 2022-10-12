import os
import numpy as np
import torch 

class ModelCheckpointing:
    def __init__(self,
                ckpt_base_dir,
                ckpt_filename,
                metrics,
                metrics_save_ckpt_mode,
                metrics_mode_dict,
                logger=None):
        self.ckpt_base_dir = ckpt_base_dir
        self.ckpt_filename = ckpt_filename
        self.metrics = metrics
        self.metrics_mode_dict = metrics_mode_dict 
        self.metrics_save_ckpt_mode = metrics_save_ckpt_mode
        self.logger = logger

        self.best_metrics_score = {}
        for metric in self.metrics:
            tm_init_value = np.Inf if self.metrics_mode_dict[metric]=='min' else 0
            self.best_metrics_score[metric] = tm_init_value
    
    def update_metric_save_ckpt(self, results, epoch, trainer=None):

        # save checkpoints and logging
        for metric in self.metrics:
            if('iou' in metric):
                for map_name in ['map', 'map_50', 'map_75']:
                    self._update_metric_save_ckpt(f'{metric}_{map_name}', results, epoch, trainer)
            else:
                self._update_metric_save_ckpt(metric, results, epoch, trainer)
        
        # update the best metric scores
        for metric in self.metrics:
            if('iou' in metric):
                for map_name in ['map', 'map_50', 'map_75']:
                    self._update_best_metric_score(f'{metric}_{map_name}', results)
            else:
                self._update_best_metric_score(metric, results)

    def _update_metric_save_ckpt(self, metric, results, epoch, trainer=None):
        if (metric not in results) or (metric not in self.metrics_save_ckpt_mode):
            return

        if self._is_best_score(metric, results[metric]):
            if self.metrics_save_ckpt_mode[metric]:
                ckpt_filepath = f'{self.ckpt_base_dir}/best_{metric}_{self.ckpt_filename}'
                self._save_model(ckpt_filepath, trainer)
                self.logger.log(f'Model save to {ckpt_filepath}')
            
            if self.logger is not None:
                log_txt = f'\n###>>> Epoch {epoch}:- {metric} updated:'
                comma = ''
                for tm_metric in self.metrics:
                    if tm_metric in results:
                        tm_txt = '({:.5f} --> {:.5f})'.format(self.best_metrics_score[tm_metric], results[tm_metric])
                        log_txt = f'{log_txt}{comma} {tm_metric} {tm_txt}'
                        comma = ','
                log_txt = f'{log_txt}\n'
                self.logger.log(log_txt)

    def _update_best_metric_score(self, metric, results): 
        if metric not in results:
            return
        if self._is_best_score(metric, results[metric]):
            self.best_metrics_score[metric]=results[metric]
    
    def _save_model(self, ckpt_filepath, trainer):
        os.makedirs(os.path.dirname(ckpt_filepath), exist_ok=True)
        trainer.save_checkpoint(ckpt_filepath)
        
    def _is_best_score(self, metric, result):
        if self.metrics_mode_dict[metric]=='min':
            if self.best_metrics_score[metric]>result:
                return True
            else:
                return False
        else:
            if self.best_metrics_score[metric]<result:
                return True
            else:
                return False


    