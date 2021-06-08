import os
import json
import enum
import numpy as np


__all__ = ['Configs', 'ProjectPath']


class Configs:
    def __init__(self):
        self.init = json.load(open(ProjectPath.INITIAL_SETTINGS_PATH.value))
        self.project_name = self.init['project_name']
        self.classes = self.init['classes']
        self.num_classes = len(self.classes)

        self.model_name = self.init['model']  # ['yolo_v3', 'yolo_v3_tiny']
        self.num_scales = 3 if not self.model_name.endswith('_tiny') else 2
        self.input_size = 416
        self.input_channels = 3
        self.strides = self._get_strides()
        self.anchors = self._get_anchors()
        self.grid_sizes = self.input_size // self.strides
        self.anchors_per_scale = 3
        self.max_num_bboxes_per_scale = 100
        self.iou_loss_thr = 0.5
        
        self.batch_size = 16
        self.init_lr = 1e-4
        self.end_lr = 1e-6
        self.warmup_epochs = 2
        self.epochs = 100
        self.transfer_weights = True

        self.test_score_thr = 0.3
        self.test_iou_thr = 0.45

    def _get_strides(self):
        if self.model_name.endswith('_tiny'):
            strides = [16, 32, 64]
        else:
            strides = [8, 16, 32]
        
        strides = np.array(strides)
        return strides

    def _get_anchors(self):
        if self.model_name.endswith('_tiny'):
            anchors = [[[10,  14], [23,   27], [37,   58]],
                       [[81,  82], [135, 169], [344, 319]],
                       [[0,    0], [0,     0], [0,     0]]]
        else:
            anchors = [[[10,   13], [16,   30], [33,   23]],
                       [[30,   61], [62,   45], [59,  119]],
                       [[116,  90], [156, 198], [373, 326]]]
        anchors = (np.array(anchors).T / self.strides).T
        return anchors


@enum.unique
class ProjectPath(enum.Enum):
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIGS_DIR = os.path.join(ROOT_DIR, 'configs')
    INITIAL_SETTINGS_PATH = os.path.join(CONFIGS_DIR, 'initial_settings.json')
    LIBS_DIR = os.path.join(ROOT_DIR, 'libs')
    EVAL_DIR = os.path.join(LIBS_DIR, 'eval')

    DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets')
    PREVIEW_DIR = os.path.join(ROOT_DIR, 'preview')
    
    CKPTS_DIR = os.path.join(ROOT_DIR, 'ckpts')
    COCO_PRETRAINED_CKPT_PATH = os.path.join(CKPTS_DIR, 'yolo_v3_coco.h5')
    COCO_PRETRAINED_TINY_CKPT_PATH = os.path.join(CKPTS_DIR, 'yolo_v3_tiny_coco.h5')
    LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
