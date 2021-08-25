import os
import json
import copy
import tqdm
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from libs.utils import PathManager


__all__ = ['GetCocoEval']


class GetCocoEval:
    def __init__(self, img_prefix, coco_gt_path, yolo_inf, conf_thr=0.05, img_exts=['.png', '.jpg', '.jpeg']):
        self.img_prefix = img_prefix
        self.coco_gt_path = coco_gt_path
        self.coco_gt = json.load(open(coco_gt_path))
        self.yolo_inf = yolo_inf
        self.conf_thr = conf_thr
        self.img_exts = img_exts
        self.cat_map = self._get_coco_cat_map()
        self.fname2img_id_map = self._get_fname2img_id_map()
        self.img_fnames = self._get_img_fnames()
        self.coco_preds = self._get_coco_preds()
        
    def _get_coco_cat_map(self):
        cats = self.coco_gt['categories']
        cat_map_from_coco_gt = dict()
        for cat in cats:
            id_ = cat['id']
            name = cat['name']
            cat_map_from_coco_gt[name] = id_
        return cat_map_from_coco_gt
    
    def _get_fname2img_id_map(self):
        imgs = self.coco_gt['images']
        fname2img_id_map = dict()
        for img_info in imgs:
            img_id = img_info['id']
            img_fname = img_info['file_name']
            fname2img_id_map[img_fname] = img_id
        return fname2img_id_map
    
    def _get_img_fnames(self):
        img_fnames = list()
        fpaths = PathManager.get_fpaths(self.img_prefix, exts=self.img_exts)
        for fpath in fpaths:
            path_info = PathManager(path=fpath)
            parent_path = path_info.get_parent_path(prefix_dir=self.img_prefix)
            parent_path = parent_path[1:] if parent_path else ''
            fname = path_info.fname
            img_fname = os.path.join(parent_path, fname)  # filename == sub_path
            img_fnames.append(img_fname)
        img_fnames.sort()
        return img_fnames
    
    def _get_coco_preds(self):
        coco_preds = list()
        for img_fname in tqdm.tqdm(self.img_fnames, total=len(self.img_fnames), desc='Creating COCO Predictions'):
            if self.fname2img_id_map.get(img_fname) is None:
                print(f'* Filtered not exist in COCO GT File {img_fname}')
                continue

            # Load Image
            img_path = os.path.join(self.img_prefix, img_fname)
            img_arr = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            # Ger Predictions
            preds = self.yolo_inf.get(img_arr=img_arr, conf_thr=self.conf_thr)

            # Convert origin predictions to coco predictions
            for pred in preds:
                cls_name = pred['class_name']
                confidence = pred['confidence']
                left, top, right, bottom = pred['bbox']
                width = right - left
                height = bottom - top
                coco_pred = dict()
                coco_pred['image_id'] = self.fname2img_id_map.get(img_fname)
                coco_pred['category_id'] = self.cat_map[cls_name]
                coco_pred['bbox'] = [left, top, width, height]
                coco_pred['score'] = confidence
                coco_preds.append(coco_pred)
                
        return coco_preds

    def get(self, verbose=True):
        if not self.coco_preds:
            print('\n* No predictions\n')
            return
        cocoGt = COCO(self.coco_gt_path)
        cocoDt = cocoGt.loadRes(copy.deepcopy(self.coco_preds))
        cocoEval = COCOeval(cocoGt=cocoGt, cocoDt=cocoDt, iouType='bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        if verbose:
            cocoEval.summarize()
            
        precisions = cocoEval.eval['precision']
        cat_ids = list(self.cat_map.values())
        assert len(cat_ids) == precisions.shape[2]

        # ====== Classwise ============
        cls_ap_map = dict()
        for idx, cat_id in enumerate(cat_ids):
            cls_info = cocoGt.loadCats(cat_id)[0]
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float('nan')
            cls_name = cls_info['name']
            cls_ap_map[cls_name] = float(ap)
        # ====== ========= ============

        cls_ap_map['mAP'] = cocoEval.stats[0]
        
        if verbose:
            print('\n====== APs per Class (@[ IoU=0.50:0.95 | area=   all | maxDets=100 ]) ======')
            for cls_name, ap in cls_ap_map.items():
                print(f'* {cls_name}: {ap:.3f}')
            print('=' * 77)
        
        return cls_ap_map
