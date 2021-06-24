import os
import json
import tqdm
import cv2
import numpy as np
from libs.utils.augs import get_transform, letterbox
from configs import ProjectPath, cfg


__all__ = ['CocoDataLoader']


class CocoDataLoader:
    def __init__(self, stage, shuffle=False, aug=False, project_name=cfg.project_name, batch_size=cfg.batch_size, input_size=cfg.input_size):
        self.stage = stage
        self.shuffle = shuffle
        self.aug = aug
        self.project_name = project_name
        self.batch_size = batch_size
        self.input_size = input_size
    
    def __iter__(self):
        return _CocoDataIter(
            stage=self.stage,
            shuffle=self.shuffle,
            aug=self.aug,
            project_name=self.project_name,
            batch_size=self.batch_size,
            input_size=self.input_size
        )
    
    def __repr__(self):
        repr_form = '* project_name: {}\n- stage: {}\n- shuffle={}\n- aug={}\n- batch_size={}\n- input_size={}'
        return repr_form.format(self.project_name, self.stage, self.shuffle, self.aug, self.batch_size, self.input_size)


class _CocoDataIter:
    def __init__(self, stage, shuffle=False, aug=False, project_name=cfg.project_name, batch_size=cfg.batch_size, input_size=cfg.input_size):
        self.stage = stage
        self.shuffle = shuffle
        self.aug = aug
        self.project_name = project_name
        self.batch_size = batch_size
        self.input_size = input_size
        
        self.dataset_dir = os.path.join(ProjectPath.DATASETS_DIR.value, self.project_name)
        self.imgs_dir = os.path.join(self.dataset_dir, 'imgs', self.stage)
        self.coco_ann_path = os.path.join(self.dataset_dir, 'labels', f'{self.stage}.json')
        
        self.anns = self._load_anns()
        self.num_imgs = len(self.anns)
        self.num_batchs = int(np.ceil(self.num_imgs / self.batch_size))
        
        self.idx = 0
        self.batch_idx = 0
        
    def __len__(self):
        return self.num_batchs
    
    def __iter__(self):
        return self
        
    def __next__(self):
        img_arr_list = list()
        labels_list = list()
        for _ in range(self.batch_size):
            if self.idx >= self.num_imgs:
                break
            
            ann = self.anns[self.idx]
            img_arr = cv2.cvtColor(cv2.imread(ann['image_path']), cv2.COLOR_BGR2RGB)
            img_height, img_width = img_arr.shape[:2]
            bboxes, class_indices = list(), list()
            for a in ann['annotations']:
                bboxes.append(a['bbox'])
                class_indices.append(a['class_index'])
            
            # Augmentation
            if self.aug:
                transform = get_transform(img_height=img_height, img_width=img_width)
                transformed = transform(image=img_arr, bboxes=bboxes, class_indices=class_indices)
                img_arr = transformed['image']
                bboxes = transformed['bboxes']
                class_indices = transformed['class_indices']

                if len(bboxes) == 0:
                    self.idx += 1
                    continue
            
            bboxes = np.array(bboxes, dtype=np.float32)
            class_indices = np.array(class_indices, dtype=np.float32).reshape(-1, 1)
            
            # Letterbox
            img_arr, bboxes = letterbox(
                img_arr=img_arr,
                target_size=(self.input_size, self.input_size),
                boxes=bboxes,
            )
            img_arr /= 255.
            
            # Image
            img_arr_list.append(img_arr)
            
            # Labels
            labels = np.hstack([bboxes, class_indices])
            labels_list.append(labels.astype(np.float32))
            
            self.idx += 1
        
        if self.batch_idx >= self.num_batchs:
            raise StopIteration()
        
        self.batch_idx += 1
        imgs = np.stack(img_arr_list).astype(np.float32)
        return imgs, labels_list
    
    def _load_anns(self):
        anns = list()
        coco_ann_json = json.load(open(self.coco_ann_path))
        coco_imgs = coco_ann_json['images']
        coco_anns = coco_ann_json['annotations']
        for coco_img in tqdm.tqdm(coco_imgs, total=len(coco_imgs), desc=f'[{self.stage}]Loading Dataset'):
            img_id = coco_img['id']
            fname = coco_img['file_name']
            img_path = os.path.join(self.imgs_dir, fname)
            
            ann = dict()
            ann['image_path'] = img_path
            ann['annotations'] = list()
            
            for coco_ann in coco_anns:
                if coco_ann['image_id'] == img_id:
                    cls_idx = coco_ann['category_id'] - 1  # !Warning
                    pts = coco_ann['bbox']
                    left, top, width, height = pts
                    right = left + width
                    bottom = top + height
                    
                    ann_info = dict()
                    ann_info['bbox'] = [left, top, right, bottom]
                    ann_info['class_index'] = cls_idx
                    ann['annotations'].append(ann_info)
                    
            anns.append(ann)
            
        if self.shuffle:
            np.random.shuffle(anns)
        return anns
