import numpy as np
import tensorflow as tf
from libs.models import get_yolo
from libs.utils.augs import letterbox
from configs import cfg


__all__ = ['YoloInf']


class YoloInf:
    def __init__(self, ckpt_path, nms_method='nms', nms_iou_thr=0.45, nms_sigma=0.3, cfg=cfg):
        self.cfg = cfg
        self.ckpt_path = ckpt_path
        self.model = get_yolo(ckpt_path=self.ckpt_path, cfg=cfg, training=False)
        self.nms_method = nms_method
        self.nms_iou_thr = nms_iou_thr
        self.nms_sigma = nms_sigma
        self.input_size = cfg.input_size
        self.classes = cfg.classes

    def get(self, img_arr, conf_thr=0.3):
        origin_img_height, origin_img_width = img_arr.shape[:2]
        img_arr = img_arr.copy()
        img_arr = letterbox(img_arr, (self.input_size, self.input_size)) / 255.
        img_arr = img_arr[np.newaxis, ...].astype(np.float32)
        img_arr = tf.convert_to_tensor(img_arr, dtype=tf.float32)

        output_raw = self.model(img_arr, training=False)
        preds = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in output_raw]
        preds = tf.concat(preds, axis=0)

        preds_prep = self._postprocess_boxes(
            preds=preds,
            origin_img_height=origin_img_height,
            origin_img_width=origin_img_width,
            conf_thr=conf_thr,
        )
        preds_prep = self._nms(preds_prep)
        pred_results = list()
        for pred in preds_prep:
            x_min, y_min, x_max, y_max, confidence, cls_idx = pred
            pred_form = dict()
            pred_form['bbox'] = list(map(round, [x_min, y_min, x_max, y_max]))
            pred_form['confidence'] = float(confidence)
            pred_form['class_index'] = int(cls_idx) + 1
            pred_form['class_name'] = cfg.classes[str(int(cls_idx)+1)]
            pred_results.append(pred_form)
        return pred_results

    def _postprocess_boxes(self, preds, origin_img_height, origin_img_width, conf_thr):
        valid_scale=[0, np.inf]
        preds = np.array(preds)

        pred_xywh = preds[:, 0:4]
        pred_conf = preds[:, 4]
        pred_prob = preds[:, 5:]

        # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                    pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        resize_ratio = min(self.input_size / origin_img_width, self.input_size / origin_img_height)

        dw = (self.input_size - resize_ratio * origin_img_width) / 2
        dh = (self.input_size - resize_ratio * origin_img_height) / 2

        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # 3. clip some boxes those are out of range
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [origin_img_width-1, origin_img_height-1])], axis=-1)
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # 4. discard some invalid boxes
        ## pred_coor[:, 2:4] - pred_coor[:, 0:2] --> [[width, height], ...]
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # 5. discard boxes with low confidence
        classes = np.argmax(pred_prob, axis=-1)
        conf = pred_conf * pred_prob[:, classes]
        conf_mask = conf > conf_thr
        mask = np.logical_and(scale_mask, conf_mask)
        coors, conf, classes = pred_coor[mask], conf[mask], classes[mask]
        return np.concatenate([coors, conf[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

    def _nms(self, bboxes):
        """
        Args:
        bboxes: (left, top, right, bottom, confidence, class_index)

        Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
            https://github.com/bharatsingh430/soft-nms
        """
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]
            # Process 1: Determine whether the number of bounding boxes is greater than 0 
            while len(cls_bboxes) > 0:
                # Process 2: Select the bounding box with the highest confidence according to confidence order A
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                # Process 3: Calculate this bounding box A and
                # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold 
                iou = self._calc_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                assert self.nms_method in ['nms', 'soft-nms']

                if self.nms_method == 'nms':
                    iou_mask = iou > self.nms_iou_thr
                    weight[iou_mask] = 0.0

                if self.nms_method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / self.nms_sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]
        return best_bboxes

    def _calc_iou(self, boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)
        return ious
