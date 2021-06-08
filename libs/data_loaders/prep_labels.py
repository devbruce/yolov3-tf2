import numpy as np
from libs.utils.calc_ious import bbox_iou


__all__ = ['PrepLabels']


class PrepLabels:
    def __init__(self, batch_labels, cfg):
        self.batch_labels = batch_labels
        self.batch_size = len(batch_labels)
        self.num_classes = cfg.num_classes
        self.grid_sizes = cfg.grid_sizes
        self.anchors_per_scale = cfg.anchors_per_scale
        self.strides = cfg.strides
        self.anchors = cfg.anchors
        self.max_num_bboxes_per_scale = cfg.max_num_bboxes_per_scale
    
    def get_prep(self):
        batch_label_sbbox = np.zeros((self.batch_size, self.grid_sizes[0], self.grid_sizes[0], self.anchors_per_scale, 5+self.num_classes), dtype=np.float32)
        batch_label_mbbox = np.zeros((self.batch_size, self.grid_sizes[1], self.grid_sizes[1], self.anchors_per_scale, 5+self.num_classes), dtype=np.float32)
        batch_label_lbbox = np.zeros((self.batch_size, self.grid_sizes[2], self.grid_sizes[2], self.anchors_per_scale, 5+self.num_classes), dtype=np.float32)

        batch_sbboxes = np.zeros((self.batch_size, self.max_num_bboxes_per_scale, 4), dtype=np.float32)
        batch_mbboxes = np.zeros((self.batch_size, self.max_num_bboxes_per_scale, 4), dtype=np.float32)
        batch_lbboxes = np.zeros((self.batch_size, self.max_num_bboxes_per_scale, 4), dtype=np.float32)

        for idx, labels in enumerate(self.batch_labels):
            label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self._prep_labels_per_scale(labels=labels)
            batch_label_sbbox[idx, :, :, :, :] = label_sbbox
            batch_label_mbbox[idx, :, :, :, :] = label_mbbox
            batch_label_lbbox[idx, :, :, :, :] = label_lbbox
            batch_sbboxes[idx, :, :] = sbboxes
            batch_mbboxes[idx, :, :] = mbboxes
            batch_lbboxes[idx, :, :] = lbboxes

        batch_small_target = (batch_label_sbbox, batch_sbboxes)
        batch_medium_target = (batch_label_mbbox, batch_mbboxes)
        batch_large_target = (batch_label_lbbox, batch_lbboxes)
        return batch_small_target, batch_medium_target, batch_large_target
    
    def _prep_labels_per_scale(self, labels):
        labels_per_scale = [np.zeros((self.grid_sizes[i], self.grid_sizes[i], self.anchors_per_scale, 5+self.num_classes)) for i in range(3)]
        cxcywh_per_scale = [np.zeros((self.max_num_bboxes_per_scale, 4)) for _ in range(3)]
        n_bboxes_per_scale = np.zeros((3,))

        for label in labels:
            # Box Coordinates
            ltrb = label[:4]
            cxcy = (ltrb[2:] + ltrb[:2]) * 0.5
            wh = ltrb[2:] - ltrb[:2]
            cxcywh = np.concatenate([cxcy, wh], axis=-1)
            cxcywh_scaled = 1.0 * cxcywh[np.newaxis, :] / self.strides[:, np.newaxis]  # with broad-casting

            # Class Index
            cls_idx = label[4]
            smooth_onehot = self._smooth_onehot(class_index=cls_idx)

            iou = list()
            exist_positive = False
            for i in range(3):
                anchors_cxcywh = np.zeros((self.anchors_per_scale, 4))
                anchors_cxcywh[:, 0:2] = np.floor(cxcywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # anchor center coordinates in feature map
                anchors_cxcywh[:, 2:4] = self.anchors[i]

                iou_scale = bbox_iou(cxcywh_scaled[i][np.newaxis, :], anchors_cxcywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    x_idx, y_idx = np.floor(cxcywh_scaled[i, 0:2]).astype(np.int32)

                    labels_per_scale[i][y_idx, x_idx, iou_mask, :] = 0
                    labels_per_scale[i][y_idx, x_idx, iou_mask, 0:4] = cxcywh         # Box Points
                    labels_per_scale[i][y_idx, x_idx, iou_mask, 4:5] = 1.0            # Confidence
                    labels_per_scale[i][y_idx, x_idx, iou_mask, 5:] = smooth_onehot   # Class Index with Smooth OneHot

                    bbox_idx = int(n_bboxes_per_scale[i] % self.max_num_bboxes_per_scale)
                    cxcywh_per_scale[i][bbox_idx, :4] = cxcywh
                    n_bboxes_per_scale[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_idx = np.argmax(np.array(iou).reshape(-1), axis=-1)  # Number of ious: 3 x 3 (n_anchors x n_scales)
                best_detect_scale_idx = int(best_anchor_idx / self.anchors_per_scale)
                best_anchor_idx_in_scale = int(best_anchor_idx % self.anchors_per_scale)
                x_idx, y_idx = np.floor(cxcywh_scaled[best_detect_scale_idx, 0:2]).astype(np.int32)

                labels_per_scale[best_detect_scale_idx][y_idx, x_idx, best_anchor_idx_in_scale, :] = 0
                labels_per_scale[best_detect_scale_idx][y_idx, x_idx, best_anchor_idx_in_scale, 0:4] = cxcywh
                labels_per_scale[best_detect_scale_idx][y_idx, x_idx, best_anchor_idx_in_scale, 4:5] = 1.0
                labels_per_scale[best_detect_scale_idx][y_idx, x_idx, best_anchor_idx_in_scale, 5:] = smooth_onehot

                bbox_idx = int(n_bboxes_per_scale[best_detect_scale_idx] % self.max_num_bboxes_per_scale)
                cxcywh_per_scale[best_detect_scale_idx][bbox_idx, :4] = cxcywh
                n_bboxes_per_scale[best_detect_scale_idx] += 1

        label_sbbox, label_mbbox, label_lbbox = labels_per_scale
        sbboxes, mbboxes, lbboxes = cxcywh_per_scale
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


    def _smooth_onehot(self, class_index, delta=1e-2):
        onehot = np.zeros(self.num_classes, dtype=np.float32)
        onehot[int(class_index)] = 1.0
        uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
        smooth_onehot = onehot * (1 - delta) + delta * uniform_distribution
        return smooth_onehot
