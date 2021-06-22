import numpy as np
import tensorflow as tf
from libs.utils.calc_ious import bbox_giou, bbox_iou


__all__ = ['get_losses']


def get_losses(pred_raw, pred_decoded, label, bboxes, stride, iou_loss_thr, num_classes):
    """
    Args:
      pred_decoded: decoded yolo output
      pred_raw: raw yolo output
    """
    batch_size, grid_size = pred_raw.shape[0], pred_raw.shape[1]
    input_size  = tf.cast(stride * grid_size, tf.float32)
    pred_raw = tf.reshape(pred_raw, (batch_size, grid_size, grid_size, 3, 5+num_classes))

    pred_raw_conf = pred_raw[:, :, :, :, 4:5]
    pred_raw_prob = pred_raw[:, :, :, :, 5:]

    pred_decoded_xywh = pred_decoded[:, :, :, :, 0:4]
    pred_decoded_conf = pred_decoded[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    label_conf = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_decoded_xywh, label_xywh), axis=-1)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = label_conf * bbox_loss_scale * (1 - giou)

    # Find the value of IoU with the real box The largest prediction box
    ## pred_decoded_xywh.shape: [batch_size, y_idx, x_idx, num_scales, 4]
    ## bboxes.shape:            [batch_size, max_num_bboxes_per_scale, 4]
    iou = bbox_iou(pred_decoded_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    ## iou.shape: [batch_size, y_idx, x_idx, num_scales, max_num_bboxes_per_scale]
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    # If the largest iou is less than the threshold, it is considered that the prediction box contains no objects, then the background box
    respond_bg = (1.0 - label_conf) * tf.cast(max_iou < iou_loss_thr, tf.float32)
    conf_focal = tf.pow(label_conf - pred_decoded_conf, 2)

    # Calculate the loss of confidence
    # we hope that if the grid contains objects, then the network output prediction box has a confidence of 1 and 0 when there is no object.
    conf_loss = conf_focal * (
            label_conf * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_conf, logits=pred_raw_conf)
            +
            respond_bg * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_conf, logits=pred_raw_conf)
    )

    prob_loss = label_conf * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=pred_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))
    return giou_loss, conf_loss, prob_loss
