import tensorflow as tf
from libs.models.yolos import yolo_v3, yolo_v3_tiny
from configs import cfg


__all__ = ['get_yolo']


def get_yolo(ckpt_path=None, training=False, cfg=cfg):
    if cfg.model_name == 'yolo_v3':
        model = yolo_v3
    elif cfg.model_name == 'yolo_v3_tiny':
        model = yolo_v3_tiny
    else:
        raise ValueError(f"{cfg.model_name} doesn't exist.\n* Available model: ['yolo_v3', 'yolo_v3_tiny']\n")

    input_layer = tf.keras.layers.Input([cfg.input_size, cfg.input_size, cfg.input_channels])
    branches_per_scale = model(input_layer, cfg.num_classes)
    # branches_per_scale: [sobj_branch, mobj_branch, lobj_branch] (tiny: [mobj_branch, lobj_branch])

    out_branches = list()
    for scale_idx, branch in enumerate(branches_per_scale):
        if training:
            out_branches.append(branch)
        decoded_branch = _decode(branch, cfg.num_classes, cfg.strides, cfg.anchors, scale_idx)
        out_branches.append(decoded_branch)
    yolo = tf.keras.models.Model(input_layer, out_branches)

    if ckpt_path:
        yolo.load_weights(ckpt_path)
    return yolo


def _decode(yolo_branch, num_classes, strides, anchors, scale_idx):
    conv_shape = tf.shape(yolo_branch)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]

    yolo_branch = tf.reshape(yolo_branch, (batch_size, output_size, output_size, 3, 5+num_classes))

    #conv_raw_dxdy = yolo_branch[:, :, :, :, 0:2] # offset of center position     
    #conv_raw_dwdh = yolo_branch[:, :, :, :, 2:4] # Prediction box length and width offset
    #conv_raw_conf = yolo_branch[:, :, :, :, 4:5] # confidence of the prediction box
    #conv_raw_prob = yolo_branch[:, :, :, :, 5: ] # category probability of the prediction box
    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(yolo_branch, (2, 2, 1, num_classes), axis=-1)

    # next need Draw the grid. Where output_size is equal to 13, 26 or 52
    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    # Calculate the center position of the prediction box:
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * strides[scale_idx]
    # Calculate the length and width of the prediction box:
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors[scale_idx]) * strides[scale_idx]

    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf)  # object box calculates the predicted confidence
    pred_prob = tf.sigmoid(conv_raw_prob)  # calculating the predicted probability category box object

    # calculating the predicted probability category box object
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
