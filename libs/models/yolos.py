import tensorflow as tf
from libs.models.darknets import darknet53, darknet19_tiny
from libs.models.layers import conv_block, upsample


__all__ = ['yolo_v3', 'yolo_v3_tiny']


def yolo_v3(input_layer, num_classes):
    sobj_fm, mobj_fm, x = darknet53(input_layer)
    x = conv_block(x, out_channels=512, kernel_size=1)
    x = conv_block(x, out_channels=1024, kernel_size=3)
    x = conv_block(x, out_channels=512, kernel_size=1)
    x = conv_block(x, out_channels=1024, kernel_size=3)
    x = conv_block(x, out_channels=512, kernel_size=1)
    
    # lobj_branch is used to predict large-sized objects , Shape = [None, 13, 13, (num_classes+5)*3]
    lobj_branch = conv_block(x, out_channels=1024, kernel_size=3)
    lobj_branch = conv_block(lobj_branch, out_channels=3*(num_classes+5), kernel_size=1, activate=None, bn=False)

    x = conv_block(x, out_channels=256, kernel_size=1)
    x = upsample(x)

    x = tf.concat([x, mobj_fm], axis=-1)
    x = conv_block(x, out_channels=256, kernel_size=1)
    x = conv_block(x, out_channels=512, kernel_size=3)
    x = conv_block(x, out_channels=256, kernel_size=1)
    x = conv_block(x, out_channels=512, kernel_size=3)
    x = conv_block(x, out_channels=256, kernel_size=1)

    # mobj_branch is used to predict medium-sized objects, shape = [None, 26, 26, (num_classes+5)*3]
    mobj_branch = conv_block(x, out_channels=512, kernel_size=3)
    mobj_branch = conv_block(mobj_branch, out_channels=3*(num_classes+5), kernel_size=1, activate=None, bn=False)

    x = conv_block(x, out_channels=128, kernel_size=1)
    x = upsample(x)

    x = tf.concat([x, sobj_fm], axis=-1)
    x = conv_block(x, out_channels=128, kernel_size=1)
    x = conv_block(x, out_channels=256, kernel_size=3)
    x = conv_block(x, out_channels=128, kernel_size=1)
    x = conv_block(x, out_channels=256, kernel_size=3)
    x = conv_block(x, out_channels=128, kernel_size=1)
    
    # sobj_branch is used to predict small size objects, shape = [None, 52, 52, (num_classes+5)*3]
    sobj_branch = conv_block(x, out_channels=256, kernel_size=3)
    sobj_branch = conv_block(sobj_branch, out_channels=3*(num_classes+5), kernel_size=1, activate=None, bn=False)
    return [sobj_branch, mobj_branch, lobj_branch]


def yolo_v3_tiny(input_layer, num_classes):
    mobj_fm, x = darknet19_tiny(input_layer)
    x = conv_block(x, out_channels=256, kernel_size=1)
    
    # lobj_branch is used to predict large-sized objects , Shape = [None, 26, 26, (num_classes+5)*3]
    lobj_branch = conv_block(x, out_channels=512, kernel_size=3)
    lobj_branch = conv_block(lobj_branch, out_channels=3*(num_classes+5), kernel_size=1, activate=None, bn=False)

    x = conv_block(x, out_channels=128, kernel_size=1)
    x = upsample(x)
    
    x = tf.concat([x, mobj_fm], axis=-1)
    
    # mobj_branch is used to predict medium size objects, shape = [None, 13, 13, (num_classes+5)*3]
    mobj_branch = conv_block(x, out_channels=256, kernel_size=3)
    mobj_branch = conv_block(mobj_branch, out_channels=3*(num_classes+5), kernel_size=1, activate=None, bn=False)
    return [mobj_branch, lobj_branch]
