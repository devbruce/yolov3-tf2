import tensorflow as tf
from libs.models.layers import conv_block, residual_block


__all__ = ['darknet53', 'darknet19_tiny']


def darknet53(x):
    x = conv_block(x, out_channels=32, kernel_size=3)
    x = conv_block(x, out_channels=64, kernel_size=3, downsample=True)

    for i in range(1):
        x = residual_block(x, mid_channels=32, out_channels=64)

    x = conv_block(x, out_channels=128, kernel_size=3, downsample=True)

    for i in range(2):
        x = residual_block(x, mid_channels=64, out_channels=128)

    x = conv_block(x, out_channels=256, kernel_size=3, downsample=True)

    for i in range(8):
        x = residual_block(x, mid_channels=128, out_channels=256)

    sobj_fm = x
    x = conv_block(x, out_channels=512, kernel_size=3, downsample=True)

    for i in range(8):
        x = residual_block(x, mid_channels=256, out_channels=512)

    mobj_fm = x
    x = conv_block(x, out_channels=1024, kernel_size=3, downsample=True)

    for i in range(4):
        x = residual_block(x, mid_channels=512, out_channels=1024)

    return sobj_fm, mobj_fm, x


def darknet19_tiny(x):
    x = conv_block(x, out_channels=16, kernel_size=3)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = conv_block(x, out_channels=32, kernel_size=3)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = conv_block(x, out_channels=64, kernel_size=3)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = conv_block(x, out_channels=128, kernel_size=3)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = conv_block(x, out_channels=256, kernel_size=3)
    mobj_fm = x
    
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = conv_block(x, out_channels=512, kernel_size=3)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=1, padding='same')(x)
    x = conv_block(x, out_channels=1024, kernel_size=3)
    return mobj_fm, x
