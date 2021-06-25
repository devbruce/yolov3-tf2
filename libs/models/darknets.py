import tensorflow as tf
from libs.models.layers import conv_block, residual_block


__all__ = ['darknet53', 'darknet19_tiny']


def darknet53(x):
    x = conv_block(x, out_channels=32, kernel_size=3)
    x = conv_block(x, out_channels=64, kernel_size=3, downsample=True)

    for _ in range(1):
        x = residual_block(x, mid_channels=32, out_channels=64)

    x = conv_block(x, out_channels=128, kernel_size=3, downsample=True)

    for _ in range(2):
        x = residual_block(x, mid_channels=64, out_channels=128)

    x = conv_block(x, out_channels=256, kernel_size=3, downsample=True)

    for _ in range(8):
        x = residual_block(x, mid_channels=128, out_channels=256)

    x_256 = x
    x = conv_block(x, out_channels=512, kernel_size=3, downsample=True)

    for _ in range(8):
        x = residual_block(x, mid_channels=256, out_channels=512)

    x_512 = x
    x = conv_block(x, out_channels=1024, kernel_size=3, downsample=True)

    for _ in range(4):
        x = residual_block(x, mid_channels=512, out_channels=1024)

    return x_256, x_512, x


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
    x_256 = x
    
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = conv_block(x, out_channels=512, kernel_size=3)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=1, padding='same')(x)
    x = conv_block(x, out_channels=1024, kernel_size=3)
    return x_256, x
