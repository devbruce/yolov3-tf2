import tensorflow as tf


__all__ = ['conv_block', 'residual_block', 'upsample']


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer,
    so the layer will use stored moving `var` and `mean` in the "inference mode",
    and both `gama` and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def conv_block(x, out_channels, kernel_size, downsample=False, activate='leaky', bn=True):
    if downsample:
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)  # Top, Left half padding
        strides, padding = 2, 'valid'
    else:
        strides, padding = 1, 'same'

    x = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=not bn,
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.)
    )(x)

    x = BatchNormalization()(x) if bn else x
    if activate == 'leaky':
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    else:
        x = tf.keras.layers.Activation(activate)(x)
    return x


def residual_block(x, mid_channels, out_channels, activate='leaky'):
    short_cut = x
    x = conv_block(x, out_channels=mid_channels, kernel_size=1, activate=activate)
    x = conv_block(x, out_channels=out_channels, kernel_size=3, activate=activate)
    x = short_cut + x
    return x


def upsample(x):
    """
    Upsample here uses the nearest neighbor interpolation method, which has the advantage that the
    upsampling process does not need to learn, thereby reducing the network parameter
    """
    height, width = x.shape[1], x.shape[2]
    return tf.image.resize(x, (height*2, width*2), method='nearest')
