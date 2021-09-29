import tensorflow as tf
from tensorflow.keras.optimizers import Adam

initializer = tf.keras.initializers.HeNormal(seed=543)


def double_conv_block_down(initializer, x, filters_first, filters_second):
    x = tf.keras.layers.Conv2D(filters_first, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters_second, 3, strides=2, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def double_conv_block_up(initializer, skip, x, filters_first, filters_second):
    if skip is not None:
        x = tf.keras.layers.Concatenate()([skip, x])
    x = tf.keras.layers.Conv2DTranspose(filters_first, 3, strides=1, padding='same',
                                        kernel_initializer=initializer)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(filters_second, 3, strides=2, padding='same',
                                        kernel_initializer=initializer)(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def no_pool(input, initializer, filters_first, filters_second):
    x = tf.keras.layers.Conv2D(filters_first, 3, strides=1, padding='same', kernel_initializer=initializer)(input)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters_second, 3, strides=1, padding='same', kernel_initializer=initializer)(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def create_model():
    inputs = tf.keras.layers.Input(shape=[None, None, 1])

    x = no_pool(inputs, initializer, 16, 16)
    skips = [x]

    x = double_conv_block_down(initializer, x, filters_first=64, filters_second=64)
    skips.append(x)
    x = double_conv_block_down(initializer, x, filters_first=128, filters_second=128)
    skips.append(x)
    # middle code
    x = double_conv_block_down(initializer, x, filters_first=128, filters_second=128)

    x = double_conv_block_up(initializer, None, x, filters_first=128, filters_second=128)
    x = double_conv_block_up(initializer, skips[-1], x, filters_first=128, filters_second=128)
    x = double_conv_block_up(initializer, skips[-2], x, filters_first=64, filters_second=64)

    x = tf.keras.layers.Concatenate()([skips[0], x])
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(1, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.activations.sigmoid(x)
    return tf.keras.Model(inputs=inputs, outputs=x)