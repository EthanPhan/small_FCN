import tensorflow as tf
import warnings
from distutils.version import LooseVersion

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn(
        'No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


KEEP_PROB = 0.75


def kernel_initializer():
    return tf.contrib.layers.xavier_initializer()


def regularizer():
    return tf.contrib.layers.l2_regularizer(scale=0.01)


def coordconv_wraper(input):
    batch_size_tensor = tf.shape(input)[0]
    x_dim = tf.shape(input)[1]
    y_dim = tf.shape(input)[2]
    xx_ones = tf.ones([batch_size_tensor, x_dim], dtype=tf.int32)
    xx_ones = tf.expand_dims(xx_ones, -1)
    xx_range = tf.tile(tf.expand_dims(
        tf.range(y_dim), 0), [batch_size_tensor, 1])
    xx_range = tf.expand_dims(xx_range, 1)
    xx_channel = tf.matmul(xx_ones, xx_range)
    xx_channel = tf.expand_dims(xx_channel, -1)

    yy_ones = tf.ones([batch_size_tensor, y_dim], dtype=tf.int32)
    yy_ones = tf.expand_dims(yy_ones, 1)
    yy_range = tf.tile(tf.expand_dims(
        tf.range(x_dim), 0), [batch_size_tensor, 1])
    yy_range = tf.expand_dims(yy_range, -1)
    yy_channel = tf.matmul(yy_range, yy_ones)
    yy_channel = tf.expand_dims(yy_channel, -1)

    xx_channel = tf.cast(xx_channel, 'float32') / \
        tf.cast((x_dim - 1), 'float32')
    yy_channel = tf.cast(yy_channel, 'float32') / \
        tf.cast((y_dim - 1), 'float32')
    xx_channel = xx_channel*2 - 1
    yy_channel = yy_channel*2 - 1
    ret = tf.concat([input, xx_channel, yy_channel], axis=-1)

    return ret


def conv2d_layer(inp_tensor, num_kernels, kernel_size, name,
                 reuse=None):
    inp_tensor = coordconv_wraper(inp_tensor)
    conv = tf.layers.conv2d(
        inputs=inp_tensor,
        filters=num_kernels,
        kernel_size=(kernel_size, kernel_size),
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu,
        normalizer_fn=tf.contrib.layers.layer_norm,
        kernel_initializer=kernel_initializer(),
        kernel_regularizer=regularizer(),
        name=name,
        reuse=reuse
    )
    return conv


def rcl(X, num_kernels, kernel_size, scope_name=None):
    with tf.variable_scope(scope_name) as scope:
        conv1 = conv2d_layer(X, num_kernels, kernel_size,
                             'rcl')
        rcl1 = tf.add(conv1, X)
        bn1 = tf.contrib.layers.batch_norm(rcl1)
        #
        conv2 = conv2d_layer(bn1, num_kernels, kernel_size,
                             'rcl', True)
        rcl2 = tf.add(conv2, X)
        bn2 = tf.contrib.layers.batch_norm(rcl2)
        #
        conv3 = conv2d_layer(bn2, num_kernels, kernel_size,
                             'rcl', True)
        rcl3 = tf.add(conv3, X)
        bn3 = tf.contrib.layers.batch_norm(rcl3)

        return bn3


def residual_layer(input, num_classes, name=None):
    input = coordconv_wraper(input)
    res = tf.layers.conv2d(
        inputs=input,
        filters=num_classes,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=kernel_initializer(),
    )
    res = rcl(res, num_classes, 1, name)  # 96, 64, 64
    return res


def deconv2d_x2_layer(input, num_classes, name=None):
    input = coordconv_wraper(input)
    deconv = tf.layers.conv2d_transpose(
        inputs=input,
        filters=num_classes,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same',
        kernel_initializer=kernel_initializer(),
        normalizer_fn=tf.contrib.layers.layer_norm
    )
    deconv = rcl(deconv, num_classes, 4, name)  # 96, 64, 64
    return deconv


def full_network(num_classes, training=True):
    _input = tf.placeholder(dtype=tf.float32, shape=[
                            None, 768, 512, 1], name='input_tensor')

    # conv1
    conv1_1 = conv2d_layer(_input, 8, 3, 'conv1_1')
    conv1_2 = rcl(conv1_1, 8, 3, 'rcl1')  # 768, 512, 8
    pool1 = tf.layers.max_pooling2d(
        conv1_2, (2, 2), (2, 2), padding='same', name='pool1')

    # conv2
    conv2_1 = conv2d_layer(pool1, 16, 3, 'conv2_1')
    conv2_2 = rcl(conv2_1, 16, 3, 'rcl2')  # 384, 256, 16
    pool2 = tf.layers.max_pooling2d(
        conv2_2, (2, 2), (2, 2), padding='same', name='pool2')

    # conv3
    conv3_1 = conv2d_layer(pool2, 32, 3, 'conv3_1')
    conv3_2 = rcl(conv3_1, 32, 3, 'rcl3')  # 192, 128, 32
    pool3 = tf.layers.max_pooling2d(
        conv3_2, (2, 2), (2, 2), padding='same', name='pool3')

    # conv4
    conv4_1 = conv2d_layer(pool3, 64, 3, 'conv4_1')
    conv4_2 = rcl(conv4_1, 64, 3, 'rcl4')  # 96, 64, 64
    pool4 = tf.layers.max_pooling2d(
        conv4_2, (2, 2), (2, 2), padding='same', name='pool4')

    # fc5
    fc5 = conv2d_layer(pool4, 512, 7, 'fc5')  # 48, 32, 512
    fc5 = rcl(fc5, 512, 7, 'rcl5')  # 48, 32, 512
    drop5 = tf.layers.dropout(fc5, rate=1 - KEEP_PROB,
                              training=training)  # 48, 32, 512

    # fc6
    fc6 = conv2d_layer(drop5, 512, 1, 'fc6')  # 48, 32, 512
    fc6 = rcl(fc6, 512, 1, 'rcl6')  # 96, 64, 64
    drop6 = tf.layers.dropout(fc6, rate=1 - KEEP_PROB,
                              training=training)  # 48, 32, 512

    # fc7
    fc7 = residual_layer(drop6, num_classes, 'fc7')  # 48, 32, num_classes

    pool3_res = residual_layer(
        pool3, num_classes, 'pool3_res')  # 96, 64, num_classes
    pool2_res = residual_layer(
        pool2, num_classes, 'pool2_res')  # 192, 128, num_classes

    # Deconv layers
    deconv8 = deconv2d_x2_layer(
        fc7, num_classes, 'deconv8')  # 96, 64, num_classes
    sum8 = tf.add(deconv8, pool3_res)  # 96, 64, num_classes

    deconv9 = deconv2d_x2_layer(
        sum8, num_classes, 'deconv9')  # 192, 128, num_classes
    sum9 = tf.add(deconv9, pool2_res)  # 192, 128, num_classes

    sum9 = coordconv_wraper(sum9)
    out = tf.layers.conv2d_transpose(
        inputs=sum9,
        filters=num_classes,
        kernel_size=(8, 8),
        strides=(4, 4),
        padding='same',
        kernel_initializer=kernel_initializer(),
        kernel_regularizer=regularizer())  # 768, 512, num_classes
    out = rcl(out, num_classes, 8, 'out')  # 96, 64, 64

    return out, _input
