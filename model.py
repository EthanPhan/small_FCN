import tensorflow as tf
from tensorflow.layers import conv2d, conv2d_transpose, max_pooling2d
from distutils.version import LooseVersion

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)

"""
# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn(
        'No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
"""


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


def conv2d_layer(inp_tensor, num_kernels, kernel_size, name, reuse=None):
    inp_tensor = coordconv_wraper(inp_tensor)
    conv = conv2d(
        inputs=inp_tensor,
        filters=num_kernels,
        kernel_size=(kernel_size, kernel_size),
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=kernel_initializer(),
        kernel_regularizer=regularizer(),
        name=name,
        reuse=reuse
    )
    return conv


def rcl(X, num_kernels, kernel_size, scope_name=None):
    with tf.variable_scope(scope_name) as scope:
        input = conv2d_layer(X, num_kernels, kernel_size, 'input')

        for i in range(3):
            reuse = (i > 0)
            conv = conv2d_layer(input, num_kernels, kernel_size,
                                'rcl', reuse)
            rcl = tf.add(conv, input)
            bn = tf.contrib.layers.batch_norm(rcl)

            input = bn

        return bn


def residual_layer(input, num_classes):
    input = coordconv_wraper(input)
    res = conv2d(
        inputs=input,
        filters=num_classes,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=kernel_initializer(),
    )
    return res


def deconv2d_x2_layer(input, num_classes):
    input = coordconv_wraper(input)
    deconv = conv2d_transpose(
        inputs=input,
        filters=num_classes,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same',
        kernel_initializer=kernel_initializer(),
    )
    return deconv


'''
def attention(tensor, att_tensor, n_filters=512, kernel_size=[1, 1]):
    g1 = conv2d(tensor, n_filters, kernel_size=kernel_size)
    x1 = conv2d(att_tensor, n_filters, kernel_size=kernel_size)
    net = add(g1, x1)
    net = tf.nn.relu(net)
    net = conv2d(net, 1, kernel_size=kernel_size)
    net = tf.nn.sigmoid(net)
    #net = tf.concat([att_tensor, net], axis=-1)
    net = net * att_tensor
    return net
'''


def hw_flatten(x):
    print(x.shape)
    return tf.reshape(x, shape=[tf.shape(x)[0], -1, x.shape[-1]])


def attention(x, ch, scope='attention', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        f = conv2d(x, ch // 8, kernel_size=(1, 1), strides=(1, 1), padding='same',
                   kernel_initializer=kernel_initializer())
        g = conv2d(x, ch // 8, kernel_size=(1, 1), strides=(1, 1), padding='same',
                   kernel_initializer=kernel_initializer())  # [bs, h, w, c']
        h = conv2d(x, ch, kernel_size=(1, 1), strides=(1, 1), padding='same',
                   kernel_initializer=kernel_initializer())  # [bs, h, w, c]

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(
            f), transpose_b=True)  # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable(
            "gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=tf.shape(x))  # [bs, h, w, C]
        x = gamma * o + x

    return x


def full_network(num_classes, training=True):
    _input = tf.placeholder(dtype=tf.float32, shape=[
                            None, 768, 512, 1], name='input_tensor')

    # conv1
    rcl1 = rcl(_input, 8, 3, 'rcl1')  # 768, 512, 8
    pool1 = max_pooling2d(
        rcl1, (2, 2), (2, 2), padding='same', name='pool1')

    # conv2
    rcl2 = rcl(pool1, 16, 3, 'rcl2')  # 384, 256, 16
    #rcl2 = attention(rcl2, 16, 'a1')
    pool2 = max_pooling2d(
        rcl2, (2, 2), (2, 2), padding='same', name='pool2')

    # conv3
    rcl3 = rcl(pool2, 32, 3, 'rcl3')  # 192, 128, 32
    #rcl3 = attention(rcl3, 32, 'a2')
    pool3 = max_pooling2d(
        rcl3, (2, 2), (2, 2), padding='same', name='pool3')

    # conv4
    rcl4 = rcl(pool3, 64, 3, 'rcl4')  # 96, 64, 64
    rcl4 = attention(rcl4, 64, 'a3')
    pool4 = max_pooling2d(
        rcl4, (2, 2), (2, 2), padding='same', name='pool4')

    drop6 = tf.layers.dropout(pool4, rate=1 - KEEP_PROB,
                              training=training)  # 48, 32, 512

    # fc7
    fc7 = residual_layer(drop6, num_classes)  # 48, 32, num_classes

    pool3_res = residual_layer(pool3, num_classes)  # 96, 64, num_classes
    pool2_res = residual_layer(pool2, num_classes)  # 192, 128, num_classes

    # Deconv layers
    deconv8 = deconv2d_x2_layer(fc7, 64)  # 96, 64, num_classes
    deconv8 = attention(deconv8, 64, 'a4')
    sum8 = tf.add(deconv8, pool3_res)  # 96, 64, num_classes

    deconv9 = deconv2d_x2_layer(sum8, num_classes)  # 192, 128, num_classes
    sum9 = tf.add(deconv9, pool2_res)  # 192, 128, num_classes

    sum9 = coordconv_wraper(sum9)
    out = conv2d_transpose(
        inputs=sum9,
        filters=num_classes,
        kernel_size=(8, 8),
        strides=(4, 4),
        padding='same',
        kernel_initializer=kernel_initializer(),
        kernel_regularizer=regularizer())  # 768, 512, num_classes

    return out, _input
