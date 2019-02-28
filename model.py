import tensorflow as tf
from tensorflow.layers import conv2d, conv2d_transpose, max_pooling2d
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, flatten
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


def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(
            inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network


def Batch_Norm(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(tf.cast(training, dtype=tf.bool),
                       lambda: batch_norm(
                           inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=tf.cast(training, dtype=tf.bool), reuse=True))


def Drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def Relu(x):
    return tf.nn.relu(x)


def Average_pooling(x, pool_size=[2, 2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3, 3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Concat(layers):
    return tf.concat(layers, axis=3)


def Linear(x):
    return tf.layers.dense(inputs=x, units=class_num, name='linear')


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
    return tf.reshape(x, shape=[tf.shape(x)[0], -1, tf.shape(x)[-1]])


def self_attention(x, ch, scope='attention', reuse=False):
    batch_size, height, width, num_channels = tf.shape(
        x)[0], tf.shape(x)[1], tf.shape(x)[2], x.shape[-1]
    with tf.variable_scope(scope, reuse=reuse):
        f = conv2d(x, ch // 8, kernel_size=(1, 1), strides=(1, 1), padding='same',
                   kernel_initializer=kernel_initializer())
        f = max_pooling2d(f, (2, 2), (2, 2), padding='same', name='poolf')

        g = conv2d(x, ch // 8, kernel_size=(1, 1), strides=(1, 1), padding='same',
                   kernel_initializer=kernel_initializer())  # [bs, h, w, c']

        h = conv2d(x, ch // 2, kernel_size=(1, 1), strides=(1, 1), padding='same',
                   kernel_initializer=kernel_initializer())  # [bs, h, w, c]
        h = max_pooling2d(h, (2, 2), (2, 2), padding='same', name='poolh')

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(
            f), transpose_b=True)  # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable(
            "gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=[batch_size, height,
                                 width, ch // 2])  # [bs, h, w, C]
        o = conv2d(x, num_channels, kernel_size=(1, 1), strides=(1, 1), padding='same',
                   kernel_initializer=kernel_initializer())  # [bs, h, w, c]
        x = gamma * o + x

    return x


'''
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

    drop4 = tf.layers.dropout(pool4, rate=1 - KEEP_PROB,
                              training=training)  # 48, 32, 64

    fc5 = conv2d_layer(drop4, 128, 3, 'fc5')

    # # start up path here
    up5 = deconv2d_x2_layer(fc5, 64)  # 96, 64, 64
    up5 = tf.concat([rcl4, up5], -1)  # 96, 64, 128
    up5 = conv2d_layer(up5, 64, 3, 'up5')
    #up5 = attention(up5, 64, 'a4')

    up6 = deconv2d_x2_layer(up5, 32)  # 192, 128, 32
    up6 = tf.concat([rcl3, up6], -1)  # 192, 128, 64
    up6 = conv2d_layer(up6, 32, 3, 'up6')

    wrap7 = coordconv_wraper(up6)
    up7 = conv2d_transpose(
        inputs=wrap7,
        filters=8,
        kernel_size=(8, 8),
        strides=(4, 4),
        padding='same',
        kernel_initializer=kernel_initializer(),
        kernel_regularizer=regularizer())  # 768, 512, 8
    # up7 = tf.concat([rcl1, up7], -1)  # 768, 512, 16
    up7 = conv2d_layer(up7, 8, 3, 'up7')

    out = conv2d_layer(up7, num_classes, 3, 'out')
    return out, _input
'''

# for dense unet here
dropout_rate = 0.2


def bottleneck_layer(x, filters, scope, training=True):
    with tf.name_scope(scope):
        x = Batch_Norm(x, training=training, scope=scope+'_batch1')
        x = Relu(x)
        x = conv_layer(x, filter=4 * filters,
                       kernel=[1, 1], layer_name=scope+'_conv1')
        x = Drop_out(x, rate=dropout_rate, training=training)

        x = Batch_Norm(x, training=training, scope=scope+'_batch2')
        x = Relu(x)
        x = conv_layer(x, filter=filters, kernel=[
                       3, 3], layer_name=scope+'_conv2')
        x = Drop_out(x, rate=dropout_rate, training=training)
        return x


def transition_layer(x, filters, scope, training=True):
    with tf.name_scope(scope):
        x = Batch_Norm(x, training=training, scope=scope+'_batch1')
        x = Relu(x)
        x = conv_layer(x, filter=filters, kernel=[
                       1, 1], layer_name=scope+'_conv1')
        x = Drop_out(x, rate=dropout_rate, training=training)
        x = Average_pooling(x, pool_size=[2, 2], stride=2)
        return x


def dense_block(input_x, filters, nb_layers, layer_name, training=True):
    with tf.name_scope(layer_name):
        layers_concat = list()
        layers_concat.append(input_x)

        x = bottleneck_layer(
            input_x, filters, scope=layer_name + '_bottleN_' + str(0))

        layers_concat.append(x)

        for i in range(nb_layers - 1):
            x = Concat(layers_concat)
            x = bottleneck_layer(
                x, filters, scope=layer_name + '_bottleN_' + str(i + 1))
            layers_concat.append(x)

        x = Concat(layers_concat)

        return x


def full_network(num_classes, filters=12, training=True):
    _input = tf.placeholder(dtype=tf.float32, shape=[
                            None, 768, 512, 1], name='input_tensor')

    conv0 = conv_layer(_input, filter=2 * filters,
                       kernel=[3, 3], stride=1, layer_name='conv0')

    dense1 = dense_block(conv0, filters, 4, 'dense1', training)  # 768, 512, 8
    print('dense1', dense1.shape)
    pool1 = max_pooling2d(
        dense1, (2, 2), (2, 2), padding='same', name='pool1')

    dense2 = dense_block(pool1, filters, 4, 'dense2', training)  # 384, 256, 16
    print('dense2', dense2.shape)
    pool2 = max_pooling2d(
        dense2, (2, 2), (2, 2), padding='same', name='pool2')

    # conv3
    dense3 = dense_block(pool2, filters, 4, 'dense3', training)  # 192, 128, 32
    print('dense3', dense3.shape)
    pool3 = max_pooling2d(
        dense3, (2, 2), (2, 2), padding='same', name='pool3')

    dense4 = dense_block(pool3, filters, 4, 'dense4', training)
    print('dense4', dense4.shape)

    # # start up path here
    up5 = deconv2d_x2_layer(dense4, 64)  # 96, 64, 64
    print('up5', up5.shape)
    up5 = tf.concat([dense3, up5], -1)  # 96, 64, 128
    up5 = conv2d_layer(up5, 64, 3, 'up5')

    up6 = deconv2d_x2_layer(up5, 32)  # 192, 128, 32
    print('up6', up6.shape)
    up6 = tf.concat([dense2, up6], -1)  # 192, 128, 64
    up6 = conv2d_layer(up6, 32, 3, 'up6')

    up7 = conv2d_transpose(
        inputs=up6,
        filters=8,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same',
        kernel_initializer=kernel_initializer(),
        kernel_regularizer=regularizer())  # 768, 512, 8
    # up7 = tf.concat([rcl1, up7], -1)  # 768, 512, 16
    up7 = conv2d_layer(up7, 8, 3, 'up7')
    print('up7', up7.shape)

    out = conv2d_layer(up7, num_classes, 3, 'out')
    return out, _input
