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


def kernel_initializer():
    # return tf.contrib.layers.variance_scaling_initializer()
    return tf.contrib.layers.xavier_initializer()


def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(
            inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME',
            kernel_initializer=kernel_initializer())
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


def gated_attention(tensor, gate, n_filters, kernel_size=[1, 1]):
    g1 = conv_layer(gate, n_filters, kernel=kernel_size)
    g1 = tf.image.resize_bilinear(g1, [tf.shape(tensor)[1], tf.shape(tensor)[2]])
    x1 = conv_layer(tensor, n_filters, kernel=kernel_size)
    net = tf.add(g1, x1)
    net = tf.nn.relu(net)
    net = conv_layer(net, n_filters, kernel=kernel_size)
    net = tf.nn.sigmoid(net)
    # net = tf.concat([att_tensor, net], axis=-1)
    net = net * tensor
    return net


def hw_flatten(x):
    return tf.reshape(x, shape=[tf.shape(x)[0], -1, tf.shape(x)[-1]])


def self_attention(x, ch, scope='attention', reuse=False):
    batch_size, height, width, num_channels = tf.shape(
        x)[0], tf.shape(x)[1], tf.shape(x)[2], x.shape[-1]
    with tf.variable_scope(scope, reuse=reuse):
        f = conv_layer(x, ch // 8, kernel=1)
        f = max_pooling2d(f, (2, 2), (2, 2), padding='same', name='poolf')

        g = conv_layer(x, ch // 8, kernel=1)  # [bs, h, w, c']

        h = conv_layer(x, ch // 2, kernel=1)  # [bs, h, w, c]
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
        o = conv_layer(x, num_channels, kernel=1)  # [bs, h, w, c]
        x = gamma * o + x

    return x


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


def dense_layer(x, filters, scope, training=True):
    with tf.name_scope(scope):
        x = Batch_Norm(x, training=training, scope=scope+'_batch1')
        x = Relu(x)
        x = conv_layer(x, filter=filters, kernel=[
                       3, 3], layer_name=scope+'_conv1')
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

        x = dense_layer(
            input_x, filters, scope=layer_name + '_bottleN_' + str(0))

        layers_concat.append(x)

        for i in range(nb_layers - 1):
            x = Concat(layers_concat)
            x = dense_layer(
                x, filters, scope=layer_name + '_bottleN_' + str(i + 1))
            layers_concat.append(x)

        x = Concat(layers_concat)

        return x


def transition_down(x, filters, layer_name, training=True):
    with tf.name_scope(layer_name):
        x = Batch_Norm(x, training=training, scope=layer_name + '_bn')
        x = Relu(x)
        x = conv_layer(x, filter=filters, kernel=[
                       1, 1], layer_name=layer_name + '_conv1')
        x = Drop_out(x, rate=dropout_rate, training=training)
        x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1],
                           padding='SAME', name=layer_name+'_maxpool2x2')
    return x


def transition_up(x, filters, layer_name, training=True):
    with tf.name_scope(layer_name):
        x = tf.layers.conv2d_transpose(x,
                                       filters=filters,
                                       kernel_size=[3, 3],
                                       strides=[2, 2],
                                       padding='SAME',
                                       activation=None,
                                       kernel_initializer=kernel_initializer(),
                                       name=layer_name+'_trans_conv3x3')

        return x


def full_network(num_classes, filters=8, training=True):
    # # Down path
    _input = tf.placeholder(dtype=tf.float32, shape=[
                            None, 512, 448, 1], name='input_tensor')

    conv0 = conv_layer(_input, filter=filters,
                       kernel=[3, 3], stride=1, layer_name='conv0')

    # 768, 512, 20 (filters * (nb_layers + 2))
    dense1 = dense_block(conv0, filters, 2, 'dense1', training)
    pool1 = transition_down(dense1, dense1.get_shape()
                            [-1], 'down1', training)  # 384, 256, 20

    dense2 = dense_block(pool1, 2 * filters, 2, 'dense2',
                         training)  # 384 x 256 x 24
    pool2 = transition_down(dense2, dense2.get_shape()
                            [-1], 'down2', training)  # 192 x 128 x 24

    # conv3
    dense3 = dense_block(pool2, 3 * filters, 2, 'dense3',
                         training)  # 192 x 128 x 28
    dense3 = self_attention(dense3, dense3.get_shape()[-1], scope='attention1')
    pool3 = transition_down(dense3, dense3.get_shape()
                            [-1], 'down3', training)  # 96 x 64 x 28

    dense4 = dense_block(pool3, 4 * filters, 2, 'dense4', training)  # 96 x 64 x 32
    dense4 = self_attention(dense4, dense4.get_shape()[-1], scope='attention2')

    # # Up path
    up5 = transition_up(dense4, dense3.get_shape()
                        [-1], 'up5', training)  # 192 x 128 x 28
    up5 = self_attention(up5, up5.get_shape()[-1], scope='attention3')
    g_dense3 = gated_attention(dense3, dense4, dense3.get_shape()[-1])
    up5 = tf.concat([dense3, up5], -1)  # 192 x 128 x 56
    dense5 = dense_block(up5, 3 * filters, 2, 'dense5', training)  # 192 x 128 x 28

    up6 = transition_up(dense5, dense2.get_shape()
                        [-1], 'up6', training)  # 384 x 256 x 24
    g_dense2 = gated_attention(dense2, dense5, dense2.get_shape()[-1])
    up6 = tf.concat([dense2, up6], -1)  # 384 x 256 x 48
    dense6 = dense_block(up6, 2 * filters, 2, 'dense6', training)  # 384 x 256 x 24

    up7 = transition_up(dense6, dense1.get_shape()
                        [-1], 'up7', training)  # 768 x 512 x 20
    g_dense1 = gated_attention(dense1, dense6, dense1.get_shape()[-1])
    up7 = tf.concat([dense1, up7], -1)  # 768 x 512 x 40
    dense7 = dense_block(up7, filters, 2, 'dense7', training)  # 768 x 512 x 20
    up7 = conv_layer(dense7, 8, kernel=1)

    out = tf.layers.conv2d(up7,
                           filters=num_classes,
                           kernel_size=[1, 1],
                           strides=[1, 1],
                           padding='SAME',
                           dilation_rate=[1, 1],
                           activation=None,
                           kernel_initializer=kernel_initializer(),
                           name='last_conv1x1')
    return out, _input


def cu_net(num_classes, filters=6, training=True):
    # # Down path
    _input = tf.placeholder(dtype=tf.float32, shape=[
                            None, 512, 448, 1], name='input_tensor')

    conv0 = conv_layer(_input, filter=2 * filters,
                       kernel=[3, 3], stride=1, layer_name='conv0')

    # 768, 512, 20 (filters * (nb_layers + 2))
    dense1 = dense_block(conv0, filters, 2, 'dense1', training)
    pool1 = transition_down(dense1, dense1.get_shape()
                            [-1], 'down1', training)  # 384, 256, 20

    dense2 = dense_block(pool1, filters, 3, 'dense2',
                         training)  # 384 x 256 x 24
    pool2 = transition_down(dense2, dense2.get_shape()
                            [-1], 'down2', training)  # 192 x 128 x 24

    # conv3
    dense3 = dense_block(pool2, filters, 4, 'dense3',
                         training)  # 192 x 128 x 28
    dense3 = self_attention(dense3, dense3.get_shape()[-1], scope='attention1')
    pool3 = transition_down(dense3, dense3.get_shape()
                            [-1], 'down3', training)  # 96 x 64 x 28

    dense4 = dense_block(pool3, filters, 5, 'dense4', training)  # 96 x 64 x 32
    dense4 = self_attention(dense4, dense4.get_shape()[-1], scope='attention2')

    # # Up path
    up5 = transition_up(dense4, dense3.get_shape()
                        [-1], 'up5', training)  # 192 x 128 x 28
    up5 = self_attention(up5, up5.get_shape()[-1], scope='attention3')
    g_dense3 = gated_attention(dense3, dense4, dense3.get_shape()[-1])
    up5 = tf.concat([g_dense3, up5], -1)  # 192 x 128 x 56
    dense5 = dense_block(up5, filters, 4, 'dense5', training)  # 192 x 128 x 28

    up6 = transition_up(dense5, dense2.get_shape()
                        [-1], 'up6', training)  # 384 x 256 x 24
    g_dense2 = gated_attention(dense2, dense5, dense2.get_shape()[-1])
    up6 = tf.concat([g_dense2, up6], -1)  # 384 x 256 x 48
    dense6 = dense_block(up6, filters, 3, 'dense6', training)  # 384 x 256 x 24

    up7 = transition_up(dense6, dense1.get_shape()
                        [-1], 'up7', training)  # 768 x 512 x 20
    g_dense1 = gated_attention(dense1, dense6, dense1.get_shape()[-1])
    up7 = tf.concat([g_dense1, up7], -1)  # 768 x 512 x 40
    dense7 = dense_block(up7, filters, 2, 'dense7', training)  # 768 x 512 x 20

    # # Output path
    out = conv2d_layer(dense7, 8, 3, 'up7_1')
    out = tf.layers.conv2d(out,
                           filters=num_classes,
                           kernel_size=[1, 1],
                           strides=[1, 1],
                           padding='SAME',
                           dilation_rate=[1, 1],
                           activation=None,
                           kernel_initializer=kernel_initializer(),
                           name='last_conv1x1')
    return out, _input
