import os.path
import tensorflow as tf
import helper
import warnings
import numpy as np
import sys
import cv2
import scipy

from distutils.version import LooseVersion
from model import full_network

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
EPOCHS = 100000
BATCH_SIZE = 4
LR = 0.00003
CLASS_LOSS_WEIGHTS = [0.0949784, 5, 5]


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    gl_step = tf.Variable(
        0, dtype=tf.int32, trainable=False, name='global_step')
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    #    logits=logits, labels=labels))

    # Because the data is imbalance (the number of pixel with label
    # '0' is many time more than the number of pixel with label '1'
    # use pos weight to avoid lazy learning - always predict '0' )
    classes_weights = tf.constant(CLASS_LOSS_WEIGHTS)
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
        targets=tf.cast(labels, tf.float32), logits=logits,
        pos_weight=classes_weights))

    train_op = tf.train.AdamOptimizer(
        learning_rate).minimize(loss, global_step=gl_step)

    return logits, train_op, loss, gl_step


# tests.test_optimize(optimize)


def run_train(sess, training_steps, batch_size, get_batches_fn, train_op,
              cross_entropy_loss, input_image, correct_label,
              learning_rate, saver, gl_step):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    loss = 99999
    samples_plot = []
    loss_plot = []
    sample = 0
    iteration = gl_step.eval()
    epoch = 0
    while True:
        for image, image_c in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={
                input_image: image,
                correct_label: image_c,
                learning_rate: LR
            })
            samples_plot.append(sample)
            loss_plot.append(loss)
            iteration += 1
            sample = sample + batch_size
            print("#%4d  (%10d): %.20f" % (iteration, sample, loss))

            if iteration % 10 == 0:
                saver.save(sess, os.path.join('checkpoints',
                                              'fcn'), global_step=gl_step)
        print("%4d Loss: %f" % (epoch, loss))
        epoch += 1


def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(
        os.path.dirname('checkpoints' + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the model")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the model")


def train():
    global EPOCHS, KEEP_PROB, BATCH_SIZE
    if len(sys.argv) > 1:
        EPOCHS = int(sys.argv[1])
    if len(sys.argv) > 2:
        BATCH_SIZE = int(sys.argv[2])
    if len(sys.argv) > 3:
        KEEP_PROB = float(sys.argv[3])
    num_classes = 3
    image_shape = (768, 512)
    data_dir = './data'
    # tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    # helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        # vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(
            os.path.join(data_dir, 'data_date/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        correct_label = tf.placeholder(tf.int32)
        learning_rate = tf.placeholder(tf.float32)

        # Build NN using load_vgg, layers, and optimize function
        last_layer, _input = full_network(num_classes)
        logits, train_op, cross_entropy_loss, gl_step = optimize(
            last_layer, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        _check_restore_parameters(sess, saver)

        # train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op,
        #          cross_entropy_loss, _input, correct_label, keep_prob, learning_rate, saver, gl_step)
        run_train(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op,
                  cross_entropy_loss, _input, correct_label, learning_rate, saver, gl_step)

        # Save the variables to disk.
        save_path = saver.save(
            sess, "./runs/model_E%04d-B%04d-K%f.ckpt" % (EPOCHS, BATCH_SIZE, KEEP_PROB))
        print("Model saved in file: %s" % save_path)


def infer(input_img):
    num_classes = 3
    image_shape = (768, 512)
    data_dir = './data'
    with tf.Session() as sess:
        get_batches_fn = helper.gen_batch_function(
            os.path.join(data_dir, 'data_date/training'), image_shape)
        # for image, image_c in get_batches_fn(1):
        image = scipy.misc.imresize(input_img, image_shape)
        labels = tf.placeholder(tf.int32)
        logits, _input = full_network(num_classes, training=False)
        logits = tf.reshape(logits, (-1, num_classes))

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        _check_restore_parameters(sess, saver)

        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img = np.expand_dims(img, axis=2)
        im_softmax = sess.run([tf.nn.softmax(logits)], {
            _input: [img]})

        c1_softmax = im_softmax[0][:, 1].reshape(
            image_shape[0], image_shape[1])
        c2_softmax = im_softmax[0][:, 2].reshape(
            image_shape[0], image_shape[1])

        segmentation_1 = (c1_softmax > 0.5).reshape(
            image_shape[0], image_shape[1], 1)
        segmentation_2 = (c2_softmax > 0.5).reshape(
            image_shape[0], image_shape[1], 1)
        mask_1 = np.dot(segmentation_1, np.array([[255, 0, 0, 127]]))
        mask_2 = np.dot(segmentation_2, np.array([[0, 0, 255, 127]]))
        mask_full_1 = scipy.misc.imresize(mask_1, input_img.shape)
        mask_full_2 = scipy.misc.imresize(mask_2, input_img.shape)
        mask_full_1 = scipy.misc.toimage(mask_full_1, mode="RGBA")
        mask_full_2 = scipy.misc.toimage(mask_full_2, mode="RGBA")
        mask_1 = scipy.misc.toimage(mask_1, mode="RGBA")
        mask_2 = scipy.misc.toimage(mask_2, mode="RGBA")

        # street_im = scipy.misc.toimage(image)
        # street_im.paste(mask, box=None, mask=mask)

        im_full = scipy.misc.toimage(input_img)
        im_full.paste(mask_full_1, box=None, mask=mask_full_1)
        im_full.paste(mask_full_2, box=None, mask=mask_full_2)

        cv2.imwrite("output.jpg", np.array(im_full))
        cv2.imwrite("mask_1.jpg", np.array(mask_full_1))
        cv2.imwrite("mask_2.jpg", np.array(mask_full_2))
        exit(0)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        img_file = sys.argv[1]
        img = cv2.imread(img_file)
        infer(img)
    else:
        train()
