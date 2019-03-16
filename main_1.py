import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import numpy as np
import sys
import cv2
import glob
from scipy import misc
from model import full_network
from data_augment import gen_batch_function, label_2_image, get_image_n_label

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


EPOCHS = 100000
BATCH_SIZE = 1
LR = 0.0003
CLASSES = ['None', "partner_code", "contact_document_number", "issued_date",
           "car_number", "amount_including_tax", "amount_excluding_tax", "item_name"]
IMG_SIZE = (512, 448)


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
    c0_loss_w = 0.017
    c1_loss_w = 4
    c2_loss_w = 1 * c1_loss_w
    # classes_weights = tf.constant([c0_loss_w, c1_loss_w, c2_loss_w])
    classes_weights = tf.constant([0.02, 5, 5, 5, 5, 5, 5, 2])
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
        targets=tf.cast(labels, tf.float32), logits=logits, pos_weight=classes_weights))

    train_op = tf.train.AdamOptimizer(
        learning_rate).minimize(loss, global_step=gl_step)

    return logits, train_op, loss, gl_step


# tests.test_optimize(optimize)


def run_train(sess, training_steps, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
              model_out, correct_label, learning_rate, saver, gl_step):
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
            _, loss, pred = sess.run([train_op, cross_entropy_loss, model_out], feed_dict={
                input_image: image,
                correct_label: image_c,
                learning_rate: LR
            })
            samples_plot.append(sample)
            loss_plot.append(loss)
            iteration += 1
            sample = sample + batch_size
            print("#%4d  (%10d): %.20f" % (iteration, sample, loss))

            if iteration % 100 == 0:
                saver.save(sess, os.path.join('checkpoints',
                                              'fcn'), global_step=gl_step)
                img = label_2_image(image[0], pred[0])
                cv2.imwrite("train_debug.jpg", np.array(img))

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
    num_classes = len(CLASSES)
    image_shape = IMG_SIZE

    train_jsons = glob.glob('data/toyota/train/jsons/*.json')
    train_imgs = glob.glob('data/toyota/train/org_imgs/*.jpg')

    with tf.Session() as sess:
        # Create function to get batches
        get_batches_fn = gen_batch_function(
            train_jsons, train_imgs, image_shape, BATCH_SIZE)

        correct_label = tf.placeholder(tf.int32)
        learning_rate = tf.placeholder(tf.float32)

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
                  cross_entropy_loss, _input, last_layer, correct_label, learning_rate, saver, gl_step)


def infer(input_img, input_json):
    num_classes = len(CLASSES)
    image_shape = IMG_SIZE

    # Load json
    with open(input_json, 'r') as f:
        json_ct = json.load(f)

    # load image
    image = misc.imread(input_img)
    with open('data/toyota/Corpus/corpus.json') as fh:
        corpus = json.load(fh)

    img, gt = get_image_n_label(json_ct, image, corpus)

    with tf.Session() as sess:
        # labels = tf.placeholder(tf.int32)
        logits, _input = full_network(num_classes, training=False)
        logits = tf.reshape(logits, (-1, num_classes))

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        _check_restore_parameters(sess, saver)

        pred = sess.run([tf.nn.softmax(logits)], {
            _input: [img]})
        img = label_2_image(img, pred[0])
        cv2.imwrite("output.jpg", np.array(img))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        img_file = sys.argv[1]
        img = cv2.imread(img_file)
        infer(img)
    else:
        train()
