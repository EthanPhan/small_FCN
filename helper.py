import random
import numpy as np
import os.path
import scipy.misc
import shutil
import time
import tensorflow as tf
from glob import glob
import cv2


def brigthness(image, brigthness):
    table = np.array([i + brigthness for i in np.arange(0, 256)])
    table[table < 0] = 0
    table[table > 255] = 255
    table = table.astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    rows, cols, _ = image.shape
    return cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)


def scale_imge(image, scale):
    rows, cols, _ = image.shape

    h, w = rows / scale, cols / scale
    h, w = int(h), int(w)
    if scale > 1:
        image = image[:h, cols - w: cols, :]
        ret = scipy.misc.imresize(image, (rows, cols))
    else:
        image = np.pad(image, ((0, h - rows), (0, w - cols), (0, 0)),
                       'constant', constant_values=255)
        ret = scipy.misc.imresize(image, (rows, cols))
    return ret


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'imgs', '*.jpg'))
        label_paths = {
            os.path.basename(path): path
            for path in glob(os.path.join(data_folder, 'label', '*.jpg'))
        }
        CLASS0_COLOR = np.array([255, 255, 255])  # Background
        CLASS1_COLOR = np.array([0, 0, 255])
        CLASS2_COLOR = np.array([255, 0, 0])

        random.shuffle(image_paths)
        images = []
        gt_images = []
        for batch_i in range(0, len(image_paths), batch_size):
            for aug in range(20):
                for image_file in image_paths[batch_i:batch_i+batch_size]:
                    gt_image_file = label_paths[os.path.basename(image_file)]

                    image = scipy.misc.imresize(
                        cv2.imread(image_file), image_shape)
                    gt_image = scipy.misc.imresize(
                        cv2.imread(gt_image_file), image_shape)

                    # Random scale
                    scale = random.uniform(0.8, 1.2)
                    image = scale_imge(image, scale)
                    gt_image = scale_imge(gt_image, scale)

                    # Random translate x,y
                    x = random.randint(-50, 50)
                    y = random.randint(-80, 80)
                    image = translate(image, x, y)
                    gt_image = translate(gt_image, x, y)

                    # cv2.imshow('gt_img',gt_image)
                    c0_gt = np.all(gt_image == CLASS0_COLOR, axis=2)
                    c0_gt = c0_gt.reshape(*c0_gt.shape, 1)
                    c1_gt = np.all(np.expand_dims(
                        gt_image[:, :, 2], axis=2) < 128, axis=2)
                    c1_gt = c1_gt.reshape(*c1_gt.shape, 1)
                    c2_gt = np.all(np.expand_dims(
                        gt_image[:, :, 0], axis=2) < 128, axis=2)
                    c2_gt = c2_gt.reshape(*c2_gt.shape, 1)
                    '''
                    c2_img = np.dot(c2_gt, np.array([[0, 255, 0, 127]]))
                    c2_img = scipy.misc.toimage(c2_img, mode="RGBA")
                    cv2.imwrite("c2.jpg", np.array(c2_img))
                    '''
                    gt_image = np.concatenate(
                        (c0_gt, c1_gt, c2_gt), axis=2)

                    # Augmentation
                    image = brigthness(image, random.randint(-100, 50))
                    # cv2.imshow('img', image)
                    # cv2.imshow('gt_image1',gt_image[:,:,1].reshape(-1,image.shape[1],1)*1.0)
                    # cv2.imshow('img_flip',image[:,::-1,:])
                    # cv2.imshow('gt_image_flip1',gt_image[:,::-1,1].reshape(-1,image.shape[1],1)*1.0)
                    # cv2.waitKey(1)

                    # print("image",image.shape,image.dtype,np.min(image),np.max(image))
                    # print("gt_image",gt_image.shape,gt_image.dtype,np.min(gt_image),np.max(gt_image))
                    # convert image to grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    image = np.expand_dims(image, axis=2)
                    images.append(image)
                    gt_images.append(gt_image)

                    #images.append(image[:, ::-1, :])
                    #gt_images.append(gt_image[:, ::-1, :])

                    if len(images) >= batch_size:
                        yield np.array(images), np.array(gt_images)
                        images = []
                        gt_images = []
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(
            image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(
            image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, run_label=""):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, run_label + str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
