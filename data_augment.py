#!venv/bin/python
# -*- coding: utf-8 -*-

"""
Filename: data_augment.py
to randomly merge pairs of key and value together to augment data for toyota project
"""


__author__ = 'Ethan, Andy'


import glob
import json
import random
import copy
import numpy as np
import os
from scipy import ndimage
from scipy import misc


all_class = ["partner_code", "contact_document_number", "issued_date",
             "car_number", "amount_including_tax", "amount_excluding_tax", "item_name"]


IMG_SIZE = (512, 448)
TEST_RATE = 0.3


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def get_v_intersec(loc1, loc2):
    x1_1, y1_1, w1, h1 = loc1
    x2_1, y2_1, w2, h2 = loc2
    y1_2 = y1_1 + h1
    y2_2 = y2_1 + h2
    ret = max(0, min(y1_2 - y2_1, y2_2 - y1_1))
    return ret


def get_v_union(loc1, loc2):
    x1_1, y1_1, w1, h1 = loc1
    x2_1, y2_1, w2, h2 = loc2
    y1_2 = y1_1 + h1
    y2_2 = y2_1 + h2
    ret = min(h1 + h2, max(y2_2 - y1_1, y1_2 - y2_1))
    return ret


def merge_loc(loc1, loc2):
    x1_1, y1_1, w1, h1 = loc1
    x2_1, y2_1, w2, h2 = loc2
    y1_2 = y1_1 + h1
    y2_2 = y2_1 + h2
    x1_2 = x1_1 + w1
    x2_2 = x2_1 + w2
    x1 = min(x1_1, x2_1)
    y1 = min(y1_1, y2_1)
    x2 = max(x1_2, x2_2)
    y2 = max(y1_2, y2_2)
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]


def _verify_close_pair(loc1, loc2):
    if loc1[0] > loc2[0]:
        return False
    inters = get_v_intersec(loc1, loc2)
    union = get_v_union(loc1, loc2)
    if inters < 0.6 * union:
        return False

    dt = loc2[0] - loc1[0] - loc1[2]
    if dt > 0.4 * min(loc1[2], loc1[2]):
        return False

    return True


def _get_random_money(text):
    if random.random() > 0.1:
        return text
    r_value = random.randint(1000, 1000000)
    r_text = str(r_value)
    ret = r_text[:-3] + ',' + r_text[-3:]
    return ret


def random_merged_key_value(data):
    ret = {}
    # find all pairs of key and value that close to each other
    pairs = []
    for k, it in data.items():
        k_type = it.get('type')
        if k_type != 'key':
            continue

        label = it.get('label')
        k_loc = it.get('location')
        # find value line next to this
        for v_k, v_it in data.items():
            v_type = v_it.get('type')
            if v_type != 'value':
                continue
            v_label = v_it.get('label')
            if v_label != label:
                continue
            v_loc = v_it.get('location')
            if _verify_close_pair(k_loc, v_loc):
                pairs.append([k, v_k])

    # pick 50% pair to merge at a time
    merge_pairs = random.sample(pairs, int(0.5 * len(pairs)))
    merge_lines = []
    [merge_lines.extend(p) for p in merge_pairs]

    # now merge them
    for k, it in data.items():
        k_type = it.get('type')

        label = it.get('label')
        loc = it.get('location')
        text = it.get('value')

        if k_type != 'value':
            label = 'None'

        if label == 'amount_excluding_tax':
            text = _get_random_money(text)
        if k not in merge_lines:
            ret[k] = {'value': text, 'location': loc, 'label': label}

    for p in merge_pairs:
        k_l, v_l = p
        label = data[v_l].get('label')
        loc = merge_loc(data[k_l].get('location'), data[v_l].get('location'))
        text = data[k_l].get('value') + data[v_l].get('value')
        ret[k_l] = {'value': text, 'location': loc, 'label': label}

    return ret


@static_vars(st_ametx_keys=[], st_amitx_keys=[])
def parse(file_name, data):
    temp = {}
    ca_regions = data['_via_img_metadata']
    _, ca_regions = list(ca_regions.items())[0]
    ca_regions = ca_regions['regions']
    for i, it in enumerate(ca_regions):
        attr = it.get('region_attributes')
        text = attr.get('label')
        shape_attr = it.get('shape_attributes')
        try:
            x, y, w, h = shape_attr['x'], shape_attr['y'], shape_attr['width'], shape_attr['height']
        except:
            continue
        loc = [x, y, w, h]
        label = 'None'
        fm_key = attr.get('formal_key')
        if fm_key.startswith('table'):
            fm_key = fm_key.replace('table_total', 'amount')
        if fm_key != '' and fm_key in all_class:
            label = fm_key
        else:
            label = "None"

        t_type = attr.get('type')
        line_name = 'text_line' + str(i)

        temp[line_name] = {'value': text, 'location': loc,
                           'label': label, 'type': t_type}

    temp['name'] = file_name
    return temp


def add_noise_to_text(cell_list, corpus):
    cell_list_clone = []
    for cell_dict in cell_list:
        clone = copy.deepcopy(cell_dict)
        clone['name'] = clone['name'] + '_aug'
        for k, v in clone.items():
            if k == 'name':
                continue
            else:
                try:
                    for _ in range(np.random.randint(0, len(v['value']))):
                        index = np.random.randint(0, len(v['value']))
                        if has_carnumber_format(v['value']):
                            continue
                        v['value'] = v['value'][:index] + \
                            np.random.choice(corpus) + v['value'][index+1:]
                except ValueError:
                    # print('skip add noise for text')
                    pass
        cell_list_clone.append(clone)
    final = cell_list + cell_list_clone
    return final


def split_textline(line):
    ''' get location of each character
    '''
    ret = []
    x, y, w, h = line.get('location')
    text = line.get('value')
    num_char = len(text)
    if num_char == 0:
        return ret
    elif num_char == 1:
        return [(text[0], [x, y, w, h])]
    else:
        estimated_charwidth = h
        # calculate the space between char
        space = max(0, (w - num_char * estimated_charwidth) / (num_char - 1))
        charwidth = int((w - (num_char - 1) * space) / num_char)
        space = int(space)
        current_x = x
        for i, c in enumerate(text):
            char_loc = [current_x, y, charwidth, h]
            current_x = char_loc[0] + charwidth + space
            ret.append((c, char_loc))
    return ret


def resize_json(parsed_json, src_size, dst_size=IMG_SIZE):
    ret = {}

    scale_y = dst_size[0] / src_size[0]
    scale_x = dst_size[1] / src_size[1]

    for k, it in parsed_json.items():
        if not isinstance(it, dict):
            ret[k] = it
            continue
        location = it.get('location')
        x = int(scale_x * location[0])
        w = int(scale_x * location[2])
        y = int(scale_y * location[1])
        h = int(scale_y * location[3])
        it['location'] = [x, y, w, h]
        ret[k] = it
    return ret


CLASSES = ['None', "partner_code", "contact_document_number", "issued_date",
           "car_number", "amount_including_tax", "amount_excluding_tax", "item_name"]


def get_image_n_label(json_file, img_file, corpus, name=''):
    parsed_json = parse(name, json_file)

    resized_json = resize_json(parsed_json, img_file.shape)
    resized_img = misc.imresize(img_file, IMG_SIZE)
    char_vec = np.zeros(
        (IMG_SIZE[0], IMG_SIZE[1], len(corpus)), dtype=np.float)
    label = np.zeros((IMG_SIZE[0], IMG_SIZE[1], len(CLASSES)), dtype=np.float)

    # embed text from json into the image
    for k, it in resized_json.items():
        if not isinstance(it, dict):
            continue
        chars = split_textline(it)
        for c in chars:
            if c[0] not in corpus:
                continue
            x, y, w, h = c[1]
            char_vec[y:y+h, x:x+w, corpus.index(c[0])] = 1
        x, y, w, h = it.get('location')
        lb = it.get('label')
        if lb not in CLASSES:
            continue
        label[y:y+h, x:x+w, CLASSES.index(lb)] = 1

    image = np.concatenate((resized_img, char_vec), axis=-1)
    return image, label


def label_2_image(img, label):
    color_pallet = [[0, 0, 0, 0], [68, 47, 135, 127], [138, 123, 25, 127], [59, 113, 7, 127],
                    [27, 130, 120, 127], [32, 69, 130, 127], [118, 29, 99, 127], [27, 29, 111, 127]]

    img = img[:, :, :3]
    img = misc.toimage(img)
    for i in range(label.shape[-1]):
        mask = label[:, :, i].reshape(
            IMG_SIZE[0], IMG_SIZE[1])
        bi_mask = (mask > 0.5).reshape(
            IMG_SIZE[0], IMG_SIZE[1], 1)
        mask = np.dot(bi_mask, np.array([color_pallet[i]]))
        img_mask = misc.toimage(mask, mode="RGBA")
        img.paste(img_mask, box=None, mask=img_mask)
    return img


def gen_batch_function(json_paths, img_paths, img_size, batch_size, corpus_path='data/toyota/Corpus/corpus.json'):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    with open(corpus_path) as fh:
        corpus = json.load(fh)

    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        # image_paths = glob(os.path.join(data_folder, 'imgs', '*.jpg'))
        _json_paths = {
            os.path.basename(path)[:-5]: path for path in json_paths}
        _img_paths = {
            os.path.basename(path)[:-4]: path for path in img_paths}

        random.shuffle(img_paths)
        images = []
        gt_images = []
        for batch_i in range(0, len(json_paths), batch_size):
            for json_file in json_paths[batch_i:batch_i+batch_size]:
                try:
                    image_file = _img_paths[os.path.basename(json_file)[:-5]]
                except KeyError:
                    print(os.path.basename(json_file)[:-5])
                    continue
                # Load json
                with open(json_file, 'r') as f:
                    json_ct = json.load(f)

                # load image
                image = misc.imread(image_file)
                img, gt = get_image_n_label(json_ct, image, corpus)
                images.append(img)
                gt_images.append(gt)

                if len(images) >= batch_size:
                    yield np.array(images), np.array(gt_images)
                    images = []
                    gt_images = []
    return get_batches_fn


if __name__ == "__main__":
    train_jsons = glob.glob('data/toyota/train/jsons/*.json')
    train_imgs = glob.glob('data/toyota/train/org_imgs/*.jpg')
    test_jsons = glob.glob('data/toyota/test/jsons/*.json')
    test_imgs = glob.glob('data/toyota/test/org_imgs/*.jpg')
    with open('data/toyota/Corpus/corpus.json') as fh:
        corpus = json.load(fh)

    res = []
    for each_file in train_jsons:
        base_name = each_file.split('/')[-1][:-5]
        img_file = [f for f in train_imgs if base_name + '.jpg' in f]
        if img_file:
            img_file = img_file[0]

        # Load json
        with open(each_file, 'r') as f:
            json_file = json.load(f)

        # load image
        image = misc.imread(img_file)
        img, label = get_image_n_label(json_file, image, corpus, base_name)

    for each_file in test_jsons:
        base_name = each_file.split('/')[-1][:-5]
        img_file = [f for f in test_imgs if base_name + '.jpg' in f]
        if img_file:
            img_file = img_file[0]

        # Load json
        with open(each_file, 'r') as f:
            json_file = json.load(f)

        # load image
        image = misc.imread(img_file)
        img, label = get_image_n_label(json_file, image, corpus, base_name)
