import os
import tensorflow as tf
import numpy as np
import re
import cv2


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def load_image(img, image_size, do_prewhiten=True):
    images = np.zeros((1, image_size, image_size, 3))
    img = cv2.resize(img, (image_size, image_size))
    if img.ndim == 2:
        img = to_rgb(img)
    if do_prewhiten:
        img = prewhiten(img)
    images[:, :, :, :] = img
    return images


def load_model(model_path, sess):
    meta_file, ckpt_file = get_model_filenames(model_path)
    saver = tf.train.import_meta_graph(os.path.join(model_path, meta_file))
    saver.restore(sess, os.path.join(model_path, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_file = [s for s in files if s.endswith('.meta')][0]
    ckpt_files = [s for s in files if '.ckpt' in s]
    ckpt_file = ckpt_files[0]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file
