"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: demo.py
"""

import os
import time
import sys
import numpy as np
import tensorflow as tf
from model import DepthTest
from dataflow import image_height, image_width
from utils import check_dir, check_file, get_all_files, del_dir, create_dir, get_filename
from PIL import Image

r_split = 3
c_split = 6
r_h = int(image_height / r_split)
c_w = int(image_width / c_split)


def label2img(label):
    outputs = np.zeros((image_height, image_width, 3))

    for r in range(r_split):
        for c in range(c_split):
            outputs[r * r_h:(r + 1) * r_h, c * c_w:(c + 1) * c_w, :] = np.clip(
                label[r * c_split + c], 0,
                255)

    return outputs.astype(np.uint8)


if __name__ == '__main__':

    model_path = './weights/depth.npy'

    check_file(model_path)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    image = tf.placeholder(tf.float32, [1, image_height, image_width, 3])
    cnn = DepthTest(model_path)
    with tf.name_scope('depth'):
        cnn.build(image)

    images = ['./images/1.png']
    for image_path in images:
        im = Image.open(image_path)
        pix = np.array(im.getdata()).reshape(1, image_height, image_width, 3).astype(np.float32)
        [pr] = sess.run([cnn.predict], feed_dict={image: pix})
        dt = label2img(pr[0])
        pix = pix[0].astype(np.uint8)
        imall = np.hstack((pix, dt))
        img = Image.fromarray(imall, 'RGB')
        img.save('./images/' + get_filename(image_path) + '_prediction.png')
