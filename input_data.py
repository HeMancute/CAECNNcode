#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/3/12 0012 15:56
# @Author  : jsz
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import os

def get_files(file_dir):

    cover = []
    label_cover = []
    stego = []
    label_stego = []

    for file in  os.listdir(file_dir):
        name = file.split('_')
        if name[0] == '0':
            cover.append(file_dir + file)
            label_cover.append(0)
        else:
            stego.append(file_dir + file)
            label_stego.append(1)
    print("这里有 %d cover \n这里有 %d stego"
          % (len(cover), len(stego)))

    #打乱文件顺序
    image_list = np.hstack((cover,stego))
    label_list = np.hstack((label_cover, label_stego))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list , label_list






def get_batch(image, label,
              image_W, image_H,
              batch_size, capacity):

    #将python.list 转换成tf能够识别的格式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])


    image = tf.image.decode_png(image_contents, channels=3)
    image = tf.cast(image, tf.float32)

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)



    image_batch, label_batch = tf.train.batch([image,label],
        batch_size = batch_size,
        num_threads=64,
        capacity=capacity)


    label_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, label_batch



# file_dir = 'F://CAE_CNN//data//train_imgs//'
#
#
# import matplotlib.pyplot as plt
#
# BATCH_SIZE = 2
# CAPACITY = 256
# IMG_W = 256
# IMG_H = 256
#
#
# image_list, label_list = get_files(file_dir)
# image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

# with tf.Session() as sess:
#     i = 0
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     try:
#         while not coord.should_stop() and i < 5:
#             img, label = sess.run([image_batch, label_batch])
#
#             for j in np.arange(BATCH_SIZE):
#                 print("label: %d" % label[j])
#                 plt.imshow(img[j, : , : , :])
#                 plt.show()
#             i += 1
#     except tf.errors.OutOfRangeError:
#         print("done!")
#     finally:
#         coord.request_stop()
#     coord.join(threads)

