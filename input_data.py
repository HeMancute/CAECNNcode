#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/3/12 0012 15:56
# @Author  : jsz
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import os

#定义读取函数，返回两个list,image_list是含有图片路径的string，label_list含有0，1

def get_files(file_dir):
    cover = []
    label_cover = []
    stego = []
    label_stego = []
    #打标签
    for file in  os.listdir(file_dir):
        # if  file.endswith('0') or file.startswith('.'):
        #     continue  # Skip!
        name = file.split('_')
        if name[0] == 'C0':
            cover.append(file_dir + file)
            label_cover.append(0)
        if name[0] == 'S1' :
            stego.append(file_dir + file)
            label_stego.append(1)
    print("这里有 %d cover \n这里有 %d stego"
          % (len(cover), len(stego)))
    #打乱文件顺序shuffle
    image_list = np.hstack((cover,stego))
    label_list = np.hstack((label_cover, label_stego))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list , label_list

#定义batch函数
def get_batch(image, label,
              image_W, image_H,
              batch_size, capacity):

    # #将python.list 转换成tf能够识别的格式

    label = tf.cast(label, tf.int32)
    image = tf.cast(image, tf.string)

    input_queue = tf.train.slice_input_producer([image, label])


    label = input_queue[1]

    image_contents = tf.read_file(input_queue[0])

    print(input_queue[0])

    image = tf.image.decode_png(image_contents, channels=0)

    image = tf.reshape(image, [ 256, 256, 1])
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
        batch_size = batch_size,
        num_threads=64,
        capacity=capacity,
                                    )

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch

def read_and_decode(tfrecords_file, batch_size):
    '''read and decode tfrecord file, generate (image, label) batches
    Args:
        tfrecords_file: the directory of tfrecord file
        batch_size: number of images in each batch
    Returns:
        image: 4D tensor - [batch_size, width, height, channel]
        label: 1D tensor - [batch_size]
    '''
    # make an input queue from the tfrecord file
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)

    ##########################################################
    # you can put data augmentation here, I didn't use it
    ##########################################################
    # all the images of notMNIST are 28*28, you need to change the image size if you use other dataset.

    image = tf.reshape(image, [512, 512, 1])
    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=2000)

    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, tf.reshape(label_batch, [batch_size])

# file_dir = 'F://CAE_CNN//data//pgm_coverstego//'
# file_dir = 'F://CAE_CNN//data//train//'
# get_files(file_dir)
# file_dir = 'G://PGMtoPNG//train_imgs//'
#
# import matplotlib.pyplot as plt
#
# BATCH_SIZE = 2
# CAPACITY = 256
# IMG_W = 256
# IMG_H = 256
#
# image_list, label_list = get_files(file_dir)
# image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
# with tf.Session() as sess:
#     i = 0
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     try:
#         while not coord.should_stop() and i < 2:
#             img, label = sess.run([image_batch, label_batch])
#
#             for j in np.arange(BATCH_SIZE):
#                 print("label: %d" % label[j])
#
#                 plt.imshow(img[j])
#                 # plt.imshow('F://CAE_CNN//data//pgm_cover//Cover.1.pgm')
#
#                 plt.show()
#                 # print(img.eval())
#             i += 1
#     except tf.errors.OutOfRangeError:
#         print("done!")
#     finally:
#         coord.request_stop()
#     coord.join(threads)
#

