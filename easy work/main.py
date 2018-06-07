#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/6/7 0007 14:23
# @Author  : jsz
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as io
from scipy.misc import imread, imresize
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def rename():
    count=1
    path = '.\\data\\'

    for dirpath, dirnames, filenames in os.walk(path):
        # for filename in filenames:
        #     print(os.path.join(dirpath, filename))

        for files in filenames:
            # if files.endswith('.pgm'):
                name = files.split('_')
                if name[0] == 'C':
                    pass
                else:
                    Olddir = os.path.join(dirpath, files)
                    if os.path.isdir(Olddir):
                        continue
                    filename = os.path.splitext(files)[0]
                    filetype = os.path.splitext(files)[1]

            #直接改名字的
                    # Newdir = os.path.join(path, 'S' + filetype)

            # 文件名前自动增加S
                    Newdir = os.path.join(dirpath, ('S_'+filename) + filetype)

            # 文件序号一次递增
            #         Newdir = os.path.join(path, str(count) + filetype)



                    # 批量取分隔符（___）前面 / 后面的名称
                    # if filename.find('---')>=0:#如果文件名中含有---
                    #
                    # Newdir=os.path.join(direc,filename.split('---')[0]+filetype);
                    #
                    # #取---前面的字符，若需要取后面的字符则使用filename.split('---')[1]
                    #
                    # if not os.path.isfile(Newdir):

                    os.rename(Olddir, Newdir)

                    count+= 1

def get_file(file_dir):
    cover = []
    label_cover = []
    stego = []
    label_stego = []
    # 打标签
    for file in os.listdir(file_dir):
        # if  file.endswith('0') or file.startswith('.'):
        #     continue  # Skip!
        name = file.split('_')
        if name[0] == 'C':
            cover.append(file_dir + file)
            label_cover.append(0)
        if name[0] == 'S':
            stego.append(file_dir + file)
            label_stego.append(1)
    print("这里有 %d cover \n这里有 %d stego"
          % (len(cover), len(stego)))
    # 打乱文件顺序shuffle
    image_list = np.hstack((cover, stego))
    label_list = np.hstack((label_cover, label_stego))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecord(images, labels, save_dir, name):
    '''convert all images and labels to one tfrecord file.
    Args:
        images: list of image directories, string type
        labels: list of labels, int type
        save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/'
        name: the name of tfrecord file, string type, e.g.: 'train'
    Return:
        no return
    Note:
        converting needs some time, be patient...
    '''

    filename = os.path.join(save_dir, name + '.tfrecords')
    n_samples = len(labels)

    if np.shape(images)[0] != n_samples:
        raise ValueError('Images size %d does not match label size %d.' % (images.shape[0], n_samples))

    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    for i in np.arange(0, n_samples):
        try:
            #            image = imread(image[i])

            image = io.imread(images[i])  # type(image) must be array!
            image = imresize(image, (256, 256))
            image_raw = image.tostring()
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': int64_feature(label),
                'image_raw': bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' % e)
            print('Skip it!\n')
    writer.close()
    print('Transform done!')


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


    image = tf.reshape(image, [256, 256, 1])
    # image = tf.reshape(image, [256, 256]) #for plot
    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      num_threads=64,
                                                      capacity=2000,
                                                      min_after_dequeue=20)
    return image_batch, tf.reshape(label_batch, [batch_size])


def plot_images(images, labels):
    '''plot one batch size
    '''
    for i in np.arange(0, BATCH_SIZE):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        #        plt.title(chr(ord('D') + labels[i] - 1), fontsize=14)

        if labels[i] == 1:
            plt.title(str('Stego'), fontsize=14)
        else:
            plt.title(str('Cover'), fontsize=14)

        plt.subplots_adjust(top=1.5)
        plt.imshow(images[i])
    plt.show()

#####
# for test tfrecords
#####
# BATCH_SIZE = 25
# BATCH_SIZE1 = 25
# image_batch, label_batch = read_and_decode(tfrecords_file, batch_size=BATCH_SIZE)
# image_batch1, label_batch1 = read_and_decode(tfrecords_file1, batch_size=BATCH_SIZE1)
#
# with tf.Session()  as sess:
#     i = 0
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     try:
#         while not coord.should_stop() and i < 1:
#             # just plot one batch size
#             image, label = sess.run([image_batch, label_batch])
#             plot_images(image, label)
#
#             image, label = sess.run([image_batch, label_batch])
#             plot_images(image, label)
#
#             image, label = sess.run([image_batch1, label_batch1])
#             plot_images(image, label)
#
#             image, label = sess.run([image_batch1, label_batch1])
#             plot_images(image, label)
#
#             i += 1
#
#     except tf.errors.OutOfRangeError:
#         print('done!')
#     finally:
#         coord.request_stop()
#     coord.join(threads)



# model

def inference(images, batch_size, n_classes):
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  #kernel size, kernel size, channels, kernel number
                                  shape=[3, 3, 1, 32],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[32],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

#    with tf.variable_scope('pooling1_lrn') as scope:
#        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
#        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')


    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 32, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

        # pool2 and norm2
#    with tf.variable_scope('pooling2_lrn') as scope:
#        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
#        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pooling2')

    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(conv2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[256, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

        # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[256, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    return softmax_linear

#logits 是inference的返回值，labels是ground truth
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy


#training set
N_CLASSES = 2  # cover与stego
IMG_W = 256  # resize
IMG_H = 256
BATCH_SIZE = 16
CAPACITY = 300
MAX_STEP = 250000  # 一般大于10K
learning_rate = 0.00001  # 一般小于0.0001

def run_training(tfrecords_file,tfrecords_file1):

    logs_train_dir = '.\\logs\\train'
    # logs_val_dir = 'H:\\dataWOW_0.05random\\logs\\val'

    tfrecords_traindir = tfrecords_file
    tfrecords_valdir = tfrecords_file1

    # 获得batch tfrecord方法
    train_batch, train_label_batch = read_and_decode(tfrecords_traindir, BATCH_SIZE)
    val_batch, val_label_batch = read_and_decode(tfrecords_valdir, BATCH_SIZE)


    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 256, 256, 1])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])


    logits = inference(x, BATCH_SIZE, N_CLASSES)
    loss = losses(logits, y_)
    acc = evaluation(logits, y_)
    train_op = trainning(loss, learning_rate)


    sess = tf.Session()

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):

            if coord.should_stop():
                break

            tra_images, tra_labels = sess.run([train_batch, train_label_batch])


            _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                                feed_dict={
                                                    x: tra_images,
                                                    y_: tra_labels})
            if step % 2 == 0:
                print(tfrecords_traindir)
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))



            if step % 4 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([val_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, acc],
                                              feed_dict={
                                                  x: val_images,
                                                  y_: val_labels})
                print(tfrecords_valdir)
                print(' **  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc * 100.0))


            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)

