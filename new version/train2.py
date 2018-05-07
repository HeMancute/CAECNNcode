#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/5/4 0004 10:19
# @Author  : jsz
# @Software: PyCharm

import os
import numpy as np
import tensorflow as tf
import input_data1
import model1

N_CLASSES = 2  # cover与stego
IMG_W = 256  # resize
IMG_H = 256
BATCH_SIZE = 32
CAPACITY = 300
MAX_STEP = 15000  # 一般大于10K
learning_rate = 0.0001  # 一般小于0.0001


def run_training():

    logs_train_dir = 'G:\\dataS-UNIWARD0.4\\logs\\train'
    logs_val_dir = 'G:\\dataS-UNIWARD0.4\\logs\\val'

    tfrecords_traindir = 'G:\\dataS-UNIWARD0.4\\S_UNIWARD0.4train.tfrecords'
    tfrecords_valdir = 'G:\\dataS-UNIWARD0.4\\S_UNIWARD0.4val.tfrecords'

    # 获得batch tfrecord方法
    train_batch, train_label_batch = input_data1.read_and_decode(tfrecords_traindir, BATCH_SIZE)
    val_batch, val_label_batch = input_data1.read_and_decode(tfrecords_valdir, BATCH_SIZE)


    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 256, 256, 1])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])


    logits = model1.inference(x, BATCH_SIZE, N_CLASSES)
    loss = model1.losses(logits, y_)
    acc = model1.evaluation(logits, y_)
    train_op = model1.trainning(loss, learning_rate)


    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break

            tra_images, tra_labels = sess.run([train_batch, train_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                                feed_dict={x: tra_images, y_: tra_labels})
            if step % 2 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                # summary_str = sess.run(summary_op)
                # train_writer.add_summary(summary_str, step)

            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([val_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, acc],
                                              feed_dict={x: val_images, y_: val_labels})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc * 100.0))
                # summary_str = sess.run(summary_op)
                # val_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)


run_training()



