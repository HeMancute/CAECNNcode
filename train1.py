#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/4/12 0012 13:50
# @Author  :
# @Software: PyCharm
# ! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/3/13 0013 15:34
# @Author  : jsz
# @Software: PyCharm

import os
import numpy as np
import tensorflow as tf
import input_data
import model

N_CLASSES = 2  # cover与stego
IMG_W = 256  # resize
IMG_H = 256
BATCH_SIZE = 16
CAPACITY = 300
MAX_STEP = 15000  # 一般大于10K
learning_rate = 0.0001  # 一般小于0.0001


def run_training():

    logs_train_dir = 'F://CAE_CNN//log//train1//'
    tfrecords_dir = 'F://CAE_CNN//data//test.tfrecords'

    # 获得batch tfrecord方法
    train_batch, train_label_batch = input_data.read_and_decode(tfrecords_dir, BATCH_SIZE)

    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)

    train_loss = model.losses(train_logits, train_label_batch)
    # 训练
    train_op = model.trainning(train_loss, learning_rate)

    train__acc = model.evaluation(train_logits, train_label_batch)
    # merge到一块？
    summary_op = tf.summary.merge_all()  # 这个是log汇总记录

    # 产生一个会话
    sess = tf.Session()
    # 产生一个writer来写log文件
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    # 产生一个saver来存储训练好的模型
    saver = tf.train.Saver()
    # 所有节点初始化
    sess.run(tf.global_variables_initializer())

    # 队列监控
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # for step in np.arange(MAX_STEP):
    #     _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
    #     # 每隔50步打印一次当前的loss以及acc，同时记录log，写入writer
    #     if step % 2 == 0:
    #         print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
    #         summary_str = sess.run(summary_op)
    #         train_writer.add_summary(summary_str, step)
    #         # 每隔2000步，保存一次训练好的模型
    #     if step % 2000 == 0 or (step + 1) == MAX_STEP:
    #         checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
    #         saver.save(sess, checkpoint_path, global_step=step)
    try:
        # 执行MAX_STEP步的训练，一步一个batch
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
                # 启动以下操作节点，有个疑问，为什么train_logits在这里没有开启？
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
            # 每隔50步打印一次当前的loss以及acc，同时记录log，写入writer
            if step % 2 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
                # 每隔2000步，保存一次训练好的模型
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


run_training()
