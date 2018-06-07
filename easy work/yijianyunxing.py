#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/6/7 0007 14:51
# @Author  : jsz
# @Software: PyCharm

import main

main.rename()

name_test = 'testrandomtrain'
name_test1 = 'testrandomval'
tfrecords_file = '.\\testrandomtrain.tfrecords'
tfrecords_file1 = '.\\testrandomval.tfrecords'

test_dir = '.\\data\\train\\'
save_dir = '.\\'
test_dir1 = '.\\data\\val\\'
save_dir1= '.\\'

images, labels = main.get_file(test_dir)
main.convert_to_tfrecord(images, labels, save_dir, name_test)
images1, labels1 = main.get_file(test_dir1)
main.convert_to_tfrecord(images1, labels1, save_dir1, name_test1)

main.run_training(tfrecords_file,tfrecords_file1)