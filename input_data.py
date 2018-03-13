#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/3/12 0012 15:56
# @Author  : jsz
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import os

def get_file(file_dir):

    cover = []
    label_cover = []

    stego = []
    label_stego = []

    1