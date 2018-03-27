#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/3/12 0012 16:17
# @Author  : jsz
# @Software: PyCharm
import os


def rename():
    count=1
    path = 'F:\CAE_CNN\data\lldata'
    filelist = os.listdir(path)
    for files in filelist:
        Olddir = os.path.join(path, files)
        if os.path.isdir(Olddir):
            continue
        filename = os.path.splitext(files)[0]
        filetype = os.path.splitext(files)[1]

#直接改名字的
        # Newdir = os.path.join(path, 'S' + filetype)

# 文件名前自动增加S
#         Newdir = os.path.join(path, ('Stego.'+filename) + filetype)

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

rename()
