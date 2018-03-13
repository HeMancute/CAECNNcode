import tensorlayer as tl
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorlayer.layers import *
import numpy as np

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 512, 512, 1])
F0 = np.array([[-1, 2, -2, 2, -1],
                       [2, -6, 8, -6, 2],
                       [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2],
                       [-1, 2, -2, 2, -1]], dtype=np.float32)
F0 = F0 / 12.
high_pass_filter = tf.constant_initializer(value=F0, dtype=tf.float32)
net = InputLayer(x, name='inputlayer')
net = Conv2d(net, 1, (5, 5), (1, 1), act=tf.identity,
             padding='SAME', W_init=high_pass_filter, name='HighPass')
y = net.outputs
tl.layers.initialize_global_variables(sess)

img = cv2.imread('cover.pgm',0).astype(np.float32).reshape([1,512,512,1])

img_after = y.eval(feed_dict = {x:img})




if __name__ == '__main__':
#    plt.imshow(img.reshape([256,256]))
#    plt.imshow(img_after.reshape([256,256]))
#    
    pgm_info = np.where(img_after > 10, 1, 0)
    plt.imshow(pgm_info.reshape([512,512]))
    plt.show()

