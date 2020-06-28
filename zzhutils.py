# -*- coding: utf-8 -*-
# @File    : zzhutils.py
# @Author  : Zhangzihao from APM,CAS
# @Time    : 2020/06/05
# @Desc    :


import os
import numpy as np
from cv2 import imread
import matplotlib.pyplot as plt


def generate_train_data(datafolder):
    # 生成训练数据和标签数据
    x = []; y = []
    class_names = os.listdir((datafolder))
    # print(classes_names)
    u = 0
    for class_name in class_names:
        image_path = os.path.join(datafolder, class_name)
        image_names = os.listdir(image_path)
        for image_name in image_names:
            image = imread(os.path.join(image_path, image_name))
            # print(image)
            image_arr = np.zeros((256, 256, 3))
            if image.shape[0] <= 256 and image.shape[1] <= 256:     # 图片尺寸小于256*256的，用0填充
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        for k in range(image.shape[2]):
                           image_arr[i, j, k] = image[i, j, k]
            else:                                                   # 图片尺寸大于256*256的，截取(这部分很少，不影响结果)
                for i in range(256):
                    for j in range(256):
                        for k in range(3):
                           image_arr[i, j, k] = image[i, j, k]
            x.append(image_arr); y.append(u)
        u += 1
    return np.array(x), np.array(y)

def plot_acc_loss(history):
    # 生成当前训练的acc和loss曲线
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('acc_loss.png')
    plt.show()
