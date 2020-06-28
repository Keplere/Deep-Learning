# -*- coding: utf-8 -*-
# @File    : zzhtrain.py
# @Author  : Zhangzihao from APM,CAS
# @Time    : 2020/06/05
# @Desc    :


import numpy as np
# np.set_printoptions(np.inf)
import os
import tensorflow as tf
import sys
workpath = os.getcwd()
sys.path.append((workpath))
from zzhmodel import ResNet18, Inception10
from zzhgetdata import generate_train_data, plot_acc_loss


# 配置GPU
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
# print(gpus, cpus)
# tf.config.experimental.set_virtual_device_configuration(gpus[1],
#                                                         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])


Dataset_Path = 'UCMerced_LandUse/Images'


if os.path.exists('remoteimage.npz'):
    remotedata = np.load('remoteimage.npz')
    x_train = remotedata['x_train']
    y_train = remotedata['y_train']
else:
    x_train, y_train = generate_train_data(Dataset_Path)
    np.savez('remoteimage', x_train=x_train, y_train=y_train)
print(y_train.shape)
print(y_train)
print(max(y_train))
np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)
# print(type(x_train))
# print(type(y_train))
# print(x_train.shape)
# print(y_train.shape)
# print(x_train[1])
# print(y_train)
# print(max(y_train))
x_train = x_train / 255.0


# model = ResNet18([2, 2, 2, 2])
model = Inception10(num_blocks=2, num_classes=21)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/Inception10.ckpt"     # Inception10.ckpt or ResNet18.ckpt
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=60, validation_split=0.2, validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

# print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

# 显示训练集和验证集的acc和loss曲线
plot_acc_loss(history)
