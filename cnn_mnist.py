"""
Created on Sun Mar  15 10:48:36 2020

@author: Zhang Zihao
@organization: UCAS;WHIGG (server time:201806-202106)
Ackonwledge:(1)https://www.icourse163.org/course/PKU-1002536002; (2)https://www.bilibili.com/video/av95051759
"""
import tensorflow as tf

mnist = tf.keras.datasets.mnist    # 数据集自动加载

(x_train, y_train), (x_test, y_test) = mnist.load_data()    #数据预处理
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_train = x_train / 255.0; x_test = x_test / 255.0
# CNN搭建（卷积->下采样->卷积->下采样->转换为一位数组-》->全连接1- >输出层）
model = tf.keras.Sequential([tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
                             tf.keras.layers.MaxPool2D(2,2),
                             tf.keras.layers.Conv2D(64, (3,3),activation='relu'),
                             tf.keras.layers.MaxPool2D(2,2),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(10, activation='softmax')])
# 优化器、损失函数和评价指标编译
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
# x_train与y_train训练好的模型预测x_test和y_test，查看正确率
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
# 网络结构汇总
model.summary()