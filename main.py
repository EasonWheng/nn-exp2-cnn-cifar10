import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10

(train_data, train_label), (test_data, test_label) = cifar10.load_data()

x_data = train_data.astype('float32') / 255.
y_data = test_data.astype('float32') / 255.
import numpy as np

def one_hot(label, num_classes):
    # your code here# 补充one-hot编码函数
    return label_one_hot

num_classes = 10
train_label = train_label.astype('int32')
train_label = np.squeeze(train_label)
x_label = one_hot(train_label, num_classes)
test_label = test_label.astype('int32')
y_label = np.squeeze(test_label)
print(f'x_label[0:5]={x_label[0:5]}')
print(f'train_label[0:5]={train_label[0:5]}')

from keras import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout

cnn = Sequential()
#unit1
cnn.add(Convolution2D(32, kernel_size=[3, 3], input_shape=(32, 32, 3), activation='relu', padding='same'))
cnn.add(Convolution2D(32, kernel_size=[3, 3], activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=[2, 2], padding='same'))
cnn.add(Dropout(0.5))

#unit2
# 编写网络的第二部分，可自行尝试增加更多的卷积层，改变通道数、激活函数等，以下设置仅供参考：
# (两个2D卷积层，均为64个通道，卷积核为(3, 3)，激活函数为relu，padding为same；一个2D池化层，pool_size为(2, 2)，padding为same，最后是Dropout层，保留概率为0.5)
# your code here#

cnn.add(Flatten())

cnn.add(Dense(512, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))

print(cnn.summary())

# 采用model.compile模型编译，设置优化器、学习率、损失函数和准确率观测
# your code here

history_cnn = cnn.fit(x_data,  x_label,
                      epochs=50, batch_size=32, shuffle=True, verbose=1, validation_split=0.1)

history_dict = history_cnn.history
print(f'history_dict.keys()={history_dict.keys()}')


import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(np.array(history_cnn.history['loss']))
plt.plot(np.array(history_cnn.history['val_loss']))
plt.xlabel('Epoch')
plt.ylabel('Train loss')
plt.legend(['loss', 'val_loss'])
plt.show()
plt.figure(2)
plt.plot(np.array(history_cnn.history['acc']))
plt.plot(np.array(history_cnn.history['val_acc']))
plt.xlabel('Epoch')
plt.ylabel('Train acc')
plt.legend(['acc', 'val_acc'])
plt.show()

cnn.save('model/cnn.h5')
cnn = keras.models.load_model('model/cnn.h5')
test_out = cnn.predict(y_data)


num = 0
total_num = y_data.shape[0]

for i in range(total_num):
    predict = np.argmax(test_out[i])
    if predict == y_label[i]:
        num += 1
accuracy = num / total_num
print(f'accuracy={accuracy}')