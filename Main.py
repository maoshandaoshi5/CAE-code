# from torch.nn import functional as F
from imutils import paths
import cv2
import os
import numpy as np
import json
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 14
import tensorflow as tf
import tensorflow.python.keras
from tensorflow.python.keras import models
import time
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.models import load_model
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
import pandas as pd
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from tensorflow.keras import losses


num_class = 5
batch_size = 32
img_size=24
PATH = 'Xlnet.pth'   #模型参数保存路径
path = 'E:\VS_Programs\hanzi\HWDB1.0_yu'
# path = 'E:\\VS_Programs\\hanzi\\data\\train'
#将数据集随机排序
def shuffle_data(data, label):
    idx=np.arange(len(data))
    np.random.shuffle(idx)
    return data[idx,...],label[idx,...]

files_path = os.listdir(path)
# print(files_path)
names = [int(i) for i in files_path]
# print('这是第'+str(i)+'批次')
data = []
labels = []
address = []
for row in files_path:
    # print('row',row)[70*i:70+70*i]
    im_path = os.listdir(os.path.join(path,row))
    # print(os.listdir(os.path.join(path,row))[5*i:5+5*i])
    for row_1 in im_path:
        image = cv2.imread(os.path.join(path,row,row_1))
        # print(os.path.join(path,row,row_1))
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) #图像灰度化
        image = cv2.resize(image,(img_size,img_size))
        # cv2.imshow('11',image)
        data.append(image)
        labels.append(int(row))
        address.append(os.path.join(row,row_1))
data = np.array(data)
labels = np.array(labels)
# print(data,labels,names)
# print(data.shape, labels.shape, len(names))
labels1 = []
path_txt = 'E:\\VS_Programs\\hanzi\\HWDB1.0_labels.txt'
# path_txt = 'E:\\VS_Programs\\6.20\\data_load\\train_labels.txt'
with open(path_txt,'r') as f:
    dic = json.loads(f.read())
    for i in range(len(labels)):
        for key, value in dic.items():
            if value == labels[i]:
                # print('原标签{}'.format(key))
                labels1.append(key)

lst_labels = list(labels)
lst = []
for i in lst_labels:
    if i not in lst:
        lst.append(i)
# print(lst,len(lst))
for i in range(len(lst_labels)):
    lst_labels[i] = lst.index(lst_labels[i])
# print(lst_labels)

labels = to_categorical(lst_labels)

indices = np.random.permutation(len(data))
data = data[indices]
labels = labels[indices]

# labels = to_categorical(labels)
print(data.shape,labels.shape,len(names))

from alexnet import AlexNet
from CAE import CAE
from resnet import create_resnet
from cnn import cnn
from googlenet import GoogLeNetBN

for i in range(5):
    classes = len(names)   #标签数量
    epochs = 50
    pc='13'
    early_stopper = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10, mode='max')
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-2)
    if i == 0 :
        print('---'*10)
        print('model en_de')
        start_time1 = time.time()
        # print(model.summary())
        model = AlexNet(input_shape=(img_size, img_size, 1),latent_dim=3200,num_classes=classes)
        # 编译分类模型，这里使用分类任务的损失函数和指标
        optimizer = tf.keras.optimizers.Adam(0.001)
        model.compile(optimizer=optimizer,#'Adagrad'
                    loss='binary_crossentropy',  # 适用于多类别分类任务
                    metrics=['accuracy'])  # 分类准确率作为指标

        # 打印分类模型摘要
        # model.summary()
        history1 = model.fit(data, labels,#epochs>=100
                    batch_size=32,
                    epochs=epochs,
                    validation_split=0.2,
                    shuffle=True,
                    verbose=2,
                    callbacks=[lr_reducer]
                    )
        end_time1 = time.time()
        ov_time1 = end_time1 - start_time1
        print('---'*10)
    elif i == 1 :       
        
        start_time2 = time.time()         
        model = CAE(size=img_size,outsize=classes)
        optimizer = tf.keras.optimizers.Adam(0.001)
        model.compile(optimizer=optimizer,#'Adagrad'
                    loss='binary_crossentropy',  # 适用于多类别分类任务
                    metrics=['accuracy'])  # 分类准确率作为指标
        history2 = model.fit(data, labels,batch_size=32,epochs=epochs,validation_split=0.2,shuffle=True,verbose=2)#epochs=20,99%
        end_time2 = time.time()
        ov_time2 = end_time2 - start_time2
        print('---'*10)
        model.save('E:\\VS_Programs\\6.20\\save\\cnn_test1_1.h5')  #保存最后一个训练的模型
    elif i == 2:
        start_time3 = time.time()
        model = create_resnet(input_shape=(img_size,img_size,1),num_classes=classes)
        optimizer = tf.keras.optimizers.Adam(0.001)
        model.compile(optimizer=optimizer,#'Adagrad'
                    loss='binary_crossentropy',  # 适用于多类别分类任务
                    metrics=['accuracy'])  # 分类准确率作为指标
        history3 = model.fit(data, labels,batch_size=32,epochs=epochs,validation_split=0.2,shuffle=True,verbose=2)#epochs=10,90%
        end_time3 = time.time()
        ov_time3 = end_time3 - start_time3
        # print(ov_time3)
        print('---'*10)
    if i == 3:
        start_time4 = time.time()
        model = cnn(input_shape=(img_size,img_size,1),classes=classes)
        optimizer = tf.keras.optimizers.Adam(0.001)
        model.compile(optimizer=optimizer,#'Adagrad'
                    loss='binary_crossentropy',  # 适用于多类别分类任务
                    metrics=['accuracy'])  # 分类准确率作为指标
        history4 = model.fit(data, labels,batch_size=32,epochs=epochs,validation_split=0.2,shuffle=True,verbose=2)#epochs=
        end_time4 = time.time()
        ov_time4 = end_time4 - start_time4
        print('---'*10)
    elif i == 4:
        start_time5 = time.time()
        model = GoogLeNetBN(input_shape=(img_size,img_size,1),classes=classes)
        optimizer = tf.keras.optimizers.Adam(0.01)
        model.compile(optimizer=optimizer,#'Adagrad'
                    loss='binary_crossentropy',  # 适用于多类别分类任务
                    metrics=['accuracy'])  # 分类准确率作为指标
        history5 = model.fit(data, labels,batch_size=32,epochs=epochs,validation_split=0.2,shuffle=True,verbose=2)#epochs=
        end_time5 = time.time()
        ov_time5 = end_time5 - start_time5
        print('---'*10)
print(ov_time2)

# one model
# plt.plot(history2.history['accuracy'],linestyle='--',label='accuracy')
# plt.plot(history2.history['loss'],label='loss')
# plt.plot(history2.history['val_accuracy'],label='val_accuracy')
# plt.plot(history2.history['val_loss'],label='val_loss')
# plt.title('accuracy')
# plt.savefig('E:\\桌面\\提交\\初稿\\训练\\'+pc+'loss_accuracy.png',dpi=600, bbox_inches='tight')
# plt.legend()
# plt.show()

#all of models
# plt.plot(history1.history['accuracy'],label='AlexNet')
# plt.plot(history2.history['accuracy'],label='ResNet')
# plt.plot(history3.history['accuracy'],label='GoogleNet')
# plt.plot(history4.history['accuracy'],linestyle='--',label='CAE')
# plt.plot(history5.history['accuracy'],label='CNN')
# plt.legend()
# plt.title('accuracy')
# plt.savefig('E:\\桌面\\提交\\初稿\\训练\\'+pc+'accuracy.png',dpi=600, bbox_inches='tight')
# plt.show()
# plt.plot(history1.history['loss'],label='AlexNet')
# plt.plot(history2.history['loss'],label='ResNet')
# plt.plot(history3.history['loss'],label='GoogleNet')
# plt.plot(history4.history['loss'],linestyle='--',label='CAE')
# plt.plot(history5.history['loss'],label='CNN')
# plt.legend()
# plt.title('loss')
# plt.savefig('E:\\桌面\\提交\\初稿\\训练\\'+pc+'loss.png',dpi=600, bbox_inches='tight')
# plt.show()
# plt.plot(history1.history['val_accuracy'],label='AlexNet')
# plt.plot(history2.history['val_accuracy'],label='ResNet')
# plt.plot(history3.history['val_accuracy'],label='GoogleNet')
# plt.plot(history4.history['val_accuracy'],linestyle='--',label='CAE')
# plt.plot(history5.history['val_accuracy'],label='CNN')
# plt.legend()
# plt.title('val_accuracy')
# plt.savefig('E:\\桌面\\提交\\初稿\\训练\\'+pc+'val_accuracy.png',dpi=600, bbox_inches='tight')
# plt.show()
# plt.plot(history1.history['val_loss'],label='AlexNet')
# plt.plot(history2.history['val_loss'],label='ResNet')
# plt.plot(history3.history['val_loss'],label='GoogleNet')
# plt.plot(history4.history['val_loss'],linestyle='--',label='CAE')
# plt.plot(history5.history['val_loss'],label='CNN')
# plt.legend()
# plt.title('val_loss')
# plt.savefig('E:\\桌面\\提交\\初稿\\训练\\'+pc+'al_loss.png',dpi=600, bbox_inches='tight')
# plt.show()

#model times
# print(ov_time1)
# print('---'*10)
# print(ov_time2)
# print('---'*10)
# print(ov_time3)
# print('---'*10)
# print(ov_time4)
# print('---'*10)
# print(ov_time5)
# print('---'*10)


