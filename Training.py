import csv
import cv2
import os
import numpy as np
import json
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 14
import tensorflow as tf
import time
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from CAE import CAE

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

def get_flops_params():
    sess = tf.compat.v1.Session()
    graph = sess.graph
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

# for row in files_path[500:700]:
with open('E:\\VS_Programs\\6.20\\recognize\\models\\data.csv', 'r') as f:
    reader = csv.reader(f)
    # print(type(reader))
    k_num = 10
    for row_list in reader:
        print(row_list)
        names = row_list
        k_num = k_num +1
        data = []
        labels = []
        address = []
        address1 = []
        for row in row_list:
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
                address1.append(int(row))
        data = np.array(data)
        labels = np.array(labels)
        # print(data,labels,names)
        # print(data.shape, labels.shape, len(names))
        labels1 = []
        path_txt = 'E:\\VS_Programs\\hanzi\\HWDB1.0_labels.txt'
        with open(path_txt,'r') as f:
            dic = json.loads(f.read())
            for i in range(len(labels)):
                for key, value in dic.items():
                    if value == labels[i]:
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


        classes = len(names)   #标签数量
        epochs = 30
        start_time1 = time.time()
        # print(model.summary())

        early_stopper = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=20, mode='max')
        model = CAE(size=img_size,outsize=classes)
        # 获取模型浮点运算总次数和模型的总参数
        get_flops_params()

        # model.cuda()
        # model.to(device)
        # modelsize(model)
        # 编译分类模型，这里使用分类任务的损失函数和指标
        optimizer = tf.keras.optimizers.Adam(0.001)
        model.compile(optimizer=optimizer,#'Adagrad'
                loss='binary_crossentropy',  # 适用于多类别分类任务
                metrics=['accuracy']
                )  # 分类准确率作为指标

        # 打印分类模型摘要
        # model.summary()
        history = model.fit(data, labels,#epochs>=100
                batch_size=16,
                epochs=20,
                validation_split=0.2,
                shuffle=True,
                verbose=2,
                callbacks=[early_stopper]
                )
        end_time1 = time.time()
        ov_time1 = end_time1 - start_time1
        print("model times:",ov_time1)
        
        # print("Total memory usage:", profile_info.total_requested_bytes)
        # print("Peak memory usage:", profile_info.total_peak_bytes)
        # model.save("E:\\VS_Programs\\6.20\\save\\model_1\\"+str(k_num)+".h5")
        fig1 = plt.figure(1)
        plt.plot(history.history['accuracy'],label = 'acc')
        plt.plot(history.history['loss'],label='loss')
        plt.plot(history.history['val_accuracy'],label='val_acc')
        plt.plot(history.history['val_loss'],label='val_loss')
        plt.title('accuracy & loss')
        plt.legend()
        # plt.savefig("E:\\VS_Programs\\6.20\\save\model_1\\"+str(k_num)+".png",dpi=600, bbox_inches='tight')
        plt.show()
        # plt.pause(6)# 间隔的秒数：6s
        # plt.close(fig1)
















