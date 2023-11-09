#读取一张图片，经过模型的预测后，准确值最高的是最正确的

import cv2
import os
import csv
import json
import keras
import pandas as pd
import numpy as np
from keras.models import load_model

labels_lsit = []
with open('E:\\VS_Programs\\6.20\\recognize\\models\\data.csv', 'r') as f:#读取列表
    reader = csv.reader(f)
    # print(type(reader))
    for row_list in reader:
        labels_lsit.append(row_list)
#         print(len(row_list))

# print(len(labels_lsit))


image_size = 24

# path = 'E:\\VS_Programs\\6.20\\data\\00186\\91158.png'
# path = 'E:\\VS_Programs\\hanzi\\HWDB1.0_yu\\02194\\370019.png'
path = 'E:\\VS_Programs\\hanzi\\data\\Chinese Calligraphy Styles by Calligraphers_datasets\\Chinese Calligraphy Styles by Calligraphers_data_datasets\\data\\test\\bdsr\\1963.jpg'
# path = 'E:\\VS_Programs\\hanzi\\data\\Chinese Calligraphy Styles by Calligraphers_datasets\\Chinese Calligraphy Styles by Calligraphers_data_datasets\\data\\test\\bdsr\\1618.jpg'
image = cv2.imread(path)
image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
image = cv2.resize(image,(image_size,image_size))
image = np.array([image])
# image = [image]
# image = np.expand_dims(image, axis=2)
print(image.shape)
model_path = 'E:\\VS_Programs\\6.20\\save\\model_1'

max_list = []
labels = []
num = 0
path_txt = 'E:\\VS_Programs\\hanzi\\HWDB1.0_labels.txt'

for root, dirs, files in os.walk(model_path):
    for file in files:
        model_path_1 = os.path.join(root, file)
        if model_path_1.split('.')[-1] == 'h5' :
            print(num,model_path_1)#提取所有的模型参数
            CAE = load_model(model_path_1)
            predict = CAE.predict(image)
            predict = list(predict[0])
            labels1 = []
            with open(path_txt,'r') as f:
                dic = json.loads(f.read())
                for i in range(len(labels_lsit[num])):
                    for key, value in dic.items():
                        if value == int(labels_lsit[num][i]):
                            # print('原标签{}'.format(key))
                            labels1.append(key)
            print(len(predict),len(labels1))
            df = {'num2':predict,
                'num3':labels1
                }
            dataf = pd.DataFrame(df)
            dataf = dataf.sort_values(by=['num2'],ascending=False)
            print(dataf.head(12))
            num += 1













