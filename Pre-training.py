# from torch.nn import functional as F
from imutils import paths
import cv2
import os
import csv
import numpy as np
import json
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus']=False # 解决负号不显示问题
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

n1_time = time.time()

files_path = os.listdir(path)
# print(files_path)
names = [int(i) for i in files_path[500:1000]]
# print('这是第'+str(i)+'批次')
data = []
labels = []
address = []
address1 = []
for row in files_path[500:1000]:
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
        address1.append(str(row))
data = np.array(data)
labels = np.array(labels)
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

#对所有的数据进行聚类，并获取聚类标签，聚类中心，整合为一个dataframe
data = np.reshape(data,(len(data),img_size*img_size))

k_cluster = 10

reducer = umap.UMAP(random_state=42)
# reducer = PCA(n_components=2)
embedding = reducer.fit_transform(data)

# pca_150 = PCA(n_components=2)
# embedding = pca_150.fit_transform(data)
# embedding = reducer.fit_transform(pca_result_150)
embedding = tuple(embedding)
# print('embedding',embedding)


import random
from math import *
import matplotlib.pyplot as plt
 
#初始化聚类中心
def begin_cluster_center(data_points,k):
    center=[]
    length=len(data_points)#长度
    rand_data = random.sample(range(0,length), k)#生成k个不同随机数
    for i in range(k):#得出k个聚类中心(随机选出)
        center.append(data_points[rand_data[i]])
    return center
 
#计算最短距离（欧式距离）
def distance(a,b):
    length=len(a)
    sum = 0
    for i in range(length):
        sq = (a[i] - b[i]) ** 2
        sum += sq
    return sqrt(sum)
#分配样本
# 按照最短距离将所有样本分配到k个聚类中心中的某一个
def assign_points(data_points,center,k):
    assignment=[]
    for i in range(k):
        assignment.append([])
    for point in data_points:
        min = 10000000
        flag = -1
        for i in range(k):
            value=distance(point,center[i])#计算每个点到聚类中心的距离
            if value<min:
                min=value#记录距离的最小值
                flag=i   #记录此时聚类中心的下标
        assignment[flag].append(point)
    return assignment
 
#更新聚类中心,计算每一簇中所有点的平均值
def update_cluster_center(center,assignment,k):
    for i in range(k):#assignment中的每一簇
        x=0
        y=0
        length=len(assignment[i])#每一簇的长度
        if length!=0:
            for j in range(length):  # 每一簇中的每个点
                x += assignment[i][j][0]  # 横坐标之和
                y += assignment[i][j][1]  # 纵坐标之和
            center[i] = (x / length, y / length)
    return center
 
#计算平方误差
def getE(assignment,center):
    sum_E=0
    for i in range(len(assignment)):
        for j in range(len(assignment[i])):
            sum_E+=distance(assignment[i][j],center[i])
    return sum_E
 
#计算各个聚类中心的新向量，更新距离，即每一类中每一维均值向量。
# 然后再进行分配，比较前后两个聚类中心向量是否相等，若不相等则进行循环，
# 否则终止循环，进入下一步。
def k_means(data_points,k):
    # 由于初始聚类中心是随机选择的，十分影响聚类的结果，聚类可能会出现有较大误差的现象
    # 因此如果由初始聚类中心第一次分配后有结果为空，重新选择初始聚类中心，重新再聚一遍，直到符合要求
    while 1:
        # 产生初始聚类中心
        begin_center = begin_cluster_center(data_points, k)
        # 第一次分配样本
        assignment = assign_points(data_points, begin_center, k)
        for i in range(k):
            if len(assignment[i]) == 0:#第一次分配之后有结果为空，说明聚类中心没选好，重新产生初始聚类中心
                continue
        break
    #第一次的平方误差
    begin_sum_E=getE(assignment,begin_center)
    # 更新聚类中心
    end_center = update_cluster_center(begin_center, assignment, k)
    # 第二次分配样本
    assignment = assign_points(data_points, end_center, k)
    # 第二次的平方误差
    end_sum_E = getE(assignment, end_center)
    count = 2  # 计数器
    #比较前后两个聚类中心向量是否相等
    #print(compare(end_center,begin_center)==False)
    while( begin_sum_E != end_sum_E):
        begin_center=end_center
        begin_sum_E=end_sum_E
        # 再次更新聚类中心
        end_center = update_cluster_center(begin_center, assignment, k)
        # 进行分配
        assignment = assign_points(data_points, end_center, k)
        #计算误差
        end_sum_E = getE(assignment, end_center)
        count = count + 1      #计数器加1
    return assignment,end_sum_E,end_center,count
def print_result(count,end_sum_E,k,assignment):
    # 打印最终聚类结果
    print('经过', count, '次聚类，平方误差为：', end_sum_E)
    print('---------------------------------分类结果---------------------------------------')
    for i in range(k):
        print('第',i+1,'类数据：',assignment[i])
    print('--------------------------------------------------------------------------------\n')
 
def plot(k, assignment,center):
    #初始坐标列表
    x = []
    y = []
    for i in range(k):
        x.append([])
        y.append([])
    # 填充坐标 并绘制散点图
    for j in range(k):
        for i in range(len(assignment[j])):
            x[j].append(assignment[j][i][0])# 横坐标填充
        for i in range(len(assignment[j])):
            y[j].append(assignment[j][i][1])# 纵坐标填充
        plt.scatter(x[j], y[j],marker='o')
        plt.scatter(center[j][0], center[j][1],c='b',marker='*')#画聚类中心
    # 设置标题
    plt.title('Improved K-means Scatter Diagram')
    # plt.title('K-means Scatter Diagram')
    # 设置X轴标签
    # plt.xlabel('X')
    # 设置Y轴标签
    # plt.ylabel('Y')
    # 显示散点图
    plt.savefig('E:\\fig9.png',dpi=600,bbox_inches='tight')
    plt.show()
 
def main():
    # 3个聚类中心
    k = 10
    data_points = embedding
    assignment, end_sum_E, end_center, count = k_means(data_points, k)
    min_sum_E = 1000
    #返回较小误差
    while min_sum_E>end_sum_E:
        min_sum_E=end_sum_E
        assignment, end_sum_E, end_center, count = k_means(data_points,k)
    print_result(count, min_sum_E, k, assignment)#输出结果
    plot(k, assignment,end_center)#画图
main()













embedding = np.array(embedding)

# # k-means聚类，问题在于标签不同，分簇的结果不相同，大概分类准确率在70左右
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=k_cluster)
km = kmeans.fit(embedding)
centroid = km.cluster_centers_
print('center',centroid.shape,centroid)
center_df = pd.DataFrame(centroid, columns=['x', 'y'])
# new_df['centroid'] = list(centroid)
km_labels = km.labels_
print(km.labels_.shape,km.labels_)
test = np.array(km_labels)-np.array(labels)
num = 0
for k in test:
    if k == 0:
        num+=1
print('accuracy',num,num/len(labels))
print(km_labels[:30],labels[:30])

df_2d = pd.DataFrame()
df_2d['X'] = list(embedding[:,0].squeeze())
df_2d['Y'] = list(embedding[:,1].squeeze())
df_2d['标注类别'] = list(labels)
df_2d['标注文字'] = list(labels1)
df_2d['预测类别'] = list(km_labels)
df_2d['图像名称'] = list(address)
df_2d['图像路径'] = list(address1)
print(df_2d)
#散点图
show_feature = '预测类别'
import csv

def clear_csv_file(file_path):
    with open(file_path, 'w') as file:
        file.truncate(0)
clear_csv_file("data.csv")#删除csv的内容

def gamma_function(x, gamma):
    return np.power(x, gamma)
x = list(np.random.randint(0,100,10))


for i in range(k_cluster):
    df_2d_1 = pd.DataFrame(columns=df_2d.columns)
    for j in range(len(df_2d)):
        if int(df_2d['预测类别'][j]) == i :
            df_2d_1.loc[len(df_2d_1)] = df_2d.values[j]
    # print('label == 1',df_2d_1)
    y = df_2d_1['标注文字'].value_counts().index
    x = df_2d_1['标注文字'].value_counts()

    l_list = []
    for i in range(len(x)):
        if x[i] <= int(len(names)*0.04):
            l_list.append(y[i])

    for i in range(len(df_2d_1)):
        if df_2d_1['标注文字'][i] in l_list:
            df_2d_1 = df_2d_1.drop(i)
    # print('df_2d_1删除低值',df_2d_1)
    df_2d_1 = df_2d_1.reset_index()

    y = df_2d_1['标注文字'].value_counts().index
    y = list(y)
    x = df_2d_1['标注文字'].value_counts()
    new_label = list(range(1,len(y)+1))
    plt.pie(x)
    plt.savefig('E:\\fig10.png',dpi=600,bbox_inches='tight')
    plt.show()
    demo1 = list(df_2d_1['图像路径'].value_counts().index)
    print(len(demo1),demo1)
    with open('E:\\VS_Programs\\6.20\\recognize\\models\\data.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(demo1)

n2_time = time.time()
print('times:',n2_time-n1_time)





















