#将HWDB解压的方法
import os
import numpy as np
import struct
from PIL import Image
import json
import cv2
# data文件夹存放转换后的.png文件
data_dir = 'E:\VS_Programs\hanzi\HWDB1.0_yu'
# 路径为存放数据集解压后的.gnt文件
train_data_dir = os.path.join(data_dir, 'E:\VS_Programs\hanzi\Gnt1.0TrainPart1')
# test_data_dir = os.path.join(data_dir, '/home/admina/下载/OCR数据集/HWDB1.1tst_gnt')


def read_from_gnt_dir(gnt_dir=train_data_dir):
    def one_file(f):
        header_size = 10
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size: break
            sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
            tagcode = header[5] + (header[4] << 8)
            width = header[6] + (header[7] << 8)
            height = header[8] + (header[9] << 8)
            if header_size + width * height != sample_size:
                break
            image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
            yield image, tagcode

    for file_name in os.listdir(gnt_dir):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            with open(file_path, 'rb') as f:
                for image, tagcode in one_file(f):
                    yield image, tagcode


char_set = set()
for _, tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gbk')
    char_set.add(tagcode_unicode)
char_list = list(char_set)
char_dict = dict(zip(sorted(char_list), range(len(char_list))))
print(len(char_dict))
print("char_dict=", char_dict)
with open('E:\VS_Programs\hanzi\HWDB1.0_labels.txt','w') as f:
    f.write(json.dumps(char_dict))
    f.close()

import pickle

f = open('char_dict', 'wb')
pickle.dump(char_dict, f)
f.close()
train_counter = 0
test_counter = 0
for image, tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gbk')
    im = Image.fromarray(image)
    print('im',im)
    # im = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
# 路径为data文件夹下的子文件夹，train为存放训练集.png的文件夹  
    dir_name = 'E:\VS_Programs\hanzi\HWDB1.0_yu' + '\%0.5d' % char_dict[tagcode_unicode]
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    # im
    im.convert('RGB').resize((48, 48),Image.ANTIALIAS).save(dir_name + '/' + str(train_counter) + '.png')
    print("train_counter=", train_counter)
    train_counter += 1


# for image, tagcode in read_from_gnt_dir(gnt_dir=test_data_dir):
#     tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
#     im = Image.fromarray(image)
# 路径为data文件夹下的子文件夹，test为存放测试集.png的文件夹 
    # dir_name = '/home/admina/下载/OCR数据集/HWDB1.1tst_gnt_test' + '%0.5d' % char_dict[tagcode_unicode]
    # if not os.path.exists(dir_name):
    #     os.mkdir(dir_name)
    # im.convert('RGB').save(dir_name + '/' + str(test_counter) + '.png')
    # print("test_counter=", test_counter)
    # test_counter += 1