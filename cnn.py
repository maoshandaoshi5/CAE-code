import tensorflow as tf
from tensorflow.keras import layers, models

def cnn(input_shape,classes):
    # 创建一个序贯模型
    model = models.Sequential()

    # 添加卷积层
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(32, (1, 1), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (1, 1), activation='relu'))

    # 将卷积层的输出展平为一维
    model.add(layers.Flatten())

    # 添加全连接层
    model.add(layers.Dense(6400, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(classes, activation='softmax'))  # 500个类别，使用softmax作为输出层的激活函数
    return model

if __name__ == '__main__':
    model = cnn(input_shape=(48,48,1),classes=1000)
   
    # 编译模型
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',  # 适用于多类别分类任务
                metrics=['accuracy'])

    # 打印模型摘要
    model.summary()
    print( model.summary())
