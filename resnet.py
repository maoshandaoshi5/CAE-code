import tensorflow as tf
from tensorflow.keras import layers, models
# 定义ResNet块
def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False):
    shortcut = x
    if conv_shortcut:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

# 创建ResNet模型
def create_resnet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # 堆叠残差块
    num_blocks_list = [2, 2, 2, 2]  # 每个阶段的残差块数量
    # num_blocks_list = [4, 4, 4, 4]  # 每个阶段的残差块数量
    for stage, num_blocks in enumerate(num_blocks_list):
        for block in range(num_blocks):
            stride = 1
            if stage > 0 and block == 0:
                stride = 2  # 第一个残差块的步幅为2

            x = residual_block(x, 64 * (2**stage), stride=stride, conv_shortcut=True)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, x, name='resnet')
    return model

if __name__ == '__main__':
    # 创建ResNet模型
    input_shape = (24, 24, 3)  # 输入图像的形状
    num_classes = 500  # 类别数量
    resnet_model = create_resnet(input_shape, num_classes)

    # 编译模型
    resnet_model.compile(optimizer='adam',
                        loss='categorical_crossentropy',  # 适用于多类别分类任务
                        metrics=['accuracy'])

    # 打印模型摘要
    resnet_model.summary()