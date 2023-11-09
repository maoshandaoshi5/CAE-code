import tensorflow as tf
from tensorflow.keras import layers, models

def AlexNet(input_shape,latent_dim,num_classes):
    # 创建编码器
    def create_encoder(input_shape, latent_dim):
        inputs = tf.keras.Input(shape=input_shape)
        # x = layers.Flatten()
        x = layers.Conv2D(128,(1,1),activation='relu',padding='same')(inputs)
        x = layers.Conv2D(128,(1,1),activation='relu',padding='same')(x)
        x = layers.Conv2D(64,(1,1),activation='relu',padding='same')(x)
        x = layers.Conv2D(64,(1,1),activation='relu',padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(latent_dim, activation='relu')(x)  # 潜在表示的维度
        encoder = models.Model(inputs, x, name='encoder')
        # encoder.summary()
        return encoder

    # 创建解码器
    def create_decoder(latent_dim, original_dim):
        inputs = tf.keras.Input(shape=(input_shape[0],input_shape[0],latent_dim))
        # inputs = tf.keras.Input(shape=(latent_dim))
        x = layers.Conv2D(64,(1,1),activation='relu',padding='same')(inputs)
        x = layers.Conv2D(64,(1,1),activation='relu',padding='same')(x)
        x = layers.Conv2D(128,(1,1),activation='relu',padding='same')(x)
        x = layers.Conv2D(128,(1,1),activation='relu',padding='same')(x)
        x = layers.Dense(original_dim, activation='sigmoid')(x)  # 输出层使用sigmoid激活函数
        decoder = models.Model(inputs, x, name='decoder')
        return decoder

    # 创建自编码器
    def create_autoencoder(encoder, decoder,input_shape):

        inputs = tf.keras.Input(shape=input_shape)
        encoded = encoder(inputs)
        decoded = decoder(encoded)
        autoencoder = models.Model(inputs, decoded, name='autoencoder')
        return autoencoder

    # 添加分类头
    def add_classification_head(base_model, num_classes):
        x = layers.Dense(num_classes, activation='softmax')(base_model.layers[-1].output)
        classification_model = models.Model(inputs=base_model.input, outputs=x, name='classification_head')
        return classification_model



    # input_shape = inputs
    # 创建编码器
    encoder = create_encoder(input_shape, latent_dim)
    # 创建解码器s
    # decoder = create_decoder(latent_dim, input_shape[0] * input_shape[1] * input_shape[2])
    # 创建自编码器
    # autoencoder = create_autoencoder(encoder, decoder,input_shape)

    # 添加分类头
    classification_model = add_classification_head(encoder, num_classes)
    return classification_model

# if __name__ == '__main__':
#      # 输入图像的形状
#     input_shape = (24, 24, 3)
#     # 潜在表示的维度
#     latent_dim = 32  # 这是自编码器的编码器部分的输出维度
#     # 类别数量
#     num_classes = 500
#     model = en_de(input_shape,latent_dim,num_classes)
#     # 编译分类模型，这里使用分类任务的损失函数和指标
#     model.compile(optimizer='adam',
#                 loss='categorical_crossentropy',  # 分类任务的损失函数
#                 metrics=['accuracy'])  # 分类准确率作为指标

#     # 打印分类模型摘要
#     model.summary()


