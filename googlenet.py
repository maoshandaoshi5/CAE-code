

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import utils as keras_utils
# from models.pym_attention import se_block,cbam_block,pyramid_attention_block_1

def conv2d_bn(x, filters, kernel_size=(3, 3), padding='same', strides=(1, 1),
              name=None):

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x

def inception(x, filters):
    """Utility function to implement the inception module.

    # Arguments
        x: input tensor.
        filters: a list of filter sizes.

    # Returns
        Output tensor after applying the inception.
    """
    if len(filters) != 4:
        raise ValueError('filters should have 4 components')
    if len(filters[1]) != 2 or len(filters[2]) != 2:
        raise ValueError('incorrect spec of filters')

    branch1x1 = conv2d_bn(x, filters[0], (1, 1))

    branch3x3 = conv2d_bn(x, filters[1][0], (1, 1))
    branch3x3 = conv2d_bn(branch3x3, filters[1][1], (3, 3))

    branch5x5 = conv2d_bn(x, filters[2][0], (1, 1))
    branch5x5 = conv2d_bn(branch5x5, filters[2][1], (5, 5))

    branchpool = layers.AveragePooling2D(
        pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branchpool = conv2d_bn(branchpool, filters[3], (1, 1))

    if backend.image_data_format() == 'channels_first':
        concat_axis = 1
    else:
        concat_axis = 3
    x = layers.concatenate(
        [branch1x1, branch3x3, branch5x5, branchpool], axis=concat_axis)
    return x


def GoogLeNetBN(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling='max',
                classes=1000,
                **kwargs):
   
    att_block = None
    if weights is not None:
        raise ValueError('weights is not currently supported')
    if input_tensor is None:
        if input_shape is None:
            raise ValueError('neither input_tensor nor input_shape is given')
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = conv2d_bn(img_input, 64, (7, 7), strides=(1, 1))
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    # if att_block is not None:
    # x = att_block(x, name='a')
    x = conv2d_bn(x,  64, (1, 1))
    x = conv2d_bn(x, 192, (3, 3))
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    # if att_block is not None:
    #     x = att_block(x, name='b')
    x = inception(x, ( 64,  (96, 128), (16,  32),  32))  # 3a: 28x28x256
    x = inception(x, (128, (128, 192), (32,  96),  64))  # 3b: 28x28x480
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    # if att_block is not None:
    #     x = att_block(x, name='c')
    # x = inception(x, (192,  (96, 208), (16,  48),  64))  # 4a: 14x14x512
    # x = inception(x, (160, (112, 224), (24,  64),  64))  # 4b: 14x14x512
    # x = inception(x, (128, (128, 256), (24,  64),  64))  # 4c: 14x14x512
    # x = inception(x, (112, (144, 288), (32,  64),  64))  # 4d: 14x14x528
    # x = inception(x, (256, (160, 320), (32, 128), 128))  # 4e: 14x14x832
    # # x = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    # if att_block is not None:
    #     x = att_block(x, name='d')
    # x = inception(x, (256, (160, 320), (32, 128), 128))  # 5a: 7x7x832
    # x = inception(x, (384, (192, 384), (48, 128), 128))  # 5b: 7x7x1024

    if include_top:
        # Classification block
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='global_pool')(x)
        else:
            raise ValueError('bad spec of global pooling')
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='googlenet_bn')

    return model

if __name__ == '__main__':
    model = GoogLeNetBN(input_shape=(48,48,1),classes=1024)
    print( model.summary())
