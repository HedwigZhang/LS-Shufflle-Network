import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2
from  tensorflow.keras import backend as K

def channel_split(input_batch):
    '''
    channel split layers
    '''
    ch = input_batch.get_shape().as_list()[-1]
    sub_ch = ch // 2
    x1 = input_batch[:, :, :, 0:sub_ch]
    x2 = input_batch[:, :, :, sub_ch:ch]
    return [x1, x2]


def channel_shuffle_layer(x):
    '''
    A helper function to realize channel shuffle
    :param x: 4D tensor
    :return: 4D tensor.
    '''
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0,1,2,4,3))
    x = K.reshape(x, [-1, height, width, channels])
    return x

def bn_relu_conv(inputs, ch, k_size, strs, name):
    '''
    the BN + relu + conv.
    :param inputs: 4D tensor (batch, h, w, ch)
    :param ch: the output channels
    :params str: 1D integer, strides
    '''
    x = layers.BatchNormalization(axis=-1, name='{}_bn'.format(name))(inputs)
    x = layers.Activation('relu', name='{}_relu'.format(name))(x)
    x = layers.Conv2D(ch, kernel_size = k_size, strides=(strs, strs), padding='same', kernel_regularizer = l2(0.002), bias_regularizer=l2(0.002), name='{}_conv'.format(name))(x)
    # kernel_initializer='he_uniform',   use_bias = False,
    return x

def bn_dw_conv(inputs, k_size, strs, name):
    '''
    the BN + relu + conv.
    :param inputs: 4D tensor (batch, h, w, ch)
    :param ch: the output channels
    :params str: 1D integer, strides
    '''
    x = layers.BatchNormalization(axis=-1, name='{}_bn'.format(name))(inputs)
    #x = layers.Activation('relu', name='{}_relu'.format(name))(x)
    x = layers.DepthwiseConv2D(kernel_size = k_size, strides=(strs, strs), padding='same', depthwise_regularizer = l2(0.002), bias_regularizer=l2(0.002), name='{}_dwconv'.format(name))(x)
    # kernel_initializer='he_uniform',  use_bias = False,
    return x

def bn_dw_conv_valid(inputs, k_size, strs, name):
    '''
    the BN + relu + conv.
    :param inputs: 4D tensor (batch, h, w, ch)
    :param ch: the output channels
    :params str: 1D integer, strides
    '''
    x = layers.BatchNormalization(axis=-1, name='{}_bn'.format(name))(inputs)
    #x = layers.Activation('relu', name='{}_relu'.format(name))(x)
    x = layers.DepthwiseConv2D(kernel_size = k_size, strides=(strs, strs), padding='valid', depthwise_regularizer = l2(0.002), bias_regularizer=l2(0.002), name='{}_dwconv'.format(name))(x)
    # kernel_initializer='he_uniform',  use_bias = False,
    return x

def long_short_residual(inputs, ch, name):
    '''
    The long and short residual structure.
    '''
    sub_ch = ch // 2
    ############################### channel split
    x1, res1 = layers.Lambda(channel_split, name = '{}_sp1'.format(name))(inputs)
    ############################### stage 1
    br1 = bn_relu_conv(x1, sub_ch, 1, 1, name = '{}_11'.format(name))
    br1 = bn_dw_conv(br1, 3, 1, name = '{}_12'.format(name))
    br1 = bn_relu_conv(br1, sub_ch, 1, 1, name = '{}_13'.format(name))
    ################################ short res
    res = layers.Add(name = '{}_add'.format(name))([br1, res1])
    ############################### stage 2
    br2 = bn_relu_conv(br1, sub_ch, 1, 1, name = '{}_14'.format(name))
    br2 = bn_dw_conv(br2, 3, 1, name = '{}_15'.format(name))
    br2 = bn_relu_conv(br2, sub_ch, 1, 1, name = '{}_16'.format(name))

    ############################### long res
    out = layers.Concatenate(axis = -1, name = '{}_concat'.format(name))([br2, res])
    ############################### channel shuffle
    out = layers.Lambda(channel_shuffle_layer, name = '{}_shuffle'.format(name))(out)

    return out


def long_short_residual_first(inputs, ch, st, name):
    '''
    The long and short residual structure.
    '''
    sub_ch = ch // 2
    ############################### stage 1
    br1 = bn_relu_conv(inputs, sub_ch, 1, 1, name = '{}_11'.format(name))
    br1 = bn_dw_conv(br1, 3, st, name = '{}_12'.format(name))
    br1 = bn_relu_conv(br1, sub_ch, 1, 1, name = '{}_13'.format(name))
    ################################ short res
    res1 = bn_dw_conv(inputs, 3, st, name = '{}_r1'.format(name))
    res1 = bn_relu_conv(res1, sub_ch, 1, 1, name = '{}_r2'.format(name))
    res = layers.Add(name = '{}_add'.format(name))([br1, res1])
    ############################### stage 2
    br2 = bn_relu_conv(br1, sub_ch, 1, 1, name = '{}_14'.format(name))
    br2 = bn_dw_conv(br2, 3, 1, name = '{}_15'.format(name))
    br2 = bn_relu_conv(br2, sub_ch, 1, 1, name = '{}_16'.format(name))

    ############################### long res
    out = layers.Concatenate(axis = -1, name = '{}_concat'.format(name))([br2, res])
    ############################### channel shuffle
    out = layers.Lambda(channel_shuffle_layer, name = '{}_shuffle'.format(name))(out)

    return out


def ls_net():
    '''
    The Across compress network
    '''
    input = layers.Input((32,32,3))
    # 32 x 32 x 3
    ############################################# The first layers
    x = bn_relu_conv(input, 24, 3, 1, name = 'CW0')
    ############################################# The stage 1
    x = long_short_residual_first(x, 116, 1, name = 'st10')
    x = long_short_residual(x, 116, name = 'st11')
    #x = long_short_residual(x, 64, name = 'st12')

    ############################################# The stage 2
    x = long_short_residual_first(x, 232, 2, name = 'st20')
    x = long_short_residual(x, 232, name = 'st21')
    x = long_short_residual(x, 232, name = 'st22')
    x = long_short_residual(x, 232, name = 'st23')
    ############################################# The stage 3
    x = long_short_residual_first(x, 464, 2, name = 'st30')
    x = long_short_residual(x, 464, name = 'st31')
    #x = long_short_residual(x, 256, name = 'st32')
    #########################################################
    x = bn_dw_conv_valid(x, 8, 1, name='last_dw')
    #########################################################
    #x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1000, activation="softmax", kernel_regularizer = l2(0.002), bias_regularizer = l2(0.002))(x) #, activity_regularizer=l1(0.0002)
    #x = layers.Dropout(0.5)(x)

    model = Model(input, x)

    return model