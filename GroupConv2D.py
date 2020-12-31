'''
@Author:        ZM
@Date and Time: 2020/6/11 16:07
@File:          GroupConv2D.py
'''

import math
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Layer
from keras import initializers, regularizers

'''
https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf
Aggregated Residual Transformations for Deep Neural Networks
'''
class GroupConv2D(Layer):
    def __init__(self, out_filters, kernel_size, strides=1, use_bias=False, padding='same',
                 kernel_initializer='glorot_uniform', kernel_regularizer=None, groups=32, **kwargs):
        super(GroupConv2D, self).__init__(**kwargs)
        self.out_filters = out_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias
        self.padding = padding
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.groups = groups

    def build(self, input_shape):
        super(GroupConv2D, self).build(input_shape)
        filters = input_shape[-1]
        assert filters % self.groups == 0 and self.out_filters % self.groups == 0, '分组数必须能同时被输入通道数和输出通道数整除！'
        width_per_group = self.out_filters // self.groups
        self.convs = [Conv2D(width_per_group, self.kernel_size, strides=self.strides, use_bias=self.use_bias,
                             padding=self.padding, kernel_initializer=self.kernel_initializer,
                             kernel_regularizer=self.kernel_regularizer) for _ in range(self.groups)]

    def call(self, inputs, **kwargs):
        xs = tf.split(inputs, self.groups, axis=-1)
        xs = [conv(xs[i]) for i, conv in enumerate(self.convs)]
        return K.concatenate(xs)

    def compute_output_shape(self, input_shape):
        b, h, w, _ = input_shape
        if self.padding == 'same':
            new_h = math.ceil(h / self.strides)
            new_w = math.ceil(w / self.strides)
        else:
            new_h = math.ceil((h - self.kernel_size + 1) / self.strides)
            new_w = math.ceil((w - self.kernel_size + 1) / self.strides)

        return (b, new_h, new_w, self.out_filters)

    def get_config(self):
        config = {
            'out_filters': self.out_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'use_bias': self.use_bias,
            'padding': self.padding,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'groups': self.groups
        }
        base_config = super(GroupConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
