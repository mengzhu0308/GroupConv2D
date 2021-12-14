'''
@Author:        ZM
@Date and Time: 2020/6/11 16:07
@File:          GroupConv2D.py
'''

from keras import backend as K
from keras.layers import InputSpec, Layer
from keras import activations, constraints, initializers, regularizers

def normalize_tuple(value):
    if isinstance(value, int):
        return (value,) * 2
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError('The argument must be a tuple of 2 integers. Received: {}'.format(value))
        if len(value_tuple) != 2:
            raise ValueError('The argument must be a tuple of 2 integers. Received: {}'.format(value))
        for single_value in value_tuple:
            try:
                int(single_value)
            except ValueError:
                raise ValueError('The argument must be a tuple of 2 integers. '
                                 'Received: {} including element {} of type {}'.format(value,
                                                                                       single_value,
                                                                                       type(single_value)))
    return value_tuple

def normalize_padding(value):
    padding = value.lower()
    allowed = {'valid', 'same'}
    if padding not in allowed:
        raise ValueError('The `padding` argument must be one of "valid", "same". Received: {}'.format(padding))
    return padding

def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
    if input_length is None:
        return None
    assert padding in {'same', 'valid'}
    dilated_filter_size = (filter_size - 1) * dilation + 1
    if padding == 'same':
        output_length = input_length
    else:
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

class GroupConv2D(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 groups=1,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GroupConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = normalize_tuple(kernel_size)
        self.strides = normalize_tuple(strides)
        self.padding = normalize_padding(padding)
        self.dilation_rate = normalize_tuple(dilation_rate)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.groups = groups
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape):
        channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]

        if input_dim % self.groups != 0 or self.filters % self.groups != 0:
            raise ValueError('The argument "groups" must be common divisor of input filters and output filters')

        kernel_shape = self.kernel_size + (input_dim, self.filters // self.groups)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.input_dim_per_group = input_dim // self.groups
        self.built = True

    def call(self, inputs, **kwargs):
        outputs = [K.conv2d(
            inputs[..., i * self.input_dim_per_group:(i + 1) * self.input_dim_per_group],
            self.kernel[..., i * self.input_dim_per_group:(i + 1) * self.input_dim_per_group, :],
            strides=self.strides,
            padding=self.padding,
            data_format='channels_last',
            dilation_rate=self.dilation_rate) for i in range(self.groups)]
        outputs = K.concatenate(outputs)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format='channels_last')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.filters,)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'groups': self.groups,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(GroupConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))