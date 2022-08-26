import functools

import os
import sys
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras import layers
# from tensorflow.python.keras.engine.input_spec import InputSpec
from keras.engine.input_spec import InputSpec

from tensorflow.python.keras.utils import conv_utils

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util.tf_export import keras_export
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from lib import models


class DistributedConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1),
                 groups=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, trainable=True, name=None, rank=2, layer_index=0, **kwargs):

        if "agent_host" in kwargs.keys():
            kwargs.pop("agent_host")
        super(DistributedConv2D, self).__init__(
            trainable=trainable, name=name,activity_regularizer=regularizers.get(activity_regularizer), **kwargs
        )
        self.rank = rank
        if isinstance(filters, float):
            filters = int(filters)
        if filters is not None and filters < 0:
            raise ValueError(f'Received a negative value for `filters`.'
                             f'Was expecting a positive value, got {filters}.')
        self.filters = filters
        self.groups = groups or 1
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=self.rank + 2)

        self._validate_init()
        self._is_causal = self.padding == 'causal'
        self._channels_first = self.data_format == 'channels_first'
        self._tf_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2)

        self.distributed = False
        self.layer_index = layer_index
        self.agent_host: models.AgentHost = None

        tf.config.run_functions_eagerly(True)

    def set_agent_host(self, host: models.AgentHost):
        self.agent_host = host

    def _validate_init(self):
        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
                'The number of filters must be evenly divisible by the number of '
                'groups. Received: groups={}, filters={}'.format(
                    self.groups, self.filters))

        if not all(self.kernel_size):
            raise ValueError('The argument `kernel_size` cannot contain 0(s). '
                             'Received: %s' % (self.kernel_size,))

        if not all(self.strides):
            raise ValueError('The argument `strides` cannot contains 0(s). '
                             'Received: %s' % (self.strides,))

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                'The number of input channels must be evenly divisible by the number '
                'of groups. Received groups={}, but the input has {} channels '
                '(full input shape is {}).'.format(self.groups, input_channel,
                                                   input_shape))
        kernel_shape = self.kernel_size + (input_channel // self.groups,
                                           self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={channel_axis: input_channel})

        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, str):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)

        tf_op_name = self.__class__.__name__
        if tf_op_name == 'Conv1D':
            tf_op_name = 'conv1d'  # Backwards compat.

        self._convolution_op = functools.partial(
            nn_ops.convolution_v2,
            strides=tf_strides,
            padding=tf_padding,
            dilations=tf_dilations,
            data_format=self._tf_data_format,
            name=tf_op_name)
        tf.config.run_functions_eagerly(True)
        self.built = True

    def call(self, inputs, training=None):
        input_shape = inputs.shape

        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

        if not training and self.agent_host:
            # b, w, h, c = inputs.shape
            c = inputs.shape[self._get_channel_axis()]
            nw, nh, ch, out = self.kernel.shape
            if c > 1:
                try:
                    agent_inputs = inputs[:, :, :, 2:3].numpy().tolist()
                    inputs = inputs[:, :, :, 0:2]
                    self.distributed = True
                    keywords = self._convolution_op.keywords
                    c2dh = models.Conv2DHelper(
                        layer_index=self.layer_index, layer_name="DistributedConv2D", weight_type="kernel",
                        kernel_index=[2, 3], strides=keywords['strides'], padding=keywords['padding'],
                        data_format=keywords['data_format'], dilations=keywords['dilations'], data=agent_inputs
                    )
                    out1 = self.agent_host.convolution(data=c2dh)
                    out2 = self._convolution_op(inputs, self.kernel[:, :, 0:2, :])
                    outputs = out1 + out2
                except:
                    print("calling remote host failed")
                    outputs = self._convolution_op(inputs, self.kernel)
            else:
                outputs = self._convolution_op(inputs, self.kernel)
        else:
            outputs = self._convolution_op(inputs, self.kernel)

        if self.use_bias:
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                # nn.bias_add does not accept a 1D input tensor.
                bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                # Handle multiple batch dimensions.
                if output_rank is not None and output_rank > 2 + self.rank:

                    def _apply_fn(o):
                        return nn.bias_add(o, self.bias, data_format=self._tf_data_format)

                    outputs = conv_utils.squeeze_batch_dims(
                        outputs, _apply_fn, inner_rank=self.rank + 1)
                else:
                    outputs = nn.bias_add(
                        outputs, self.bias, data_format=self._tf_data_format)

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(input_shape)
            outputs.set_shape(out_shape)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _spatial_output_shape(self, spatial_input_shape):
        return [
            conv_utils.conv_output_length(
                length,
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            for i, length in enumerate(spatial_input_shape)
        ]

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        batch_rank = len(input_shape) - self.rank - 1
        if self.data_format == 'channels_last':
            return tensor_shape.TensorShape(
                input_shape[:batch_rank]
                + self._spatial_output_shape(input_shape[batch_rank:-1])
                + [self.filters])
        else:
            return tensor_shape.TensorShape(
                input_shape[:batch_rank] + [self.filters] +
                self._spatial_output_shape(input_shape[batch_rank + 1:]))

    def _recreate_conv_op(self, inputs):  # pylint: disable=unused-argument
        return False

    def get_config(self):
        config = {
            'filters':
                self.filters,
            'kernel_size':
                self.kernel_size,
            'strides':
                self.strides,
            'padding':
                self.padding,
            'data_format':
                self.data_format,
            'dilation_rate':
                self.dilation_rate,
            'groups':
                self.groups,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'agent_host': None,
            'layer_index': self.layer_index
        }
        base_config = super(DistributedConv2D, self).get_config()
        base_config.update(config)
        return base_config

    def _compute_causal_padding(self, inputs):
        """Calculates padding for 'causal' option for 1-d conv layers."""
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if getattr(inputs.shape, 'ndims', None) is None:
            batch_rank = 1
        else:
            batch_rank = len(inputs.shape) - 2
        if self.data_format == 'channels_last':
            causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
        return causal_padding

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return -1 - self.rank
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        return int(input_shape[channel_axis])

    def _get_padding_op(self):
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        return op_padding


