#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

__all__ = [
    "SE_ResNeXt", "SE_ResNeXt50_32x4d", "SE_ResNeXt101_32x4d",
    "SE_ResNeXt152_32x4d"
]


class SE_ResNeXt():
    def __init__(self, layers=50):
        self.layers = layers

    def net(self, input, class_dim=1000, data_format="NCHW"):
        layers = self.layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)
        if layers == 50:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 6, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu',
                name='conv1',
                data_format=data_format)
            conv = fluid.layers.pool2d(
                input=conv,
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max',
                #use_cudnn=True,
                use_cudnn=False,
                data_format=data_format)
        elif layers == 101:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 23, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu',
                name="conv1",
                data_format=data_format)
            conv = fluid.layers.pool2d(
                input=conv,
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max',
                #use_cudnn=True,
                use_cudnn=False,
                data_format=data_format)
        elif layers == 152:
            cardinality = 64
            reduction_ratio = 16
            depth = [3, 8, 36, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=3,
                stride=2,
                act='relu',
                name='conv1')
            conv = self.conv_bn_layer(
                input=conv,
                num_filters=64,
                filter_size=3,
                stride=1,
                act='relu',
                name='conv2')
            conv = self.conv_bn_layer(
                input=conv,
                num_filters=128,
                filter_size=3,
                stride=1,
                act='relu',
                name='conv3')
            conv = fluid.layers.pool2d(
                input=conv, pool_size=3, pool_stride=2, pool_padding=1, \
                pool_type='max', use_cudnn=False, data_format=data_format)
        n = 1 if layers == 50 or layers == 101 else 3
        for block in range(len(depth)):
            n += 1
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1,
                    cardinality=cardinality,
                    reduction_ratio=reduction_ratio,
                    name=str(n) + '_' + str(i + 1),
                    data_format=data_format)

        pool = fluid.layers.pool2d(
            input=conv, pool_type='avg', global_pooling=True, use_cudnn=False, data_format=data_format)
        drop = fluid.layers.dropout(x=pool, dropout_prob=0.5)
        stdv = 1.0 / math.sqrt(drop.shape[1] * 1.0)
        out = fluid.layers.fc(
            input=drop,
            size=class_dim,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name='fc6_weights'),
            bias_attr=ParamAttr(name='fc6_offset'))
        return out

    def shortcut(self, input, ch_out, stride, name, data_format):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            filter_size = 1
            return self.conv_bn_layer(
                input, ch_out, filter_size, stride, name='conv' + name + '_prj', data_format=data_format)
        else:
            return input

    def bottleneck_block(self,
                         input,
                         num_filters,
                         stride,
                         cardinality,
                         reduction_ratio,
                         data_format,
                         name=None):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name='conv' + name + '_x1',
            data_format=data_format)
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            groups=cardinality,
            act='relu',
            name='conv' + name + '_x2',
            data_format=data_format)
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 2,
            filter_size=1,
            act=None,
            name='conv' + name + '_x3',
            data_format=data_format)
        scale = self.squeeze_excitation(
            input=conv2,
            num_channels=num_filters * 2,
            reduction_ratio=reduction_ratio,
            name='fc' + name,
            data_format=data_format)

        short = self.shortcut(input, num_filters * 2, stride, name=name, data_format=data_format)

        return fluid.layers.elementwise_add(x=short, y=scale, act='relu')

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None,
                      data_format="NCHW"):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False,
            param_attr=ParamAttr(name=name + '_weights'),
            data_format=data_format)
        bn_name = name + "_bn"
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance',
            data_layout=data_format)

    def squeeze_excitation(self,
                           input,
                           num_channels,
                           reduction_ratio,
                           data_format,
                           name=None):
        pool = fluid.layers.pool2d(
            input=input, pool_type='avg', global_pooling=True, use_cudnn=False, data_format=data_format)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(
            input=pool,
            size=num_channels // reduction_ratio,
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_sqz_weights'),
            bias_attr=ParamAttr(name=name + '_sqz_offset'))
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(
            input=squeeze,
            size=num_channels,
            act='sigmoid',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_exc_weights'),
            bias_attr=ParamAttr(name=name + '_exc_offset'))
        input_in = fluid.layers.transpose(
            input, [0, 3, 1, 2]) if data_format == 'NHWC' else input
        input_in.stop_gradient = input.stop_gradient
        scale = fluid.layers.elementwise_mul(x=input_in, y=excitation, axis=0)
        scale_out = fluid.layers.transpose(
            scale, [0, 2, 3, 1]) if data_format == 'NHWC' else scale
        scale_out.stop_gradient = scale.stop_gradient
        return scale_out


def SE_ResNeXt50_32x4d():
    model = SE_ResNeXt(layers=50)
    return model


def SE_ResNeXt101_32x4d():
    model = SE_ResNeXt(layers=101)
    return model


def SE_ResNeXt152_32x4d():
    model = SE_ResNeXt(layers=152)
    return model
