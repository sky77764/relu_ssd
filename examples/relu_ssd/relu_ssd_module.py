from __future__ import print_function
import caffe
from caffe import layers as L
from caffe.model_libs import *
from google.protobuf import text_format

import math
import os
import shutil
import stat
import subprocess
import sys


def Res3Block(net, from_layer, deconv_layer, block_name,
              use_branch, branch_param):
    res_layer = []
    if use_branch[0]:
        branch1 = ResBranch(net, from_layer, block_name, "branch1", branch_param[0])
        res_layer.append(net[branch1])
    else:
        res_layer.append(net[from_layer])

    if use_branch[1]:
        branch2 = ResBranch(net, from_layer, block_name, "branch2", branch_param[1])
        res_layer.append(net[branch2])

    if use_branch[2]:
        branch3 = ResBranch(net, deconv_layer, block_name, "branch3", branch_param[2])
        res_layer.append(net[branch3])

    res_name = 'res{}'.format(block_name)

    if len(res_layer) != 1:
        net[res_name] = L.Eltwise(*res_layer)
        relu_name = '{}_relu'.format(res_name)
        net[relu_name] = L.ReLU(net[res_name], in_place=True)
    else:
        relu_name = '{}_relu'.format(res_name)
        net[relu_name] = L.ReLU(res_layer[0], in_place=True)

    return relu_name


def DeconvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
                  kernel_size, pad, stride, dilation=1, use_scale=True, lr_mult=1,
                  conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn',
                  scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias',
                  **bn_params):
    if use_bn:
        # parameters for convolution layer with batchnorm.
        kwargs = {
            'param': [dict(lr_mult=lr_mult, decay_mult=1)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_term': False,
        }
        eps = bn_params.get('eps', 0.001)
        moving_average_fraction = bn_params.get('moving_average_fraction', 0.999)
        use_global_stats = bn_params.get('use_global_stats', False)
        # parameters for batchnorm layer.
        bn_kwargs = {
            'param': [
                dict(lr_mult=0, decay_mult=0),
                dict(lr_mult=0, decay_mult=0),
                dict(lr_mult=0, decay_mult=0)],
            'eps': eps,
            'moving_average_fraction': moving_average_fraction,
        }
        bn_lr_mult = lr_mult
        if use_global_stats:
            # only specify if use_global_stats is explicitly provided;
            # otherwise, use_global_stats_ = this->phase_ == TEST;
            bn_kwargs = {
                'param': [
                    dict(lr_mult=0, decay_mult=0),
                    dict(lr_mult=0, decay_mult=0),
                    dict(lr_mult=0, decay_mult=0)],
                'eps': eps,
                'use_global_stats': use_global_stats,
            }
            # not updating scale/bias parameters
            bn_lr_mult = 0
        # parameters for scale bias layer after batchnorm.
        if use_scale:
            sb_kwargs = {
                'bias_term': True,
                'param': [
                    dict(lr_mult=bn_lr_mult, decay_mult=0),
                    dict(lr_mult=bn_lr_mult, decay_mult=0)],
                'filler': dict(type='constant', value=1.0),
                'bias_filler': dict(type='constant', value=0.0),
            }
        else:
            bias_kwargs = {
                'param': [dict(lr_mult=bn_lr_mult, decay_mult=0)],
                'filler': dict(type='constant', value=0.0),
            }
    else:
        kwargs = {
            'param': [
                dict(lr_mult=lr_mult, decay_mult=1),
                dict(lr_mult=2 * lr_mult, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)
        }

    conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
    [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
    [pad_h, pad_w] = UnpackVariable(pad, 2)
    [stride_h, stride_w] = UnpackVariable(stride, 2)
    if kernel_h == kernel_w:
        net[conv_name] = L.Deconvolution(net[from_layer],
                                         convolution_param=dict(num_output=num_output,
                                                                kernel_size=kernel_h, pad=pad_h, stride=stride_h,
                                                                weight_filler=dict(type='gaussian', std=0.01),
                                                                bias_term=False, ),
                                         param=[dict(lr_mult=lr_mult, decay_mult=1)])
    else:
        net[conv_name] = L.Deconvolution(net[from_layer],
                                         convolution_param=dict(num_output=num_output,
                                                                kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h,
                                                                pad_w=pad_w,
                                                                stride_h=stride_h, stride_w=stride_w, **kwargs))
    if dilation > 1:
        net.update(conv_name, {'dilation': dilation})
    if use_bn:
        bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
        net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
        if use_scale:
            sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
            net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
        else:
            bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
            net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
    if use_relu:
        relu_name = '{}_relu'.format(conv_name)
        net[relu_name] = L.ReLU(net[conv_name], in_place=True)


def ResBranch(net, from_layer, block_name, branch_prefix, layer_param, **bn_params):
    conv_prefix = 'res{}_'.format(block_name)
    conv_postfix = ''
    bn_prefix = 'bn{}_'.format(block_name)
    bn_postfix = ''
    scale_prefix = 'scale{}_'.format(block_name)
    scale_postfix = ''
    use_scale = True

    num_layers = len(layer_param)

    if num_layers != 1:
        name_postfix = ['a', 'b', 'c', 'd', 'e']
    else:
        name_postfix = ['']
    id = 0
    out_name = from_layer

    for param in layer_param:

        branch_name = branch_prefix + name_postfix[id]
        id += 1

        num_output = param['out']
        kernel_size = param['kernel_size']
        pad = param['pad']
        stride = param['stride']
        use_relu = id is not num_layers

        if param['name'] == 'Convolution':
            functor = ConvBNLayer
        elif param['name'] == 'Deconvolution':
            functor = DeconvBNLayer

        functor(net, out_name, branch_name, use_bn=True, use_relu=use_relu,
                num_output=num_output, kernel_size=kernel_size, pad=pad, stride=stride, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_params)

        out_name = '{}{}'.format(conv_prefix, branch_name)

    return out_name


# Add extra layers on top of a "base" network (e.g. VGGNet or Inception).
def AddExtraLayers(net, use_batchnorm=True, lr_mult=1, use_conv10=False):
    use_relu = True

    # Add additional convolutional layers.
    # 19 x 19
    from_layer = net.keys()[-1]

    # TODO(weiliu89): Construct the name using the last layer to avoid duplication.
    # 10 x 10
    out_layer = "conv6_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1,
                lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv6_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2,
                lr_mult=lr_mult)

    # 5 x 5
    from_layer = out_layer
    out_layer = "conv7_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
                lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv7_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,
                lr_mult=lr_mult)

    # 3 x 3
    from_layer = out_layer
    out_layer = "conv8_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
                lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv8_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
                lr_mult=lr_mult)

    # 1 x 1
    from_layer = out_layer
    out_layer = "conv9_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
                lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv9_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
                lr_mult=lr_mult)

    if use_conv10:
        from_layer = out_layer
        out_layer = "conv10_1"
        ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
                    lr_mult=lr_mult)

        from_layer = out_layer
        out_layer = "conv10_2"
        ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 4, 1, 1,
                    lr_mult=lr_mult)

    return net


def AddResidualLayers(net, out_dim,
                      use_branch1, use_branch2, use_branch3, use_conv3_3=False):
    out_layers = []

    branch1_type1 = [{'name': "Convolution", 'out': out_dim, 'kernel_size': 1, 'pad': 0, 'stride': 1, }]

    branch2_type1 = [
        {'name': "Convolution", 'out': out_dim / 4, 'kernel_size': 1, 'pad': 0, 'stride': 1, },
        {'name': "Convolution", 'out': out_dim / 4, 'kernel_size': 3, 'pad': 1, 'stride': 1, },
        {'name': "Convolution", 'out': out_dim, 'kernel_size': 1, 'pad': 0, 'stride': 1, }
    ]

    branch2_type2 = [
        {'name': "Convolution", 'out': out_dim, 'kernel_size': 3, 'pad': 1, 'stride': 1, },
        {'name': "Convolution", 'out': out_dim, 'kernel_size': 3, 'pad': 1, 'stride': 1, }
    ]

    branch3_type1 = [
        {'name': "Convolution", 'out': out_dim / 4, 'kernel_size': 3, 'pad': 1, 'stride': 1, },
        {'name': "Deconvolution", 'out': out_dim / 4, },
        {'name': "Convolution", 'out': out_dim, 'kernel_size': 1, 'pad': 0, 'stride': 1, }
    ]
    branch3_type2 = [
        {'name': "Deconvolution", 'out': out_dim / 4, },
        {'name': "Convolution", 'out': out_dim / 4, 'kernel_size': 3, 'pad': 1, 'stride': 1, },
        {'name': "Convolution", 'out': out_dim, 'kernel_size': 1, 'pad': 0, 'stride': 1, }
    ]

    branch1_param = branch1_type1
    branch2_param = branch2_type2
    branch3_param = branch3_type2

    deconv_param = branch3_param[0]

    use_branch = [use_branch1, use_branch2, use_branch3]
    branch_param = [branch1_param, branch2_param, branch3_param]

    out_layers = []

    # 3_3
    if use_conv3_3:
        from_layer = "conv3_3"
        deconv_layer = "conv4_3"
        block_name = "3_3"
        deconv_param.update({'kernel_size': 2, 'pad': 0, 'stride': 2, })
        out_layer = Res3Block(net, from_layer, deconv_layer, block_name, use_branch, branch_param)
        out_layers.append(out_layer)

    from_layer = "conv4_3"
    deconv_layer = "fc7"
    block_name = "4_3"
    deconv_param.update({'kernel_size': 2, 'pad': 0, 'stride': 2, })
    out_layer = Res3Block(net, from_layer, deconv_layer, block_name, use_branch, branch_param)
    out_layers.append(out_layer)

    from_layer = "fc7"
    deconv_layer = "conv6_2"
    block_name = "fc7"
    deconv_param.update({'kernel_size': 3, 'pad': 1, 'stride': 2, })
    out_layer = Res3Block(net, from_layer, deconv_layer, block_name, use_branch, branch_param)
    out_layers.append(out_layer)

    from_layer = "conv6_2"
    deconv_layer = "conv7_2"
    block_name = "6_2"
    deconv_param.update({'kernel_size': 2, 'pad': 0, 'stride': 2, })
    out_layer = Res3Block(net, from_layer, deconv_layer, block_name, use_branch, branch_param)
    out_layers.append(out_layer)

    from_layer = "conv7_2"
    deconv_layer = "conv8_2"
    block_name = "7_2"
    deconv_param.update({'kernel_size': 3, 'pad': 0, 'stride': 1, })
    out_layer = Res3Block(net, from_layer, deconv_layer, block_name, use_branch, branch_param)
    out_layers.append(out_layer)

    from_layer = "conv8_2"
    deconv_layer = "conv9_2"
    block_name = "8_2"
    deconv_param.update({'kernel_size': 3, 'pad': 0, 'stride': 1, })
    out_layer = Res3Block(net, from_layer, deconv_layer, block_name, use_branch, branch_param)
    out_layers.append(out_layer)

    from_layer = "conv9_2"
    block_name = "9_2"
    out_layer = Res3Block(net, from_layer, deconv_layer, block_name, [use_branch1, use_branch2, False], branch_param)
    out_layers.append(out_layer)

    return out_layers


def CreateUnifiedPredictionHead(net, data_layer="data", num_classes=[], from_layers=[],
                                use_objectness=False, normalizations=[], use_batchnorm=True, lr_mult=1,
                                use_scale=True, min_sizes=[], max_sizes=[], prior_variance=[0.1],
                                aspect_ratios=[], steps=[], img_height=0, img_width=0, share_location=True,
                                flip=True, clip=True, offset=0.5, inter_layer_depth=[], kernel_size=1, pad=0,
                                conf_postfix='', loc_postfix='', **bn_param):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    if aspect_ratios:
        assert len(from_layers) == len(aspect_ratios), "from_layers and aspect_ratios should have same length"
    if steps:
        assert len(from_layers) == len(steps), "from_layers and steps should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"
    if inter_layer_depth:
        assert len(from_layers) == len(inter_layer_depth), "from_layers and inter_layer_depth should have same length"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    objectness_layers = []

    loc_args = {
        'param': [
            dict(name='loc_p1', lr_mult=lr_mult, decay_mult=1),
            dict(name='loc_p2', lr_mult=2 * lr_mult, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
    }

    conf_args = {
        'param': [
            dict(name='conf_p1', lr_mult=lr_mult, decay_mult=1),
            dict(name='conf_p2', lr_mult=2 * lr_mult, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
    }

    if flip:
        num_priors_per_location = 6
    else:
        num_priors_per_location = 3

    for i in range(0, num):
        from_layer = from_layers[i]

        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)

        # Create location prediction layer.
        net[name] = L.Convolution(net[from_layer], num_output=num_priors_per_location * 4,
                                  pad=1, kernel_size=3, stride=1, **loc_args)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create confidence prediction layer.
        name = "{}_mbox_conf{}".format(from_layer, conf_postfix)
        net[name] = L.Convolution(net[from_layer], num_output=num_priors_per_location * num_classes,
                                  pad=1, kernel_size=3, stride=1, **conf_args)

        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_size,
                               clip=clip, variance=prior_variance, offset=offset)
        if max_size:
            net.update(name, {'max_size': max_size})
        if aspect_ratio:
            net.update(name, {'aspect_ratio': aspect_ratio, 'flip': flip})
        if step:
            net.update(name, {'step': step})
        if img_height != 0 and img_width != 0:
            if img_height == img_width:
                net.update(name, {'img_size': img_height})
            else:
                net.update(name, {'img_h': img_height, 'img_w': img_width})
        priorbox_layers.append(net[name])

        # Create objectness prediction layer.
        if use_objectness:
            name = "{}_mbox_objectness".format(from_layer)
            num_obj_output = num_priors_per_location * 2;
            ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                        num_output=num_obj_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            objectness_layers.append(net[flatten_name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_loc"
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_conf"
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_priorbox"
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    if use_objectness:
        name = "mbox_objectness"
        net[name] = L.Concat(*objectness_layers, axis=1)
        mbox_layers.append(net[name])

    return mbox_layers


def Train(param, py_file):
    ### Modify the following parameters accordingly ###
    # The directory which contains the caffe code.
    # We assume you are running the script at the CAFFE_ROOT.
    caffe_root = os.getcwd()

    # Set true if you want to start training right after generating all files.
    run_soon = True
    # Set true if you want to load from most recently saved snapshot.
    # Otherwise, we will load from the pretrain_model defined below.
    resume_training = True
    # If true, Remove old model files.
    remove_old_models = False

    data_set = param['data_set']
    # The database file for training data. Created by data/VOC0712/create_data.sh
    train_data = "examples/{}/{}_trainval_lmdb".format(data_set, data_set)
    # The database file for testing data. Created by data/VOC0712/create_data.sh
    test_data = "examples/{}/{}_test_lmdb".format(data_set, data_set)

    # Specify the batch sampler.
    input_dim = param['input_dim']
    resize_width = input_dim
    resize_height = input_dim
    resize = "{}x{}".format(resize_width, resize_height)
    batch_sampler = [
        {
            'sampler': {
            },
            'max_trials': 1,
            'max_sample': 1,
        },
        {
            'sampler': {
                'min_scale': 0.3,
                'max_scale': 1.0,
                'min_aspect_ratio': 0.5,
                'max_aspect_ratio': 2.0,
            },
            'sample_constraint': {
                'min_jaccard_overlap': 0.1,
            },
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'sampler': {
                'min_scale': 0.3,
                'max_scale': 1.0,
                'min_aspect_ratio': 0.5,
                'max_aspect_ratio': 2.0,
            },
            'sample_constraint': {
                'min_jaccard_overlap': 0.3,
            },
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'sampler': {
                'min_scale': 0.3,
                'max_scale': 1.0,
                'min_aspect_ratio': 0.5,
                'max_aspect_ratio': 2.0,
            },
            'sample_constraint': {
                'min_jaccard_overlap': 0.5,
            },
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'sampler': {
                'min_scale': 0.3,
                'max_scale': 1.0,
                'min_aspect_ratio': 0.5,
                'max_aspect_ratio': 2.0,
            },
            'sample_constraint': {
                'min_jaccard_overlap': 0.7,
            },
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'sampler': {
                'min_scale': 0.3,
                'max_scale': 1.0,
                'min_aspect_ratio': 0.5,
                'max_aspect_ratio': 2.0,
            },
            'sample_constraint': {
                'min_jaccard_overlap': 0.9,
            },
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'sampler': {
                'min_scale': 0.3,
                'max_scale': 1.0,
                'min_aspect_ratio': 0.5,
                'max_aspect_ratio': 2.0,
            },
            'sample_constraint': {
                'max_jaccard_overlap': 1.0,
            },
            'max_trials': 50,
            'max_sample': 1,
        },
    ]
    train_transform_param = {
        'mirror': True,
        'mean_value': [104, 117, 123],
        'resize_param': {
            'prob': 1,
            'resize_mode': P.Resize.WARP,
            'height': resize_height,
            'width': resize_width,
            'interp_mode': [
                P.Resize.LINEAR,
                P.Resize.AREA,
                P.Resize.NEAREST,
                P.Resize.CUBIC,
                P.Resize.LANCZOS4,
            ],
        },
        'distort_param': {
            'brightness_prob': 0.5,
            'brightness_delta': 32,
            'contrast_prob': 0.5,
            'contrast_lower': 0.5,
            'contrast_upper': 1.5,
            'hue_prob': 0.5,
            'hue_delta': 18,
            'saturation_prob': 0.5,
            'saturation_lower': 0.5,
            'saturation_upper': 1.5,
            'random_order_prob': 0.0,
        },
        'expand_param': {
            'prob': 0.5,
            'max_expand_ratio': 4.0,
        },
        'emit_constraint': {
            'emit_type': caffe_pb2.EmitConstraint.CENTER,
        }
    }
    test_transform_param = {
        'mean_value': [104, 117, 123],
        'resize_param': {
            'prob': 1,
            'resize_mode': P.Resize.WARP,
            'height': resize_height,
            'width': resize_width,
            'interp_mode': [P.Resize.LINEAR],
        },
    }

    # If true, use batch norm for all newly added layers.
    # Currently only the non batch norm version has been tested.
    use_batchnorm = param['use_batchnorm']
    lr_mult = 1
    # Use different initial learning rate.
    if use_batchnorm:
        base_lr = 0.0004
    else:
        # A learning rate for batch_size = 1, num_gpus = 1.
        base_lr = 0.00004

    residual_feature_depth = param['residual_feature_depth']

    # Modify the job name if you want.
    job_name = param['net_name'] + "_{}_res{}{}".format(resize, residual_feature_depth,
                                                        "_Conv3" if param['use_conv3_3'] else "")
    # The name of the model. Modify it if you want.
    model_name = "VGG_{}_{}".format(data_set, job_name)

    # Directory which stores the model .prototxt file.
    save_dir = "models/VGGNet/{}/{}".format(data_set, job_name)
    # Directory which stores the snapshot of models.
    snapshot_dir = "models/VGGNet/{}/{}".format(data_set, job_name)
    # Directory which stores the job script and log file.
    job_dir = "jobs/VGGNet/{}/{}".format(data_set, job_name)
    # Directory which stores the detection results.
    output_result_dir = "{}/data/VOCdevkit/results/{}/{}/Main".format(os.environ['HOME'], data_set, job_name)

    # model definition files.
    train_net_file = "{}/train.prototxt".format(save_dir)
    test_net_file = "{}/test.prototxt".format(save_dir)
    deploy_net_file = "{}/deploy.prototxt".format(save_dir)
    solver_file = "{}/solver.prototxt".format(save_dir)
    # snapshot prefix.
    snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
    # job script path.
    job_file = "{}/{}.sh".format(job_dir, model_name)

    # Stores the test image names and sizes. Created by data/VOC0712/create_list.sh
    name_size_file = "data/{}/test_name_size.txt".format(data_set)
    # The pretrained model. We use the Fully convolutional reduced (atrous) VGGNet.
    pretrain_model = param['pretrain_model']
    # Stores LabelMapItem.
    label_map_file = "data/{}/labelmap_voc.prototxt".format(data_set)

    # MultiBoxLoss parameters.
    if data_set is "VOC0712" or "VOC0712plus":
        num_classes = 21

    share_location = True
    background_label_id = 0
    train_on_diff_gt = True
    normalization_mode = P.Loss.VALID
    code_type = P.PriorBox.CENTER_SIZE
    ignore_cross_boundary_bbox = False
    mining_type = P.MultiBoxLoss.MAX_NEGATIVE
    neg_pos_ratio = 3.
    loc_weight = (neg_pos_ratio + 1.) / 4.
    multibox_loss_param = {
        'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
        'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
        'loc_weight': loc_weight,
        'num_classes': num_classes,
        'share_location': share_location,
        'match_type': P.MultiBoxLoss.PER_PREDICTION,
        'overlap_threshold': 0.5,
        'use_prior_for_matching': True,
        'background_label_id': background_label_id,
        'use_difficult_gt': train_on_diff_gt,
        'mining_type': mining_type,
        'neg_pos_ratio': neg_pos_ratio,
        'neg_overlap': 0.5,
        'code_type': code_type,
        'ignore_cross_boundary_bbox': ignore_cross_boundary_bbox,
    }
    loss_param = {
        'normalization': normalization_mode,
    }

    # parameters for generating priors.
    # minimum dimension of input image
    min_dim = input_dim

    use_conv3_3 = param['use_conv3_3']
    resnet_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']

    if min_dim == 512:
        use_conv10 = True
        resnet_source_layers += ['conv10_2']
    else:
        use_conv10 = False

    # 300x300
    ## conv4_3 ==> 38 x 38
    ## fc7 ==> 19 x 19
    ## conv6_2 ==> 10 x 10
    ## conv7_2 ==> 5 x 5
    ## conv8_2 ==> 3 x 3
    ## conv9_2 ==> 1 x 1
    if input_dim == 300:
        # in percent %
        min_ratio = 20
        max_ratio = 90
        step = int(math.floor((max_ratio - min_ratio) / (len(resnet_source_layers) - 2)))
        min_sizes = []
        max_sizes = []
        for ratio in xrange(min_ratio, max_ratio + 1, step):
            min_sizes.append(min_dim * ratio / 100.)
            max_sizes.append(min_dim * (ratio + step) / 100.)
        min_sizes = [min_dim * 10 / 100.] + min_sizes
        max_sizes = [min_dim * 20 / 100.] + max_sizes

        if use_conv3_3:
            min_sizes = [min_dim * 5 / 100.] + min_sizes
            max_sizes = [min_dim * 10 / 100.] + max_sizes

        steps = [8, 16, 32, 64, 100, 300]

    # 512x512
    ## conv4_3 ==> 64 x 64
    ## fc7 ==> 32 x 32
    ## conv6_2 ==> 16 x 16
    ## conv7_2 ==> 8 x 8
    ## conv8_2 ==> 4 x 4
    ## conv9_2 ==> 2 x 2
    ## conv10_2 ==> 1 x 1
    elif input_dim == 512:

        # in percent %
        min_ratio = 15
        max_ratio = 90
        step = int(math.floor((max_ratio - min_ratio) / (len(resnet_source_layers) - 2)))
        min_sizes = []
        max_sizes = []
        for ratio in xrange(min_ratio, max_ratio + 1, step):
            min_sizes.append(min_dim * ratio / 100.)
            max_sizes.append(min_dim * (ratio + step) / 100.)
        min_sizes = [min_dim * 7 / 100.] + min_sizes
        max_sizes = [min_dim * 15 / 100.] + max_sizes

        if use_conv3_3:
            min_sizes = [min_dim * 3 / 100.] + min_sizes
            max_sizes = [min_dim * 7 / 100.] + max_sizes

        steps = [8, 16, 32, 64, 128, 256, 512]

    if use_conv3_3:
        resnet_source_layers = ['conv3_3'] + resnet_source_layers
        steps = [4] + steps

    aspect_ratios = [[2, 3] for i in range(0, len(resnet_source_layers))]
    normalizations = [-1 for i in range(0, len(resnet_source_layers))]

    # variance used to encode/decode prior bboxes.
    if code_type == P.PriorBox.CENTER_SIZE:
        prior_variance = [0.1, 0.1, 0.2, 0.2]
    else:
        prior_variance = [0.1]
    flip = True
    clip = False

    # Solver parameters.
    # Defining which GPUs to use.
    gpus = param['gpus']
    gpulist = gpus.split(",")
    num_gpus = len(gpulist)

    # Divide the mini-batch to different GPUs.
    batch_size = param['batch_size']
    accum_batch_size = param['accum_batch_size']
    iter_size = accum_batch_size / batch_size
    solver_mode = P.Solver.CPU
    device_id = 0
    batch_size_per_device = batch_size
    if num_gpus > 0:
        batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
        iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
        solver_mode = P.Solver.GPU
        device_id = int(gpulist[0])

    if normalization_mode == P.Loss.NONE:
        base_lr /= batch_size_per_device
    elif normalization_mode == P.Loss.VALID:
        base_lr *= 25. / loc_weight
    elif normalization_mode == P.Loss.FULL:
        # Roughly there are 2000 prior bboxes per image.
        # TODO(weiliu89): Estimate the exact # of priors.
        base_lr *= 2000.

    # Evaluate on whole test set.
    num_test_image = param['num_test_image']
    test_batch_size = param['test_batch_size']
    # Ideally test_batch_size should be divisible by num_test_image,
    # otherwise mAP will be slightly off the true value.
    test_iter = int(math.ceil(float(num_test_image) / test_batch_size))

    solver_param = {
        # Train parameters
        'base_lr': base_lr,
        'weight_decay': 0.0005,
        'lr_policy': "multistep",
        'stepvalue': [80000, 100000, 120000, 140000],
        'gamma': 0.1,
        'momentum': 0.9,
        'iter_size': iter_size,
        'max_iter': 160000,
        'snapshot': 5000,
        'display': 10,
        'average_loss': 10,
        'type': "SGD",
        'solver_mode': solver_mode,
        'device_id': device_id,
        'debug_info': False,
        'snapshot_after_train': True,
        # Test parameters
        'test_iter': [test_iter],
        'test_interval': 5000,
        'eval_type': "detection",
        'ap_version': "11point",
        'test_initialization': False,
    }

    solver_param.update(param['solver_param'])

    # parameters for generating detection output.
    det_out_param = {
        'num_classes': num_classes,
        'share_location': share_location,
        'background_label_id': background_label_id,
        'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
        'save_output_param': {
            'output_directory': output_result_dir,
            'output_name_prefix': "comp4_det_test_",
            'output_format': "VOC",
            'label_map_file': label_map_file,
            'name_size_file': name_size_file,
            'num_test_image': num_test_image,
        },
        'keep_top_k': 200,
        'confidence_threshold': 0.01,
        'code_type': code_type,
    }

    det_out_param.update(param['det_out_param'])

    # parameters for evaluating detection results.
    det_eval_param = {
        'num_classes': num_classes,
        'background_label_id': background_label_id,
        'overlap_threshold': 0.5,
        'evaluate_difficult_gt': False,
        'name_size_file': name_size_file,
    }

    ### Hopefully you don't need to change the following ###
    # Check file.
    check_if_exist(train_data)
    check_if_exist(test_data)
    check_if_exist(label_map_file)
    check_if_exist(pretrain_model)
    make_if_not_exist(save_dir)
    make_if_not_exist(job_dir)
    make_if_not_exist(snapshot_dir)

    # Create train net.
    net = caffe.NetSpec()
    net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=batch_size_per_device,
                                                   train=True, output_label=True, label_map_file=label_map_file,
                                                   transform_param=train_transform_param, batch_sampler=batch_sampler)

    VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
               dropout=False)

    AddExtraLayers(net, use_batchnorm, lr_mult=lr_mult, use_conv10=use_conv10)

    use_branch2 = param['use_res_branch2']
    use_branch3 = param['use_res_deconv']

    mbox_source_layers = AddResidualLayers(net, residual_feature_depth, True, use_branch2, use_branch3, use_conv3_3)

    if not param["use_unified_prediction"]:
        mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
                                         use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
                                         aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
                                         num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
                                         prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult)
    else:
        mbox_layers = CreateUnifiedPredictionHead(net, data_layer='data', from_layers=mbox_source_layers,
                                                  use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
                                                  aspect_ratios=aspect_ratios, steps=steps,
                                                  normalizations=normalizations,
                                                  num_classes=num_classes, share_location=share_location, flip=flip,
                                                  clip=clip,
                                                  prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult)

    # Create the MultiBoxLossLayer.
    name = "mbox_loss"
    mbox_layers.append(net.label)
    net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
                               loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                               propagate_down=[True, True, False, False])

    with open(train_net_file, 'w') as f:
        print('name: "{}_train"'.format(model_name), file=f)
        print(net.to_proto(), file=f)
    shutil.copy(train_net_file, job_dir)

    # Create test net.
    net = caffe.NetSpec()
    net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=test_batch_size,
                                                   train=False, output_label=True, label_map_file=label_map_file,
                                                   transform_param=test_transform_param)

    VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
               dropout=False)

    AddExtraLayers(net, use_batchnorm, lr_mult=lr_mult, use_conv10=use_conv10)

    mbox_source_layers = AddResidualLayers(net, residual_feature_depth, True, use_branch2, use_branch3, use_conv3_3)

    if not param["use_unified_prediction"]:
        mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
                                         use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
                                         aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
                                         num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
                                         prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult)
    else:
        mbox_layers = CreateUnifiedPredictionHead(net, data_layer='data', from_layers=mbox_source_layers,
                                                  use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
                                                  aspect_ratios=aspect_ratios, steps=steps,
                                                  normalizations=normalizations,
                                                  num_classes=num_classes, share_location=share_location, flip=flip,
                                                  clip=clip,
                                                  prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult)

    conf_name = "mbox_conf"
    if multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.SOFTMAX:
        reshape_name = "{}_reshape".format(conf_name)
        net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
        softmax_name = "{}_softmax".format(conf_name)
        net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
        flatten_name = "{}_flatten".format(conf_name)
        net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
        mbox_layers[1] = net[flatten_name]
    elif multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.LOGISTIC:
        sigmoid_name = "{}_sigmoid".format(conf_name)
        net[sigmoid_name] = L.Sigmoid(net[conf_name])
        mbox_layers[1] = net[sigmoid_name]

    net.detection_out = L.DetectionOutput(*mbox_layers,
                                          detection_output_param=det_out_param,
                                          include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
                                             detection_evaluate_param=det_eval_param,
                                             include=dict(phase=caffe_pb2.Phase.Value('TEST')))

    with open(test_net_file, 'w') as f:
        print('name: "{}_test"'.format(model_name), file=f)
        print(net.to_proto(), file=f)
    shutil.copy(test_net_file, job_dir)

    # Create deploy net.
    # Remove the first and last layer from test net.
    deploy_net = net
    with open(deploy_net_file, 'w') as f:
        net_param = deploy_net.to_proto()
        # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
        del net_param.layer[0]
        del net_param.layer[-1]
        net_param.name = '{}_deploy'.format(model_name)
        net_param.input.extend(['data'])
        net_param.input_shape.extend([
            caffe_pb2.BlobShape(dim=[1, 3, resize_height, resize_width])])
        print(net_param, file=f)
    shutil.copy(deploy_net_file, job_dir)

    # Create solver.
    solver = caffe_pb2.SolverParameter(
        train_net=train_net_file,
        test_net=[test_net_file],
        snapshot_prefix=snapshot_prefix,
        **solver_param)

    with open(solver_file, 'w') as f:
        print(solver, file=f)
    shutil.copy(solver_file, job_dir)

    max_iter = 0
    # Find most recent snapshot.
    for file in os.listdir(snapshot_dir):
        if file.endswith(".solverstate"):
            basename = os.path.splitext(file)[0]
            iter = int(basename.split("{}_iter_".format(model_name))[1])
            if iter > max_iter:
                max_iter = iter

    train_src_param = '--weights="{}" \\\n'.format(pretrain_model)
    if resume_training:
        if max_iter > 0:
            train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)

    if remove_old_models:
        # Remove any snapshots smaller than max_iter.
        for file in os.listdir(snapshot_dir):
            if file.endswith(".solverstate"):
                basename = os.path.splitext(file)[0]
                iter = int(basename.split("{}_iter_".format(model_name))[1])
                if max_iter > iter:
                    os.remove("{}/{}".format(snapshot_dir, file))
            if file.endswith(".caffemodel"):
                basename = os.path.splitext(file)[0]
                iter = int(basename.split("{}_iter_".format(model_name))[1])
                if max_iter > iter:
                    os.remove("{}/{}".format(snapshot_dir, file))

    # Create job file.
    with open(job_file, 'w') as f:
        f.write('cd {}\n'.format(caffe_root))
        f.write('./build/tools/caffe train \\\n')
        f.write('--solver="{}" \\\n'.format(solver_file))
        f.write(train_src_param)
        if solver_param['solver_mode'] == P.Solver.GPU:
            f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(gpus, job_dir, model_name))
        else:
            f.write('2>&1 | tee {}/{}.log\n'.format(job_dir, model_name))

    # Copy the python script to job_dir.
    shutil.copy(py_file, job_dir)

    # Run the job.
    os.chmod(job_file, stat.S_IRWXU)
    if run_soon:
        subprocess.call(job_file, shell=True)