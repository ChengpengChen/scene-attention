"""convert pytorch model (resnet50) to mxnet model
resnet50: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
weight: http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar

2018.08.20
"""

import os.path as osp
import importlib

import torch
# from torchvision import models
import mxnet as mx
import numpy as np

input_dim = (1, 3, 224, 224)


def convert(weight_file, network='resnet_ori', output_prefix='resnet50_mxnet', bn_data_layer=False):
    # bn_data: integrate the data processing into the bn_data layer

    assert osp.exists(weight_file), 'weight file not exist'
    assert network == 'resnet_ori', 'only resnet50 support now'
    # model_torch = models.__dict__[arch](num_classes=num_classes)
    checkpoint = torch.load(weight_file, map_location=lambda storage, loc: storage)

    symbol = importlib.import_module('symbol_' + network).get_symbol(bn_data_layer=bn_data_layer)
    arg_shapes, _, aux_shapes = symbol.infer_shape(data=tuple(input_dim))
    arg_names = symbol.list_arguments()
    aux_names = symbol.list_auxiliary_states()
    arg_shape_dic = dict(zip(arg_names, arg_shapes))
    aux_shape_dic = dict(zip(aux_names, aux_shapes))
    arg_params = {}
    aux_params = {}

    if bn_data_layer:
        mean_torch = np.array([0.485, 0.456, 0.406])
        std_torch = np.array([0.229, 0.224, 0.225])
        arg_params['bn_data_gamma'] = mx.nd.array([1., 1., 1.])
        arg_params['bn_data_beta'] = mx.nd.array([0., 0., 0.])
        aux_params['bn_data_moving_mean'] = mx.nd.array(255 * mean_torch)
        aux_params['bn_data_moving_var'] = mx.nd.array(np.square(255 * std_torch))
        print('done converted the ba_data layer')

    for k, v in checkpoint['state_dict'].items():
        k = k.replace('module.', '')
        k = k.replace('.', '_')
        k_mx = k
        if 'bn' in k_mx or 'downsample_1' in k_mx:
            k_mx = k_mx.replace('weight', 'gamma')
            k_mx = k_mx.replace('bias', 'beta')
            k_mx = k_mx.replace('running', 'moving')
        print('processing: %s' % k_mx)
        arg_flag = False
        aux_flag = False
        if k_mx in arg_names:
            arg_flag = True
            print('    with shape:', arg_shape_dic[k_mx])
        elif k_mx in aux_names:
            aux_flag = True
            print('    with shape:', aux_shape_dic[k_mx])
        assert arg_flag or aux_flag, 'var name not in arg_names or aux_names: %s' % k_mx

        if arg_flag:
            assert v.shape == arg_shape_dic[k_mx], 'shape not match'
            arg_params[k_mx] = mx.nd.array(np.array(v))
        else:
            assert v.shape == aux_shape_dic[k_mx], 'shape not match'
            aux_params[k_mx] = mx.nd.array(np.array(v))
    print('arg number: ', len(arg_params))
    print('aux number: ', len(aux_params))
    print('state dict number: ', len(checkpoint['state_dict'].items()))

    model = mx.mod.Module(symbol=symbol, label_names=None)
    model.bind(data_shapes=[('data', tuple(input_dim))])
    model.init_params(arg_params=arg_params, aux_params=aux_params)
    model.save_checkpoint(output_prefix, 0)


if __name__ == '__main__':
    weight_file = '/media/chencp/data_ssd2/models/pytorch_model/resnet50_places365.pth.tar'
    network = 'resnet_ori'
    output_prefix = 'resnet50_mxnet'
    convert(weight_file, network=network, output_prefix=output_prefix, bn_data_layer=False)
    output_prefix = 'resnet50_mxnet_bn_data'
    convert(weight_file, network=network, output_prefix=output_prefix, bn_data_layer=True)
