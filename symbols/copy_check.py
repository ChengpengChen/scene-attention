"""to check the mxnet model transformed from pytorch

on 2018.08.20
"""
# import sys
# sys.path.insert(0, '/media/chencp/data_ssd2/work/incubator-mxnet/python/')
import os
import importlib
from collections import namedtuple

import torch
from torch.autograd import Variable as V
from torchvision import models
from torchvision import transforms as trn
from torch.nn import functional as F
import mxnet as mx
import numpy as np
from PIL import Image

input_dim = (1, 3, 224, 224)


def copy_check(model_load_prefix, bn_data_layer=False):
    # test image
    img_name = 'test.jpg'

    # pytorch model
    model_file = '/media/chencp/data_ssd2/models/pytorch_model/resnet50_places365.pth.tar'
    model = models.__dict__['resnet50'](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    # load the image transformer
    centre_crop = trn.Compose([
        trn.Resize((256, 256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(img_name)
    input_img = V(centre_crop(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    logit_np = np.array(logit.data.squeeze())
    # h_x = F.softmax(logit, 1).data.squeeze()
    # print(logit_np)

    # mxnet model
    Batch = namedtuple("Batch", ['data'])
    # model_load_prefix = 'resnet50_mxnet'
    sym, arg_params, aux_params = mx.model.load_checkpoint('%s' % model_load_prefix, 0)
    flatten = sym.get_internals()["fc_output"]
    # sym = importlib.import_module('symbol_resnet_ori').get_symbol()
    model_mx = mx.mod.Module(symbol=flatten, label_names=None)
    model_mx.bind(data_shapes=[('data', tuple(input_dim))])
    model_mx.init_params(arg_params=arg_params, aux_params=aux_params)

    input_img = np.array(input_img)
    if bn_data_layer:
        # the network equieded with ba_data_layer, so transform the image back to [0, 255]
        mean_torch = np.reshape(np.array([0.485, 0.456, 0.406]), [1, 3, 1, 1])
        std_torch = np.reshape(np.array([0.229, 0.224, 0.225]), [1, 3, 1, 1])
        input_img *= std_torch
        input_img += mean_torch
        input_img *= 255

    input_img_mx = mx.nd.array(input_img)
    model_mx.forward(Batch(data=[input_img_mx]), is_train=False)

    data = model_mx.get_outputs()[0].asnumpy()
    # print(data)

    # save the image data
    # np.save('save_img.npy', np.array(input_img))

    print('the diff between outputs from these two models: %f' % np.sum(np.abs(logit_np-data)))


if __name__ == '__main__':
    model_load_prefix = 'resnet50_mxnet_bn_data'
    copy_check(model_load_prefix='../pretrain_models/'+model_load_prefix, bn_data_layer=True)
    model_load_prefix = 'resnet50_mxnet'
    copy_check(model_load_prefix='../pretrain_models/'+model_load_prefix, bn_data_layer=False)
