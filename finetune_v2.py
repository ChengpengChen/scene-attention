"""
use config.py file for changing setting in cmd line
"""
from __future__ import print_function, division
import sys
sys.path.insert(0, '/media/chencp/data_ssd2/work/incubator-mxnet/python/')
import os
import logging
import yaml
import importlib
import mxnet as mx
import numpy as np
from easydict import EasyDict
from pprint import pprint
import argparse
from utils.config import cfg, cfg_from_file, cfg_from_list

from utils.normal_iterator import NormalIterator
from utils.misc import clean_immediate_checkpoints
from utils.module import MyModule


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='scene attention repo training')
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file',
                        default=None, type=str)
    parser.add_argument('--prefix', dest='prefix', help='the prefix to indicate the training process',
                        type=str, default='baseline')
    parser.add_argument('--dataset', dest='dataset', help='select dataset for training (mit67 or sun397)',
                        type=str, default='mit67')

    parser.add_argument('--num_non_local_block', dest='num_non_local_block',
                        help='number of non local block (only 0, 1, 5 and 10 support)', default=0, type=int)
    parser.add_argument("--no_bn_non_local", default=False, action="store_true")
    parser.add_argument("--nonlocal_sub_sample", default=False, action="store_true")

    parser.add_argument('--dropout_ratio', dest='dropout_ratio',
                        help='dropout ratio for layer before global poolint', default=0, type=float)

    parser.add_argument('--set', dest='set_cfgs', help='set config keys', default=None, nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def build_network(symbol, num_id, lr_mult_fc=1.0, dropout_ratio=0.):
    label = mx.symbol.Variable(name="softmax_label")
    group = [label]

    pooling = mx.symbol.Pooling(data=symbol, global_pool=True, pool_type='avg', name='global_avg', kernel=(1, 1))

    dropout = mx.symbol.Dropout(pooling, p=dropout_ratio) if dropout_ratio > 0 else pooling

    softmax_fc = mx.symbol.FullyConnected(data=dropout, num_hidden=num_id, name='softmax_fc', lr_mult=lr_mult_fc)
    softmax = mx.symbol.SoftmaxOutput(data=softmax_fc, label=label, name='softmax')

    group.append(softmax)
    return mx.symbol.Group(group)


def get_iterators(data_dir, img_list, batch_size_all, image_size, aug_dict, data_norm, seed):
    rand_mirror = aug_dict.get("rand_mirror", False)
    rand_crop = aug_dict.get("rand_crop", False)
    random_erasing = aug_dict.get("random_erasing", False)
    resize_shorter = aug_dict.get("resize_shorter", 0)
    force_resize = aug_dict.get("force_resize", 0)

    train = NormalIterator(data_dir=data_dir, img_list_file=img_list[0], batch_size=batch_size_all[0],
                           image_size=image_size, rand_mirror=rand_mirror, rand_crop=rand_crop,
                           random_erasing=random_erasing, shuffle=True, resize_shorter=resize_shorter,
                           force_resize=force_resize, data_norm=data_norm, random_seed=seed)

    val = NormalIterator(data_dir=data_dir, img_list_file=img_list[1], batch_size=batch_size_all[1],
                         image_size=image_size, rand_mirror=False, rand_crop=False, random_erasing=False,
                         shuffle=False, resize_shorter=resize_shorter, force_resize=force_resize,
                         data_norm=data_norm, random_seed=seed)

    return train, val


if __name__ == '__main__':
    print('mxnet version:%s' % mx.__version__)
    random_seed = 0
    mx.random.seed(random_seed)

    # load configuration
    args = parse_args()
    print('Called with args:')
    print(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
        set_cfgs = args.set_cfgs
    else:
        set_cfgs = []

    print('Using config:')
    pprint(cfg)

    dataset = args.dataset if 'dataset' not in set_cfgs else cfg.dataset
    model_load_prefix = cfg.model_load_prefix
    model_load_epoch = cfg.model_load_epoch
    network = cfg.network
    gpus = cfg.gpus
    data_dir = cfg[dataset].data_dir
    lr_step = cfg.lr_step
    optmizer = cfg.optimizer
    lr = cfg.lr
    wd = cfg.wd
    num_epoch = cfg.num_epoch
    image_size = cfg.image_size
    prefix = args.prefix if 'prefix' not in set_cfgs else cfg.prefix
    prefix = dataset + '/' + prefix
    batch_size = cfg.batch_size
    batch_size_val = cfg[dataset].batch_size_val
    num_id = cfg[dataset].num_id
    aug = cfg.aug
    begin_epoch = cfg.begin_epoch
    img_list = [cfg[dataset].list_train, cfg[dataset].list_test]
    bn_data_layer = cfg.bn_data_layer
    data_norm = not bn_data_layer  # apply data norm in data preprocessing
    lr_mult_fc = cfg.lr_mult_fc
    lr_mult_nonlocal = cfg.lr_mult_nonlocal
    bn_non_local = not args.no_bn_non_local if 'bn_non_local' in set_cfgs else cfg.bn_non_local
    num_non_local_block = args.num_non_local_block \
        if 'num_non_local_block' not in set_cfgs else cfg.num_non_local_block
    nonlocal_sub_sample = args.nonlocal_sub_sample \
        if 'nonlocal_sub_sample' not in set_cfgs else cfg.nonlocal_sub_sample
    dropout_ratio = args.dropout_ratio if 'dropout_ratio' not in set_cfgs else cfg.dropout_ratio

    # config logger
    logging.basicConfig(format="%(asctime)s %(message)s",
                        filename='log/%s/%s.log' % (dataset, os.path.basename(prefix)), filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.info(args)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)

    _, arg_params, aux_params = mx.model.load_checkpoint('%s' % model_load_prefix, model_load_epoch)
    # arg_params, aux_params = None, None

    devices = [mx.gpu(int(i)) for i in gpus.split(',')]

    train, val = get_iterators(data_dir=data_dir, img_list=img_list, batch_size_all=[batch_size, batch_size_val],
                               image_size=image_size, aug_dict=aug, seed=random_seed, data_norm=data_norm)

    steps = [int(x) for x in lr_step.split(',')]
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=[train.size * x for x in steps], factor=0.1)
    init = mx.initializer.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)

    # lr_scheduler = WarmupMultiFactorScheduler(step=[s * train.size for s in steps], factor=0.1, warmup=True,
    #                                           warmup_lr=1e-4, warmup_step=train.size * 20, mode="gradual")
    # lr_scheduler = ExponentialScheduler(base_lr=lr, exp=0.001, start_step=150 * train.size, end_step=300 * train.size)
    # init = mx.initializer.MSRAPrelu(factor_type='out', slope=0.0)

    optimizer_params = {"learning_rate": lr,
                        "wd": wd,
                        "lr_scheduler": lr_scheduler,
                        "rescale_grad": 1.0 / batch_size,
                        "begin_num_update": begin_epoch * train.size,
                        }

    resnet_class = importlib.import_module('symbols.symbol_' + network).ResnetClass()
    symbol = resnet_class.get_symbol(bn_data_layer=bn_data_layer,
                                     num_non_local_block=num_non_local_block,
                                     lr_mult_nonlocal=lr_mult_nonlocal,
                                     bn_non_local=bn_non_local,
                                     nonlocal_sub_sample=nonlocal_sub_sample)

    # initialize the bn layers in non local block
    if num_non_local_block != 0:
        # infer shape for init
        data_shape_dict = {'data': (batch_size, 3, image_size, image_size)}
        resnet_class.infer_shape(data_shape_dict)
        if bn_non_local:
            # beta and gamma of bn layers
            list_arguments = [ins for ins in symbol.list_arguments() if 'nonlocal_bn' in ins]
            for ins in list_arguments:
                print('zero init param: {}'.format(ins))
                arg_params[ins] = mx.nd.zeros(shape=resnet_class.arg_shape_dict[ins])
            # mean and var of bn layers
            list_aux = [ins for ins in symbol.list_auxiliary_states() if 'nonlocal_bn' in ins]
            for ins in list_aux:
                print('zero init param: {}'.format(ins))
                aux_params[ins] = mx.nd.zeros(shape=resnet_class.aux_shape_dict[ins])
        else:
            # init the 1x1 conv layer at non local block
            list_arguments = [ins for ins in symbol.list_arguments() if 'nonlocal_w_conv' in ins]
            for ins in list_arguments:
                print('zero init param: {}'.format(ins))
                arg_params[ins] = mx.nd.zeros(shape=resnet_class.arg_shape_dict[ins])

    net = build_network(symbol=symbol, num_id=num_id, lr_mult_fc=lr_mult_fc, dropout_ratio=dropout_ratio)

    acc = mx.metric.Accuracy(output_names=["softmax_output"], label_names=["softmax_label"], name="acc")
    ce_loss = mx.metric.CrossEntropy(output_names=["softmax_output"], label_names=["softmax_label"], name="ce")
    metric_list = [acc, ce_loss]
    metric = mx.metric.CompositeEvalMetric(metrics=metric_list)

    # model = mx.mod.Module(symbol=net, context=devices, logger=logger)
    model = MyModule(symbol=net, context=devices, logger=logger)
    # import pdb
    # pdb.set_trace()

    model.fit_(train_data=train,
               eval_data=val,
               eval_metric=metric,
               validation_metric=metric,
               arg_params=arg_params,
               aux_params=aux_params,
               allow_missing=True,
               initializer=init,
               optimizer=optmizer,
               optimizer_params=optimizer_params,
               num_epoch=num_epoch,
               begin_epoch=begin_epoch,
               batch_end_callback=mx.callback.Speedometer(batch_size=batch_size, frequent=50),
               epoch_end_callback=mx.callback.do_checkpoint("models/" + prefix, period=10),
               kvstore='device',
               eval_period=5)

    clean_immediate_checkpoints("models", prefix, num_epoch)
