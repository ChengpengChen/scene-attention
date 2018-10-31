from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#
__C.gpus = '0'


# settings for optimizer
__C.optimizer = "sgd"
__C.lr = 0.001
__C.wd = 0.0005
__C.lr_step = '20,35'
__C.momentum = 0.

__C.begin_epoch = 0
__C.num_epoch = 50
__C.batch_size = 16

# lr_mult for fully connected layer in softmax
__C.lr_mult_fc = 10
__C.lr_mult_nonlocal = 1
__C.dropout_ratio = 0.5

#settings for data
__C.dataset = "mit67"  # "sun397"
__C.image_size = 224


#settings for model
__C.bn_non_local = True
__C.nonlocal_sub_sample = False
__C.num_non_local_block = 1
#prefix: 'resnet50_sun397_nonlocal_bs32_5bk_0902'
__C.prefix = 'baseline_resnet50_mit67_bs16_1bk_0902_v2'

__C.network = "resnet50"

__C.bn_data_layer = True
__C.model_load_prefix = "pretrain_models/resnet50_mxnet_bn_data"
__C.model_load_epoch = 0


#
# data augmentation options
#
__C.aug = edict()

__C.aug.force_resize = 0
__C.aug.resize_shorter = 0  # open when random crop is selected
__C.aug.rand_mirror = True
__C.aug.rand_crop = True
__C.aug.random_erasing = False

#
# mit67 options
#
__C.mit67 = edict()

__C.mit67.num_id = 67
__C.mit67.data_dir = 'dataset/MIT67/Images'
__C.mit67.list_train = 'dataset/MIT67/list/TrainImages.label'
__C.mit67.list_test = 'dataset/MIT67/list/TestImages.label'
__C.mit67.batch_size_val = 10

#
# sun397 options
#
__C.sun397 = edict()

__C.sun397.num_id = 397
__C.sun397.data_dir = 'dataset/SUN397/Images'
__C.sun397.list_train = 'dataset/SUN397/list/TrainImages.label'
__C.sun397.list_test = 'dataset/SUN397/list/TestImages.label'
__C.sun397.batch_size_val = 10


def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v


def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
  """Set config keys via list (e.g., from command line)."""
  from ast import literal_eval
  assert len(cfg_list) % 2 == 0
  for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
    key_list = k.split('.')
    d = __C
    for subkey in key_list[:-1]:
      assert subkey in d
      d = d[subkey]
    subkey = key_list[-1]
    assert subkey in d
    try:
      value = literal_eval(v)
    except:
      # handle the case when v is a string literal
      value = v
    assert type(value) == type(d[subkey]), \
      'type {} does not match original type {}'.format(
        type(value), type(d[subkey]))
    d[subkey] = value
