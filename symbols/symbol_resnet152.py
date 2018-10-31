import mxnet as mx


def resnet152():
    data = mx.symbol.Variable(name='data')
    conv1 = mx.symbol.Convolution(name='conv1', data=data , num_filter=64, pad=(3, 3), kernel=(7,7), stride=(2,2), no_bias=True)
    bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_conv1 = bn_conv1
    conv1_relu = mx.symbol.Activation(name='conv1_relu', data=scale_conv1 , act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
    res2a_branch1 = mx.symbol.Convolution(name='res2a_branch1', data=pool1 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn2a_branch1 = mx.symbol.BatchNorm(name='bn2a_branch1', data=res2a_branch1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale2a_branch1 = bn2a_branch1
    res2a_branch2a = mx.symbol.Convolution(name='res2a_branch2a', data=pool1 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn2a_branch2a = mx.symbol.BatchNorm(name='bn2a_branch2a', data=res2a_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale2a_branch2a = bn2a_branch2a
    res2a_branch2a_relu = mx.symbol.Activation(name='res2a_branch2a_relu', data=scale2a_branch2a , act_type='relu')
    res2a_branch2b = mx.symbol.Convolution(name='res2a_branch2b', data=res2a_branch2a_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn2a_branch2b = mx.symbol.BatchNorm(name='bn2a_branch2b', data=res2a_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale2a_branch2b = bn2a_branch2b
    res2a_branch2b_relu = mx.symbol.Activation(name='res2a_branch2b_relu', data=scale2a_branch2b , act_type='relu')
    res2a_branch2c = mx.symbol.Convolution(name='res2a_branch2c', data=res2a_branch2b_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn2a_branch2c = mx.symbol.BatchNorm(name='bn2a_branch2c', data=res2a_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale2a_branch2c = bn2a_branch2c
    res2a = mx.symbol.broadcast_add(name='res2a', *[scale2a_branch1,scale2a_branch2c] )
    res2a_relu = mx.symbol.Activation(name='res2a_relu', data=res2a , act_type='relu')
    res2b_branch2a = mx.symbol.Convolution(name='res2b_branch2a', data=res2a_relu , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn2b_branch2a = mx.symbol.BatchNorm(name='bn2b_branch2a', data=res2b_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale2b_branch2a = bn2b_branch2a
    res2b_branch2a_relu = mx.symbol.Activation(name='res2b_branch2a_relu', data=scale2b_branch2a , act_type='relu')
    res2b_branch2b = mx.symbol.Convolution(name='res2b_branch2b', data=res2b_branch2a_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn2b_branch2b = mx.symbol.BatchNorm(name='bn2b_branch2b', data=res2b_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale2b_branch2b = bn2b_branch2b
    res2b_branch2b_relu = mx.symbol.Activation(name='res2b_branch2b_relu', data=scale2b_branch2b , act_type='relu')
    res2b_branch2c = mx.symbol.Convolution(name='res2b_branch2c', data=res2b_branch2b_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn2b_branch2c = mx.symbol.BatchNorm(name='bn2b_branch2c', data=res2b_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale2b_branch2c = bn2b_branch2c
    res2b = mx.symbol.broadcast_add(name='res2b', *[res2a_relu,scale2b_branch2c] )
    res2b_relu = mx.symbol.Activation(name='res2b_relu', data=res2b , act_type='relu')
    res2c_branch2a = mx.symbol.Convolution(name='res2c_branch2a', data=res2b_relu , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn2c_branch2a = mx.symbol.BatchNorm(name='bn2c_branch2a', data=res2c_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale2c_branch2a = bn2c_branch2a
    res2c_branch2a_relu = mx.symbol.Activation(name='res2c_branch2a_relu', data=scale2c_branch2a , act_type='relu')
    res2c_branch2b = mx.symbol.Convolution(name='res2c_branch2b', data=res2c_branch2a_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn2c_branch2b = mx.symbol.BatchNorm(name='bn2c_branch2b', data=res2c_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale2c_branch2b = bn2c_branch2b
    res2c_branch2b_relu = mx.symbol.Activation(name='res2c_branch2b_relu', data=scale2c_branch2b , act_type='relu')
    res2c_branch2c = mx.symbol.Convolution(name='res2c_branch2c', data=res2c_branch2b_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn2c_branch2c = mx.symbol.BatchNorm(name='bn2c_branch2c', data=res2c_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale2c_branch2c = bn2c_branch2c
    res2c = mx.symbol.broadcast_add(name='res2c', *[res2b_relu,scale2c_branch2c] )
    res2c_relu = mx.symbol.Activation(name='res2c_relu', data=res2c , act_type='relu')
    res3a_branch1 = mx.symbol.Convolution(name='res3a_branch1', data=res2c_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
    bn3a_branch1 = mx.symbol.BatchNorm(name='bn3a_branch1', data=res3a_branch1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3a_branch1 = bn3a_branch1
    res3a_branch2a = mx.symbol.Convolution(name='res3a_branch2a', data=res2c_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
    bn3a_branch2a = mx.symbol.BatchNorm(name='bn3a_branch2a', data=res3a_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3a_branch2a = bn3a_branch2a
    res3a_branch2a_relu = mx.symbol.Activation(name='res3a_branch2a_relu', data=scale3a_branch2a , act_type='relu')
    res3a_branch2b = mx.symbol.Convolution(name='res3a_branch2b', data=res3a_branch2a_relu , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn3a_branch2b = mx.symbol.BatchNorm(name='bn3a_branch2b', data=res3a_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3a_branch2b = bn3a_branch2b
    res3a_branch2b_relu = mx.symbol.Activation(name='res3a_branch2b_relu', data=scale3a_branch2b , act_type='relu')
    res3a_branch2c = mx.symbol.Convolution(name='res3a_branch2c', data=res3a_branch2b_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3a_branch2c = mx.symbol.BatchNorm(name='bn3a_branch2c', data=res3a_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3a_branch2c = bn3a_branch2c
    res3a = mx.symbol.broadcast_add(name='res3a', *[scale3a_branch1,scale3a_branch2c] )
    res3a_relu = mx.symbol.Activation(name='res3a_relu', data=res3a , act_type='relu')
    res3b1_branch2a = mx.symbol.Convolution(name='res3b1_branch2a', data=res3a_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b1_branch2a = mx.symbol.BatchNorm(name='bn3b1_branch2a', data=res3b1_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b1_branch2a = bn3b1_branch2a
    res3b1_branch2a_relu = mx.symbol.Activation(name='res3b1_branch2a_relu', data=scale3b1_branch2a , act_type='relu')
    res3b1_branch2b = mx.symbol.Convolution(name='res3b1_branch2b', data=res3b1_branch2a_relu , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn3b1_branch2b = mx.symbol.BatchNorm(name='bn3b1_branch2b', data=res3b1_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b1_branch2b = bn3b1_branch2b
    res3b1_branch2b_relu = mx.symbol.Activation(name='res3b1_branch2b_relu', data=scale3b1_branch2b , act_type='relu')
    res3b1_branch2c = mx.symbol.Convolution(name='res3b1_branch2c', data=res3b1_branch2b_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b1_branch2c = mx.symbol.BatchNorm(name='bn3b1_branch2c', data=res3b1_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b1_branch2c = bn3b1_branch2c
    res3b1 = mx.symbol.broadcast_add(name='res3b1', *[res3a_relu,scale3b1_branch2c] )
    res3b1_relu = mx.symbol.Activation(name='res3b1_relu', data=res3b1 , act_type='relu')
    res3b2_branch2a = mx.symbol.Convolution(name='res3b2_branch2a', data=res3b1_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b2_branch2a = mx.symbol.BatchNorm(name='bn3b2_branch2a', data=res3b2_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b2_branch2a = bn3b2_branch2a
    res3b2_branch2a_relu = mx.symbol.Activation(name='res3b2_branch2a_relu', data=scale3b2_branch2a , act_type='relu')
    res3b2_branch2b = mx.symbol.Convolution(name='res3b2_branch2b', data=res3b2_branch2a_relu , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn3b2_branch2b = mx.symbol.BatchNorm(name='bn3b2_branch2b', data=res3b2_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b2_branch2b = bn3b2_branch2b
    res3b2_branch2b_relu = mx.symbol.Activation(name='res3b2_branch2b_relu', data=scale3b2_branch2b , act_type='relu')
    res3b2_branch2c = mx.symbol.Convolution(name='res3b2_branch2c', data=res3b2_branch2b_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b2_branch2c = mx.symbol.BatchNorm(name='bn3b2_branch2c', data=res3b2_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b2_branch2c = bn3b2_branch2c
    res3b2 = mx.symbol.broadcast_add(name='res3b2', *[res3b1_relu,scale3b2_branch2c] )
    res3b2_relu = mx.symbol.Activation(name='res3b2_relu', data=res3b2 , act_type='relu')
    res3b3_branch2a = mx.symbol.Convolution(name='res3b3_branch2a', data=res3b2_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b3_branch2a = mx.symbol.BatchNorm(name='bn3b3_branch2a', data=res3b3_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b3_branch2a = bn3b3_branch2a
    res3b3_branch2a_relu = mx.symbol.Activation(name='res3b3_branch2a_relu', data=scale3b3_branch2a , act_type='relu')
    res3b3_branch2b = mx.symbol.Convolution(name='res3b3_branch2b', data=res3b3_branch2a_relu , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn3b3_branch2b = mx.symbol.BatchNorm(name='bn3b3_branch2b', data=res3b3_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b3_branch2b = bn3b3_branch2b
    res3b3_branch2b_relu = mx.symbol.Activation(name='res3b3_branch2b_relu', data=scale3b3_branch2b , act_type='relu')
    res3b3_branch2c = mx.symbol.Convolution(name='res3b3_branch2c', data=res3b3_branch2b_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b3_branch2c = mx.symbol.BatchNorm(name='bn3b3_branch2c', data=res3b3_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b3_branch2c = bn3b3_branch2c
    res3b3 = mx.symbol.broadcast_add(name='res3b3', *[res3b2_relu,scale3b3_branch2c] )
    res3b3_relu = mx.symbol.Activation(name='res3b3_relu', data=res3b3 , act_type='relu')
    res3b4_branch2a = mx.symbol.Convolution(name='res3b4_branch2a', data=res3b3_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b4_branch2a = mx.symbol.BatchNorm(name='bn3b4_branch2a', data=res3b4_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b4_branch2a = bn3b4_branch2a
    res3b4_branch2a_relu = mx.symbol.Activation(name='res3b4_branch2a_relu', data=scale3b4_branch2a , act_type='relu')
    res3b4_branch2b = mx.symbol.Convolution(name='res3b4_branch2b', data=res3b4_branch2a_relu , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn3b4_branch2b = mx.symbol.BatchNorm(name='bn3b4_branch2b', data=res3b4_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b4_branch2b = bn3b4_branch2b
    res3b4_branch2b_relu = mx.symbol.Activation(name='res3b4_branch2b_relu', data=scale3b4_branch2b , act_type='relu')
    res3b4_branch2c = mx.symbol.Convolution(name='res3b4_branch2c', data=res3b4_branch2b_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b4_branch2c = mx.symbol.BatchNorm(name='bn3b4_branch2c', data=res3b4_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b4_branch2c = bn3b4_branch2c
    res3b4 = mx.symbol.broadcast_add(name='res3b4', *[res3b3_relu,scale3b4_branch2c] )
    res3b4_relu = mx.symbol.Activation(name='res3b4_relu', data=res3b4 , act_type='relu')
    res3b5_branch2a = mx.symbol.Convolution(name='res3b5_branch2a', data=res3b4_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b5_branch2a = mx.symbol.BatchNorm(name='bn3b5_branch2a', data=res3b5_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b5_branch2a = bn3b5_branch2a
    res3b5_branch2a_relu = mx.symbol.Activation(name='res3b5_branch2a_relu', data=scale3b5_branch2a , act_type='relu')
    res3b5_branch2b = mx.symbol.Convolution(name='res3b5_branch2b', data=res3b5_branch2a_relu , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn3b5_branch2b = mx.symbol.BatchNorm(name='bn3b5_branch2b', data=res3b5_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b5_branch2b = bn3b5_branch2b
    res3b5_branch2b_relu = mx.symbol.Activation(name='res3b5_branch2b_relu', data=scale3b5_branch2b , act_type='relu')
    res3b5_branch2c = mx.symbol.Convolution(name='res3b5_branch2c', data=res3b5_branch2b_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b5_branch2c = mx.symbol.BatchNorm(name='bn3b5_branch2c', data=res3b5_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b5_branch2c = bn3b5_branch2c
    res3b5 = mx.symbol.broadcast_add(name='res3b5', *[res3b4_relu,scale3b5_branch2c] )
    res3b5_relu = mx.symbol.Activation(name='res3b5_relu', data=res3b5 , act_type='relu')
    res3b6_branch2a = mx.symbol.Convolution(name='res3b6_branch2a', data=res3b5_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b6_branch2a = mx.symbol.BatchNorm(name='bn3b6_branch2a', data=res3b6_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b6_branch2a = bn3b6_branch2a
    res3b6_branch2a_relu = mx.symbol.Activation(name='res3b6_branch2a_relu', data=scale3b6_branch2a , act_type='relu')
    res3b6_branch2b = mx.symbol.Convolution(name='res3b6_branch2b', data=res3b6_branch2a_relu , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn3b6_branch2b = mx.symbol.BatchNorm(name='bn3b6_branch2b', data=res3b6_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b6_branch2b = bn3b6_branch2b
    res3b6_branch2b_relu = mx.symbol.Activation(name='res3b6_branch2b_relu', data=scale3b6_branch2b , act_type='relu')
    res3b6_branch2c = mx.symbol.Convolution(name='res3b6_branch2c', data=res3b6_branch2b_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b6_branch2c = mx.symbol.BatchNorm(name='bn3b6_branch2c', data=res3b6_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b6_branch2c = bn3b6_branch2c
    res3b6 = mx.symbol.broadcast_add(name='res3b6', *[res3b5_relu,scale3b6_branch2c] )
    res3b6_relu = mx.symbol.Activation(name='res3b6_relu', data=res3b6 , act_type='relu')
    res3b7_branch2a = mx.symbol.Convolution(name='res3b7_branch2a', data=res3b6_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b7_branch2a = mx.symbol.BatchNorm(name='bn3b7_branch2a', data=res3b7_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b7_branch2a = bn3b7_branch2a
    res3b7_branch2a_relu = mx.symbol.Activation(name='res3b7_branch2a_relu', data=scale3b7_branch2a , act_type='relu')
    res3b7_branch2b = mx.symbol.Convolution(name='res3b7_branch2b', data=res3b7_branch2a_relu , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn3b7_branch2b = mx.symbol.BatchNorm(name='bn3b7_branch2b', data=res3b7_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b7_branch2b = bn3b7_branch2b
    res3b7_branch2b_relu = mx.symbol.Activation(name='res3b7_branch2b_relu', data=scale3b7_branch2b , act_type='relu')
    res3b7_branch2c = mx.symbol.Convolution(name='res3b7_branch2c', data=res3b7_branch2b_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b7_branch2c = mx.symbol.BatchNorm(name='bn3b7_branch2c', data=res3b7_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale3b7_branch2c = bn3b7_branch2c
    res3b7 = mx.symbol.broadcast_add(name='res3b7', *[res3b6_relu,scale3b7_branch2c] )
    res3b7_relu = mx.symbol.Activation(name='res3b7_relu', data=res3b7 , act_type='relu')
    res4a_branch1 = mx.symbol.Convolution(name='res4a_branch1', data=res3b7_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
    bn4a_branch1 = mx.symbol.BatchNorm(name='bn4a_branch1', data=res4a_branch1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4a_branch1 = bn4a_branch1
    res4a_branch2a = mx.symbol.Convolution(name='res4a_branch2a', data=res3b7_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
    bn4a_branch2a = mx.symbol.BatchNorm(name='bn4a_branch2a', data=res4a_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4a_branch2a = bn4a_branch2a
    res4a_branch2a_relu = mx.symbol.Activation(name='res4a_branch2a_relu', data=scale4a_branch2a , act_type='relu')
    res4a_branch2b = mx.symbol.Convolution(name='res4a_branch2b', data=res4a_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4a_branch2b = mx.symbol.BatchNorm(name='bn4a_branch2b', data=res4a_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4a_branch2b = bn4a_branch2b
    res4a_branch2b_relu = mx.symbol.Activation(name='res4a_branch2b_relu', data=scale4a_branch2b , act_type='relu')
    res4a_branch2c = mx.symbol.Convolution(name='res4a_branch2c', data=res4a_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4a_branch2c = mx.symbol.BatchNorm(name='bn4a_branch2c', data=res4a_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4a_branch2c = bn4a_branch2c
    res4a = mx.symbol.broadcast_add(name='res4a', *[scale4a_branch1,scale4a_branch2c] )
    res4a_relu = mx.symbol.Activation(name='res4a_relu', data=res4a , act_type='relu')
    res4b1_branch2a = mx.symbol.Convolution(name='res4b1_branch2a', data=res4a_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b1_branch2a = mx.symbol.BatchNorm(name='bn4b1_branch2a', data=res4b1_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b1_branch2a = bn4b1_branch2a
    res4b1_branch2a_relu = mx.symbol.Activation(name='res4b1_branch2a_relu', data=scale4b1_branch2a , act_type='relu')
    res4b1_branch2b = mx.symbol.Convolution(name='res4b1_branch2b', data=res4b1_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b1_branch2b = mx.symbol.BatchNorm(name='bn4b1_branch2b', data=res4b1_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b1_branch2b = bn4b1_branch2b
    res4b1_branch2b_relu = mx.symbol.Activation(name='res4b1_branch2b_relu', data=scale4b1_branch2b , act_type='relu')
    res4b1_branch2c = mx.symbol.Convolution(name='res4b1_branch2c', data=res4b1_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b1_branch2c = mx.symbol.BatchNorm(name='bn4b1_branch2c', data=res4b1_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b1_branch2c = bn4b1_branch2c
    res4b1 = mx.symbol.broadcast_add(name='res4b1', *[res4a_relu,scale4b1_branch2c] )
    res4b1_relu = mx.symbol.Activation(name='res4b1_relu', data=res4b1 , act_type='relu')
    res4b2_branch2a = mx.symbol.Convolution(name='res4b2_branch2a', data=res4b1_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b2_branch2a = mx.symbol.BatchNorm(name='bn4b2_branch2a', data=res4b2_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b2_branch2a = bn4b2_branch2a
    res4b2_branch2a_relu = mx.symbol.Activation(name='res4b2_branch2a_relu', data=scale4b2_branch2a , act_type='relu')
    res4b2_branch2b = mx.symbol.Convolution(name='res4b2_branch2b', data=res4b2_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b2_branch2b = mx.symbol.BatchNorm(name='bn4b2_branch2b', data=res4b2_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b2_branch2b = bn4b2_branch2b
    res4b2_branch2b_relu = mx.symbol.Activation(name='res4b2_branch2b_relu', data=scale4b2_branch2b , act_type='relu')
    res4b2_branch2c = mx.symbol.Convolution(name='res4b2_branch2c', data=res4b2_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b2_branch2c = mx.symbol.BatchNorm(name='bn4b2_branch2c', data=res4b2_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b2_branch2c = bn4b2_branch2c
    res4b2 = mx.symbol.broadcast_add(name='res4b2', *[res4b1_relu,scale4b2_branch2c] )
    res4b2_relu = mx.symbol.Activation(name='res4b2_relu', data=res4b2 , act_type='relu')
    res4b3_branch2a = mx.symbol.Convolution(name='res4b3_branch2a', data=res4b2_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b3_branch2a = mx.symbol.BatchNorm(name='bn4b3_branch2a', data=res4b3_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b3_branch2a = bn4b3_branch2a
    res4b3_branch2a_relu = mx.symbol.Activation(name='res4b3_branch2a_relu', data=scale4b3_branch2a , act_type='relu')
    res4b3_branch2b = mx.symbol.Convolution(name='res4b3_branch2b', data=res4b3_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b3_branch2b = mx.symbol.BatchNorm(name='bn4b3_branch2b', data=res4b3_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b3_branch2b = bn4b3_branch2b
    res4b3_branch2b_relu = mx.symbol.Activation(name='res4b3_branch2b_relu', data=scale4b3_branch2b , act_type='relu')
    res4b3_branch2c = mx.symbol.Convolution(name='res4b3_branch2c', data=res4b3_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b3_branch2c = mx.symbol.BatchNorm(name='bn4b3_branch2c', data=res4b3_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b3_branch2c = bn4b3_branch2c
    res4b3 = mx.symbol.broadcast_add(name='res4b3', *[res4b2_relu,scale4b3_branch2c] )
    res4b3_relu = mx.symbol.Activation(name='res4b3_relu', data=res4b3 , act_type='relu')
    res4b4_branch2a = mx.symbol.Convolution(name='res4b4_branch2a', data=res4b3_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b4_branch2a = mx.symbol.BatchNorm(name='bn4b4_branch2a', data=res4b4_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b4_branch2a = bn4b4_branch2a
    res4b4_branch2a_relu = mx.symbol.Activation(name='res4b4_branch2a_relu', data=scale4b4_branch2a , act_type='relu')
    res4b4_branch2b = mx.symbol.Convolution(name='res4b4_branch2b', data=res4b4_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b4_branch2b = mx.symbol.BatchNorm(name='bn4b4_branch2b', data=res4b4_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b4_branch2b = bn4b4_branch2b
    res4b4_branch2b_relu = mx.symbol.Activation(name='res4b4_branch2b_relu', data=scale4b4_branch2b , act_type='relu')
    res4b4_branch2c = mx.symbol.Convolution(name='res4b4_branch2c', data=res4b4_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b4_branch2c = mx.symbol.BatchNorm(name='bn4b4_branch2c', data=res4b4_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b4_branch2c = bn4b4_branch2c
    res4b4 = mx.symbol.broadcast_add(name='res4b4', *[res4b3_relu,scale4b4_branch2c] )
    res4b4_relu = mx.symbol.Activation(name='res4b4_relu', data=res4b4 , act_type='relu')
    res4b5_branch2a = mx.symbol.Convolution(name='res4b5_branch2a', data=res4b4_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b5_branch2a = mx.symbol.BatchNorm(name='bn4b5_branch2a', data=res4b5_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b5_branch2a = bn4b5_branch2a
    res4b5_branch2a_relu = mx.symbol.Activation(name='res4b5_branch2a_relu', data=scale4b5_branch2a , act_type='relu')
    res4b5_branch2b = mx.symbol.Convolution(name='res4b5_branch2b', data=res4b5_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b5_branch2b = mx.symbol.BatchNorm(name='bn4b5_branch2b', data=res4b5_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b5_branch2b = bn4b5_branch2b
    res4b5_branch2b_relu = mx.symbol.Activation(name='res4b5_branch2b_relu', data=scale4b5_branch2b , act_type='relu')
    res4b5_branch2c = mx.symbol.Convolution(name='res4b5_branch2c', data=res4b5_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b5_branch2c = mx.symbol.BatchNorm(name='bn4b5_branch2c', data=res4b5_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b5_branch2c = bn4b5_branch2c
    res4b5 = mx.symbol.broadcast_add(name='res4b5', *[res4b4_relu,scale4b5_branch2c] )
    res4b5_relu = mx.symbol.Activation(name='res4b5_relu', data=res4b5 , act_type='relu')
    res4b6_branch2a = mx.symbol.Convolution(name='res4b6_branch2a', data=res4b5_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b6_branch2a = mx.symbol.BatchNorm(name='bn4b6_branch2a', data=res4b6_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b6_branch2a = bn4b6_branch2a
    res4b6_branch2a_relu = mx.symbol.Activation(name='res4b6_branch2a_relu', data=scale4b6_branch2a , act_type='relu')
    res4b6_branch2b = mx.symbol.Convolution(name='res4b6_branch2b', data=res4b6_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b6_branch2b = mx.symbol.BatchNorm(name='bn4b6_branch2b', data=res4b6_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b6_branch2b = bn4b6_branch2b
    res4b6_branch2b_relu = mx.symbol.Activation(name='res4b6_branch2b_relu', data=scale4b6_branch2b , act_type='relu')
    res4b6_branch2c = mx.symbol.Convolution(name='res4b6_branch2c', data=res4b6_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b6_branch2c = mx.symbol.BatchNorm(name='bn4b6_branch2c', data=res4b6_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b6_branch2c = bn4b6_branch2c
    res4b6 = mx.symbol.broadcast_add(name='res4b6', *[res4b5_relu,scale4b6_branch2c] )
    res4b6_relu = mx.symbol.Activation(name='res4b6_relu', data=res4b6 , act_type='relu')
    res4b7_branch2a = mx.symbol.Convolution(name='res4b7_branch2a', data=res4b6_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b7_branch2a = mx.symbol.BatchNorm(name='bn4b7_branch2a', data=res4b7_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b7_branch2a = bn4b7_branch2a
    res4b7_branch2a_relu = mx.symbol.Activation(name='res4b7_branch2a_relu', data=scale4b7_branch2a , act_type='relu')
    res4b7_branch2b = mx.symbol.Convolution(name='res4b7_branch2b', data=res4b7_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b7_branch2b = mx.symbol.BatchNorm(name='bn4b7_branch2b', data=res4b7_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b7_branch2b = bn4b7_branch2b
    res4b7_branch2b_relu = mx.symbol.Activation(name='res4b7_branch2b_relu', data=scale4b7_branch2b , act_type='relu')
    res4b7_branch2c = mx.symbol.Convolution(name='res4b7_branch2c', data=res4b7_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b7_branch2c = mx.symbol.BatchNorm(name='bn4b7_branch2c', data=res4b7_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b7_branch2c = bn4b7_branch2c
    res4b7 = mx.symbol.broadcast_add(name='res4b7', *[res4b6_relu,scale4b7_branch2c] )
    res4b7_relu = mx.symbol.Activation(name='res4b7_relu', data=res4b7 , act_type='relu')
    res4b8_branch2a = mx.symbol.Convolution(name='res4b8_branch2a', data=res4b7_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b8_branch2a = mx.symbol.BatchNorm(name='bn4b8_branch2a', data=res4b8_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b8_branch2a = bn4b8_branch2a
    res4b8_branch2a_relu = mx.symbol.Activation(name='res4b8_branch2a_relu', data=scale4b8_branch2a , act_type='relu')
    res4b8_branch2b = mx.symbol.Convolution(name='res4b8_branch2b', data=res4b8_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b8_branch2b = mx.symbol.BatchNorm(name='bn4b8_branch2b', data=res4b8_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b8_branch2b = bn4b8_branch2b
    res4b8_branch2b_relu = mx.symbol.Activation(name='res4b8_branch2b_relu', data=scale4b8_branch2b , act_type='relu')
    res4b8_branch2c = mx.symbol.Convolution(name='res4b8_branch2c', data=res4b8_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b8_branch2c = mx.symbol.BatchNorm(name='bn4b8_branch2c', data=res4b8_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b8_branch2c = bn4b8_branch2c
    res4b8 = mx.symbol.broadcast_add(name='res4b8', *[res4b7_relu,scale4b8_branch2c] )
    res4b8_relu = mx.symbol.Activation(name='res4b8_relu', data=res4b8 , act_type='relu')
    res4b9_branch2a = mx.symbol.Convolution(name='res4b9_branch2a', data=res4b8_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b9_branch2a = mx.symbol.BatchNorm(name='bn4b9_branch2a', data=res4b9_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b9_branch2a = bn4b9_branch2a
    res4b9_branch2a_relu = mx.symbol.Activation(name='res4b9_branch2a_relu', data=scale4b9_branch2a , act_type='relu')
    res4b9_branch2b = mx.symbol.Convolution(name='res4b9_branch2b', data=res4b9_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b9_branch2b = mx.symbol.BatchNorm(name='bn4b9_branch2b', data=res4b9_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b9_branch2b = bn4b9_branch2b
    res4b9_branch2b_relu = mx.symbol.Activation(name='res4b9_branch2b_relu', data=scale4b9_branch2b , act_type='relu')
    res4b9_branch2c = mx.symbol.Convolution(name='res4b9_branch2c', data=res4b9_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b9_branch2c = mx.symbol.BatchNorm(name='bn4b9_branch2c', data=res4b9_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b9_branch2c = bn4b9_branch2c
    res4b9 = mx.symbol.broadcast_add(name='res4b9', *[res4b8_relu,scale4b9_branch2c] )
    res4b9_relu = mx.symbol.Activation(name='res4b9_relu', data=res4b9 , act_type='relu')
    res4b10_branch2a = mx.symbol.Convolution(name='res4b10_branch2a', data=res4b9_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b10_branch2a = mx.symbol.BatchNorm(name='bn4b10_branch2a', data=res4b10_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b10_branch2a = bn4b10_branch2a
    res4b10_branch2a_relu = mx.symbol.Activation(name='res4b10_branch2a_relu', data=scale4b10_branch2a , act_type='relu')
    res4b10_branch2b = mx.symbol.Convolution(name='res4b10_branch2b', data=res4b10_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b10_branch2b = mx.symbol.BatchNorm(name='bn4b10_branch2b', data=res4b10_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b10_branch2b = bn4b10_branch2b
    res4b10_branch2b_relu = mx.symbol.Activation(name='res4b10_branch2b_relu', data=scale4b10_branch2b , act_type='relu')
    res4b10_branch2c = mx.symbol.Convolution(name='res4b10_branch2c', data=res4b10_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b10_branch2c = mx.symbol.BatchNorm(name='bn4b10_branch2c', data=res4b10_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b10_branch2c = bn4b10_branch2c
    res4b10 = mx.symbol.broadcast_add(name='res4b10', *[res4b9_relu,scale4b10_branch2c] )
    res4b10_relu = mx.symbol.Activation(name='res4b10_relu', data=res4b10 , act_type='relu')
    res4b11_branch2a = mx.symbol.Convolution(name='res4b11_branch2a', data=res4b10_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b11_branch2a = mx.symbol.BatchNorm(name='bn4b11_branch2a', data=res4b11_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b11_branch2a = bn4b11_branch2a
    res4b11_branch2a_relu = mx.symbol.Activation(name='res4b11_branch2a_relu', data=scale4b11_branch2a , act_type='relu')
    res4b11_branch2b = mx.symbol.Convolution(name='res4b11_branch2b', data=res4b11_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b11_branch2b = mx.symbol.BatchNorm(name='bn4b11_branch2b', data=res4b11_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b11_branch2b = bn4b11_branch2b
    res4b11_branch2b_relu = mx.symbol.Activation(name='res4b11_branch2b_relu', data=scale4b11_branch2b , act_type='relu')
    res4b11_branch2c = mx.symbol.Convolution(name='res4b11_branch2c', data=res4b11_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b11_branch2c = mx.symbol.BatchNorm(name='bn4b11_branch2c', data=res4b11_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b11_branch2c = bn4b11_branch2c
    res4b11 = mx.symbol.broadcast_add(name='res4b11', *[res4b10_relu,scale4b11_branch2c] )
    res4b11_relu = mx.symbol.Activation(name='res4b11_relu', data=res4b11 , act_type='relu')
    res4b12_branch2a = mx.symbol.Convolution(name='res4b12_branch2a', data=res4b11_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b12_branch2a = mx.symbol.BatchNorm(name='bn4b12_branch2a', data=res4b12_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b12_branch2a = bn4b12_branch2a
    res4b12_branch2a_relu = mx.symbol.Activation(name='res4b12_branch2a_relu', data=scale4b12_branch2a , act_type='relu')
    res4b12_branch2b = mx.symbol.Convolution(name='res4b12_branch2b', data=res4b12_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b12_branch2b = mx.symbol.BatchNorm(name='bn4b12_branch2b', data=res4b12_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b12_branch2b = bn4b12_branch2b
    res4b12_branch2b_relu = mx.symbol.Activation(name='res4b12_branch2b_relu', data=scale4b12_branch2b , act_type='relu')
    res4b12_branch2c = mx.symbol.Convolution(name='res4b12_branch2c', data=res4b12_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b12_branch2c = mx.symbol.BatchNorm(name='bn4b12_branch2c', data=res4b12_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b12_branch2c = bn4b12_branch2c
    res4b12 = mx.symbol.broadcast_add(name='res4b12', *[res4b11_relu,scale4b12_branch2c] )
    res4b12_relu = mx.symbol.Activation(name='res4b12_relu', data=res4b12 , act_type='relu')
    res4b13_branch2a = mx.symbol.Convolution(name='res4b13_branch2a', data=res4b12_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b13_branch2a = mx.symbol.BatchNorm(name='bn4b13_branch2a', data=res4b13_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b13_branch2a = bn4b13_branch2a
    res4b13_branch2a_relu = mx.symbol.Activation(name='res4b13_branch2a_relu', data=scale4b13_branch2a , act_type='relu')
    res4b13_branch2b = mx.symbol.Convolution(name='res4b13_branch2b', data=res4b13_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b13_branch2b = mx.symbol.BatchNorm(name='bn4b13_branch2b', data=res4b13_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b13_branch2b = bn4b13_branch2b
    res4b13_branch2b_relu = mx.symbol.Activation(name='res4b13_branch2b_relu', data=scale4b13_branch2b , act_type='relu')
    res4b13_branch2c = mx.symbol.Convolution(name='res4b13_branch2c', data=res4b13_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b13_branch2c = mx.symbol.BatchNorm(name='bn4b13_branch2c', data=res4b13_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b13_branch2c = bn4b13_branch2c
    res4b13 = mx.symbol.broadcast_add(name='res4b13', *[res4b12_relu,scale4b13_branch2c] )
    res4b13_relu = mx.symbol.Activation(name='res4b13_relu', data=res4b13 , act_type='relu')
    res4b14_branch2a = mx.symbol.Convolution(name='res4b14_branch2a', data=res4b13_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b14_branch2a = mx.symbol.BatchNorm(name='bn4b14_branch2a', data=res4b14_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b14_branch2a = bn4b14_branch2a
    res4b14_branch2a_relu = mx.symbol.Activation(name='res4b14_branch2a_relu', data=scale4b14_branch2a , act_type='relu')
    res4b14_branch2b = mx.symbol.Convolution(name='res4b14_branch2b', data=res4b14_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b14_branch2b = mx.symbol.BatchNorm(name='bn4b14_branch2b', data=res4b14_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b14_branch2b = bn4b14_branch2b
    res4b14_branch2b_relu = mx.symbol.Activation(name='res4b14_branch2b_relu', data=scale4b14_branch2b , act_type='relu')
    res4b14_branch2c = mx.symbol.Convolution(name='res4b14_branch2c', data=res4b14_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b14_branch2c = mx.symbol.BatchNorm(name='bn4b14_branch2c', data=res4b14_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b14_branch2c = bn4b14_branch2c
    res4b14 = mx.symbol.broadcast_add(name='res4b14', *[res4b13_relu,scale4b14_branch2c] )
    res4b14_relu = mx.symbol.Activation(name='res4b14_relu', data=res4b14 , act_type='relu')
    res4b15_branch2a = mx.symbol.Convolution(name='res4b15_branch2a', data=res4b14_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b15_branch2a = mx.symbol.BatchNorm(name='bn4b15_branch2a', data=res4b15_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b15_branch2a = bn4b15_branch2a
    res4b15_branch2a_relu = mx.symbol.Activation(name='res4b15_branch2a_relu', data=scale4b15_branch2a , act_type='relu')
    res4b15_branch2b = mx.symbol.Convolution(name='res4b15_branch2b', data=res4b15_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b15_branch2b = mx.symbol.BatchNorm(name='bn4b15_branch2b', data=res4b15_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b15_branch2b = bn4b15_branch2b
    res4b15_branch2b_relu = mx.symbol.Activation(name='res4b15_branch2b_relu', data=scale4b15_branch2b , act_type='relu')
    res4b15_branch2c = mx.symbol.Convolution(name='res4b15_branch2c', data=res4b15_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b15_branch2c = mx.symbol.BatchNorm(name='bn4b15_branch2c', data=res4b15_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b15_branch2c = bn4b15_branch2c
    res4b15 = mx.symbol.broadcast_add(name='res4b15', *[res4b14_relu,scale4b15_branch2c] )
    res4b15_relu = mx.symbol.Activation(name='res4b15_relu', data=res4b15 , act_type='relu')
    res4b16_branch2a = mx.symbol.Convolution(name='res4b16_branch2a', data=res4b15_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b16_branch2a = mx.symbol.BatchNorm(name='bn4b16_branch2a', data=res4b16_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b16_branch2a = bn4b16_branch2a
    res4b16_branch2a_relu = mx.symbol.Activation(name='res4b16_branch2a_relu', data=scale4b16_branch2a , act_type='relu')
    res4b16_branch2b = mx.symbol.Convolution(name='res4b16_branch2b', data=res4b16_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b16_branch2b = mx.symbol.BatchNorm(name='bn4b16_branch2b', data=res4b16_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b16_branch2b = bn4b16_branch2b
    res4b16_branch2b_relu = mx.symbol.Activation(name='res4b16_branch2b_relu', data=scale4b16_branch2b , act_type='relu')
    res4b16_branch2c = mx.symbol.Convolution(name='res4b16_branch2c', data=res4b16_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b16_branch2c = mx.symbol.BatchNorm(name='bn4b16_branch2c', data=res4b16_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b16_branch2c = bn4b16_branch2c
    res4b16 = mx.symbol.broadcast_add(name='res4b16', *[res4b15_relu,scale4b16_branch2c] )
    res4b16_relu = mx.symbol.Activation(name='res4b16_relu', data=res4b16 , act_type='relu')
    res4b17_branch2a = mx.symbol.Convolution(name='res4b17_branch2a', data=res4b16_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b17_branch2a = mx.symbol.BatchNorm(name='bn4b17_branch2a', data=res4b17_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b17_branch2a = bn4b17_branch2a
    res4b17_branch2a_relu = mx.symbol.Activation(name='res4b17_branch2a_relu', data=scale4b17_branch2a , act_type='relu')
    res4b17_branch2b = mx.symbol.Convolution(name='res4b17_branch2b', data=res4b17_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b17_branch2b = mx.symbol.BatchNorm(name='bn4b17_branch2b', data=res4b17_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b17_branch2b = bn4b17_branch2b
    res4b17_branch2b_relu = mx.symbol.Activation(name='res4b17_branch2b_relu', data=scale4b17_branch2b , act_type='relu')
    res4b17_branch2c = mx.symbol.Convolution(name='res4b17_branch2c', data=res4b17_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b17_branch2c = mx.symbol.BatchNorm(name='bn4b17_branch2c', data=res4b17_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b17_branch2c = bn4b17_branch2c
    res4b17 = mx.symbol.broadcast_add(name='res4b17', *[res4b16_relu,scale4b17_branch2c] )
    res4b17_relu = mx.symbol.Activation(name='res4b17_relu', data=res4b17 , act_type='relu')
    res4b18_branch2a = mx.symbol.Convolution(name='res4b18_branch2a', data=res4b17_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b18_branch2a = mx.symbol.BatchNorm(name='bn4b18_branch2a', data=res4b18_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b18_branch2a = bn4b18_branch2a
    res4b18_branch2a_relu = mx.symbol.Activation(name='res4b18_branch2a_relu', data=scale4b18_branch2a , act_type='relu')
    res4b18_branch2b = mx.symbol.Convolution(name='res4b18_branch2b', data=res4b18_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b18_branch2b = mx.symbol.BatchNorm(name='bn4b18_branch2b', data=res4b18_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b18_branch2b = bn4b18_branch2b
    res4b18_branch2b_relu = mx.symbol.Activation(name='res4b18_branch2b_relu', data=scale4b18_branch2b , act_type='relu')
    res4b18_branch2c = mx.symbol.Convolution(name='res4b18_branch2c', data=res4b18_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b18_branch2c = mx.symbol.BatchNorm(name='bn4b18_branch2c', data=res4b18_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b18_branch2c = bn4b18_branch2c
    res4b18 = mx.symbol.broadcast_add(name='res4b18', *[res4b17_relu,scale4b18_branch2c] )
    res4b18_relu = mx.symbol.Activation(name='res4b18_relu', data=res4b18 , act_type='relu')
    res4b19_branch2a = mx.symbol.Convolution(name='res4b19_branch2a', data=res4b18_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b19_branch2a = mx.symbol.BatchNorm(name='bn4b19_branch2a', data=res4b19_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b19_branch2a = bn4b19_branch2a
    res4b19_branch2a_relu = mx.symbol.Activation(name='res4b19_branch2a_relu', data=scale4b19_branch2a , act_type='relu')
    res4b19_branch2b = mx.symbol.Convolution(name='res4b19_branch2b', data=res4b19_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b19_branch2b = mx.symbol.BatchNorm(name='bn4b19_branch2b', data=res4b19_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b19_branch2b = bn4b19_branch2b
    res4b19_branch2b_relu = mx.symbol.Activation(name='res4b19_branch2b_relu', data=scale4b19_branch2b , act_type='relu')
    res4b19_branch2c = mx.symbol.Convolution(name='res4b19_branch2c', data=res4b19_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b19_branch2c = mx.symbol.BatchNorm(name='bn4b19_branch2c', data=res4b19_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b19_branch2c = bn4b19_branch2c
    res4b19 = mx.symbol.broadcast_add(name='res4b19', *[res4b18_relu,scale4b19_branch2c] )
    res4b19_relu = mx.symbol.Activation(name='res4b19_relu', data=res4b19 , act_type='relu')
    res4b20_branch2a = mx.symbol.Convolution(name='res4b20_branch2a', data=res4b19_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b20_branch2a = mx.symbol.BatchNorm(name='bn4b20_branch2a', data=res4b20_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b20_branch2a = bn4b20_branch2a
    res4b20_branch2a_relu = mx.symbol.Activation(name='res4b20_branch2a_relu', data=scale4b20_branch2a , act_type='relu')
    res4b20_branch2b = mx.symbol.Convolution(name='res4b20_branch2b', data=res4b20_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b20_branch2b = mx.symbol.BatchNorm(name='bn4b20_branch2b', data=res4b20_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b20_branch2b = bn4b20_branch2b
    res4b20_branch2b_relu = mx.symbol.Activation(name='res4b20_branch2b_relu', data=scale4b20_branch2b , act_type='relu')
    res4b20_branch2c = mx.symbol.Convolution(name='res4b20_branch2c', data=res4b20_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b20_branch2c = mx.symbol.BatchNorm(name='bn4b20_branch2c', data=res4b20_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b20_branch2c = bn4b20_branch2c
    res4b20 = mx.symbol.broadcast_add(name='res4b20', *[res4b19_relu,scale4b20_branch2c] )
    res4b20_relu = mx.symbol.Activation(name='res4b20_relu', data=res4b20 , act_type='relu')
    res4b21_branch2a = mx.symbol.Convolution(name='res4b21_branch2a', data=res4b20_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b21_branch2a = mx.symbol.BatchNorm(name='bn4b21_branch2a', data=res4b21_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b21_branch2a = bn4b21_branch2a
    res4b21_branch2a_relu = mx.symbol.Activation(name='res4b21_branch2a_relu', data=scale4b21_branch2a , act_type='relu')
    res4b21_branch2b = mx.symbol.Convolution(name='res4b21_branch2b', data=res4b21_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b21_branch2b = mx.symbol.BatchNorm(name='bn4b21_branch2b', data=res4b21_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b21_branch2b = bn4b21_branch2b
    res4b21_branch2b_relu = mx.symbol.Activation(name='res4b21_branch2b_relu', data=scale4b21_branch2b , act_type='relu')
    res4b21_branch2c = mx.symbol.Convolution(name='res4b21_branch2c', data=res4b21_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b21_branch2c = mx.symbol.BatchNorm(name='bn4b21_branch2c', data=res4b21_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b21_branch2c = bn4b21_branch2c
    res4b21 = mx.symbol.broadcast_add(name='res4b21', *[res4b20_relu,scale4b21_branch2c] )
    res4b21_relu = mx.symbol.Activation(name='res4b21_relu', data=res4b21 , act_type='relu')
    res4b22_branch2a = mx.symbol.Convolution(name='res4b22_branch2a', data=res4b21_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b22_branch2a = mx.symbol.BatchNorm(name='bn4b22_branch2a', data=res4b22_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b22_branch2a = bn4b22_branch2a
    res4b22_branch2a_relu = mx.symbol.Activation(name='res4b22_branch2a_relu', data=scale4b22_branch2a , act_type='relu')
    res4b22_branch2b = mx.symbol.Convolution(name='res4b22_branch2b', data=res4b22_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b22_branch2b = mx.symbol.BatchNorm(name='bn4b22_branch2b', data=res4b22_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b22_branch2b = bn4b22_branch2b
    res4b22_branch2b_relu = mx.symbol.Activation(name='res4b22_branch2b_relu', data=scale4b22_branch2b , act_type='relu')
    res4b22_branch2c = mx.symbol.Convolution(name='res4b22_branch2c', data=res4b22_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b22_branch2c = mx.symbol.BatchNorm(name='bn4b22_branch2c', data=res4b22_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b22_branch2c = bn4b22_branch2c
    res4b22 = mx.symbol.broadcast_add(name='res4b22', *[res4b21_relu,scale4b22_branch2c] )
    res4b22_relu = mx.symbol.Activation(name='res4b22_relu', data=res4b22 , act_type='relu')
    res4b23_branch2a = mx.symbol.Convolution(name='res4b23_branch2a', data=res4b22_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b23_branch2a = mx.symbol.BatchNorm(name='bn4b23_branch2a', data=res4b23_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b23_branch2a = bn4b23_branch2a
    res4b23_branch2a_relu = mx.symbol.Activation(name='res4b23_branch2a_relu', data=scale4b23_branch2a , act_type='relu')
    res4b23_branch2b = mx.symbol.Convolution(name='res4b23_branch2b', data=res4b23_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b23_branch2b = mx.symbol.BatchNorm(name='bn4b23_branch2b', data=res4b23_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b23_branch2b = bn4b23_branch2b
    res4b23_branch2b_relu = mx.symbol.Activation(name='res4b23_branch2b_relu', data=scale4b23_branch2b , act_type='relu')
    res4b23_branch2c = mx.symbol.Convolution(name='res4b23_branch2c', data=res4b23_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b23_branch2c = mx.symbol.BatchNorm(name='bn4b23_branch2c', data=res4b23_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b23_branch2c = bn4b23_branch2c
    res4b23 = mx.symbol.broadcast_add(name='res4b23', *[res4b22_relu,scale4b23_branch2c] )
    res4b23_relu = mx.symbol.Activation(name='res4b23_relu', data=res4b23 , act_type='relu')
    res4b24_branch2a = mx.symbol.Convolution(name='res4b24_branch2a', data=res4b23_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b24_branch2a = mx.symbol.BatchNorm(name='bn4b24_branch2a', data=res4b24_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b24_branch2a = bn4b24_branch2a
    res4b24_branch2a_relu = mx.symbol.Activation(name='res4b24_branch2a_relu', data=scale4b24_branch2a , act_type='relu')
    res4b24_branch2b = mx.symbol.Convolution(name='res4b24_branch2b', data=res4b24_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b24_branch2b = mx.symbol.BatchNorm(name='bn4b24_branch2b', data=res4b24_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b24_branch2b = bn4b24_branch2b
    res4b24_branch2b_relu = mx.symbol.Activation(name='res4b24_branch2b_relu', data=scale4b24_branch2b , act_type='relu')
    res4b24_branch2c = mx.symbol.Convolution(name='res4b24_branch2c', data=res4b24_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b24_branch2c = mx.symbol.BatchNorm(name='bn4b24_branch2c', data=res4b24_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b24_branch2c = bn4b24_branch2c
    res4b24 = mx.symbol.broadcast_add(name='res4b24', *[res4b23_relu,scale4b24_branch2c] )
    res4b24_relu = mx.symbol.Activation(name='res4b24_relu', data=res4b24 , act_type='relu')
    res4b25_branch2a = mx.symbol.Convolution(name='res4b25_branch2a', data=res4b24_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b25_branch2a = mx.symbol.BatchNorm(name='bn4b25_branch2a', data=res4b25_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b25_branch2a = bn4b25_branch2a
    res4b25_branch2a_relu = mx.symbol.Activation(name='res4b25_branch2a_relu', data=scale4b25_branch2a , act_type='relu')
    res4b25_branch2b = mx.symbol.Convolution(name='res4b25_branch2b', data=res4b25_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b25_branch2b = mx.symbol.BatchNorm(name='bn4b25_branch2b', data=res4b25_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b25_branch2b = bn4b25_branch2b
    res4b25_branch2b_relu = mx.symbol.Activation(name='res4b25_branch2b_relu', data=scale4b25_branch2b , act_type='relu')
    res4b25_branch2c = mx.symbol.Convolution(name='res4b25_branch2c', data=res4b25_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b25_branch2c = mx.symbol.BatchNorm(name='bn4b25_branch2c', data=res4b25_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b25_branch2c = bn4b25_branch2c
    res4b25 = mx.symbol.broadcast_add(name='res4b25', *[res4b24_relu,scale4b25_branch2c] )
    res4b25_relu = mx.symbol.Activation(name='res4b25_relu', data=res4b25 , act_type='relu')
    res4b26_branch2a = mx.symbol.Convolution(name='res4b26_branch2a', data=res4b25_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b26_branch2a = mx.symbol.BatchNorm(name='bn4b26_branch2a', data=res4b26_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b26_branch2a = bn4b26_branch2a
    res4b26_branch2a_relu = mx.symbol.Activation(name='res4b26_branch2a_relu', data=scale4b26_branch2a , act_type='relu')
    res4b26_branch2b = mx.symbol.Convolution(name='res4b26_branch2b', data=res4b26_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b26_branch2b = mx.symbol.BatchNorm(name='bn4b26_branch2b', data=res4b26_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b26_branch2b = bn4b26_branch2b
    res4b26_branch2b_relu = mx.symbol.Activation(name='res4b26_branch2b_relu', data=scale4b26_branch2b , act_type='relu')
    res4b26_branch2c = mx.symbol.Convolution(name='res4b26_branch2c', data=res4b26_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b26_branch2c = mx.symbol.BatchNorm(name='bn4b26_branch2c', data=res4b26_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b26_branch2c = bn4b26_branch2c
    res4b26 = mx.symbol.broadcast_add(name='res4b26', *[res4b25_relu,scale4b26_branch2c] )
    res4b26_relu = mx.symbol.Activation(name='res4b26_relu', data=res4b26 , act_type='relu')
    res4b27_branch2a = mx.symbol.Convolution(name='res4b27_branch2a', data=res4b26_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b27_branch2a = mx.symbol.BatchNorm(name='bn4b27_branch2a', data=res4b27_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b27_branch2a = bn4b27_branch2a
    res4b27_branch2a_relu = mx.symbol.Activation(name='res4b27_branch2a_relu', data=scale4b27_branch2a , act_type='relu')
    res4b27_branch2b = mx.symbol.Convolution(name='res4b27_branch2b', data=res4b27_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b27_branch2b = mx.symbol.BatchNorm(name='bn4b27_branch2b', data=res4b27_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b27_branch2b = bn4b27_branch2b
    res4b27_branch2b_relu = mx.symbol.Activation(name='res4b27_branch2b_relu', data=scale4b27_branch2b , act_type='relu')
    res4b27_branch2c = mx.symbol.Convolution(name='res4b27_branch2c', data=res4b27_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b27_branch2c = mx.symbol.BatchNorm(name='bn4b27_branch2c', data=res4b27_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b27_branch2c = bn4b27_branch2c
    res4b27 = mx.symbol.broadcast_add(name='res4b27', *[res4b26_relu,scale4b27_branch2c] )
    res4b27_relu = mx.symbol.Activation(name='res4b27_relu', data=res4b27 , act_type='relu')
    res4b28_branch2a = mx.symbol.Convolution(name='res4b28_branch2a', data=res4b27_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b28_branch2a = mx.symbol.BatchNorm(name='bn4b28_branch2a', data=res4b28_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b28_branch2a = bn4b28_branch2a
    res4b28_branch2a_relu = mx.symbol.Activation(name='res4b28_branch2a_relu', data=scale4b28_branch2a , act_type='relu')
    res4b28_branch2b = mx.symbol.Convolution(name='res4b28_branch2b', data=res4b28_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b28_branch2b = mx.symbol.BatchNorm(name='bn4b28_branch2b', data=res4b28_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b28_branch2b = bn4b28_branch2b
    res4b28_branch2b_relu = mx.symbol.Activation(name='res4b28_branch2b_relu', data=scale4b28_branch2b , act_type='relu')
    res4b28_branch2c = mx.symbol.Convolution(name='res4b28_branch2c', data=res4b28_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b28_branch2c = mx.symbol.BatchNorm(name='bn4b28_branch2c', data=res4b28_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b28_branch2c = bn4b28_branch2c
    res4b28 = mx.symbol.broadcast_add(name='res4b28', *[res4b27_relu,scale4b28_branch2c] )
    res4b28_relu = mx.symbol.Activation(name='res4b28_relu', data=res4b28 , act_type='relu')
    res4b29_branch2a = mx.symbol.Convolution(name='res4b29_branch2a', data=res4b28_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b29_branch2a = mx.symbol.BatchNorm(name='bn4b29_branch2a', data=res4b29_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b29_branch2a = bn4b29_branch2a
    res4b29_branch2a_relu = mx.symbol.Activation(name='res4b29_branch2a_relu', data=scale4b29_branch2a , act_type='relu')
    res4b29_branch2b = mx.symbol.Convolution(name='res4b29_branch2b', data=res4b29_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b29_branch2b = mx.symbol.BatchNorm(name='bn4b29_branch2b', data=res4b29_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b29_branch2b = bn4b29_branch2b
    res4b29_branch2b_relu = mx.symbol.Activation(name='res4b29_branch2b_relu', data=scale4b29_branch2b , act_type='relu')
    res4b29_branch2c = mx.symbol.Convolution(name='res4b29_branch2c', data=res4b29_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b29_branch2c = mx.symbol.BatchNorm(name='bn4b29_branch2c', data=res4b29_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b29_branch2c = bn4b29_branch2c
    res4b29 = mx.symbol.broadcast_add(name='res4b29', *[res4b28_relu,scale4b29_branch2c] )
    res4b29_relu = mx.symbol.Activation(name='res4b29_relu', data=res4b29 , act_type='relu')
    res4b30_branch2a = mx.symbol.Convolution(name='res4b30_branch2a', data=res4b29_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b30_branch2a = mx.symbol.BatchNorm(name='bn4b30_branch2a', data=res4b30_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b30_branch2a = bn4b30_branch2a
    res4b30_branch2a_relu = mx.symbol.Activation(name='res4b30_branch2a_relu', data=scale4b30_branch2a , act_type='relu')
    res4b30_branch2b = mx.symbol.Convolution(name='res4b30_branch2b', data=res4b30_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b30_branch2b = mx.symbol.BatchNorm(name='bn4b30_branch2b', data=res4b30_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b30_branch2b = bn4b30_branch2b
    res4b30_branch2b_relu = mx.symbol.Activation(name='res4b30_branch2b_relu', data=scale4b30_branch2b , act_type='relu')
    res4b30_branch2c = mx.symbol.Convolution(name='res4b30_branch2c', data=res4b30_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b30_branch2c = mx.symbol.BatchNorm(name='bn4b30_branch2c', data=res4b30_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b30_branch2c = bn4b30_branch2c
    res4b30 = mx.symbol.broadcast_add(name='res4b30', *[res4b29_relu,scale4b30_branch2c] )
    res4b30_relu = mx.symbol.Activation(name='res4b30_relu', data=res4b30 , act_type='relu')
    res4b31_branch2a = mx.symbol.Convolution(name='res4b31_branch2a', data=res4b30_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b31_branch2a = mx.symbol.BatchNorm(name='bn4b31_branch2a', data=res4b31_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b31_branch2a = bn4b31_branch2a
    res4b31_branch2a_relu = mx.symbol.Activation(name='res4b31_branch2a_relu', data=scale4b31_branch2a , act_type='relu')
    res4b31_branch2b = mx.symbol.Convolution(name='res4b31_branch2b', data=res4b31_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b31_branch2b = mx.symbol.BatchNorm(name='bn4b31_branch2b', data=res4b31_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b31_branch2b = bn4b31_branch2b
    res4b31_branch2b_relu = mx.symbol.Activation(name='res4b31_branch2b_relu', data=scale4b31_branch2b , act_type='relu')
    res4b31_branch2c = mx.symbol.Convolution(name='res4b31_branch2c', data=res4b31_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b31_branch2c = mx.symbol.BatchNorm(name='bn4b31_branch2c', data=res4b31_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b31_branch2c = bn4b31_branch2c
    res4b31 = mx.symbol.broadcast_add(name='res4b31', *[res4b30_relu,scale4b31_branch2c] )
    res4b31_relu = mx.symbol.Activation(name='res4b31_relu', data=res4b31 , act_type='relu')
    res4b32_branch2a = mx.symbol.Convolution(name='res4b32_branch2a', data=res4b31_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b32_branch2a = mx.symbol.BatchNorm(name='bn4b32_branch2a', data=res4b32_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b32_branch2a = bn4b32_branch2a
    res4b32_branch2a_relu = mx.symbol.Activation(name='res4b32_branch2a_relu', data=scale4b32_branch2a , act_type='relu')
    res4b32_branch2b = mx.symbol.Convolution(name='res4b32_branch2b', data=res4b32_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b32_branch2b = mx.symbol.BatchNorm(name='bn4b32_branch2b', data=res4b32_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b32_branch2b = bn4b32_branch2b
    res4b32_branch2b_relu = mx.symbol.Activation(name='res4b32_branch2b_relu', data=scale4b32_branch2b , act_type='relu')
    res4b32_branch2c = mx.symbol.Convolution(name='res4b32_branch2c', data=res4b32_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b32_branch2c = mx.symbol.BatchNorm(name='bn4b32_branch2c', data=res4b32_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b32_branch2c = bn4b32_branch2c
    res4b32 = mx.symbol.broadcast_add(name='res4b32', *[res4b31_relu,scale4b32_branch2c] )
    res4b32_relu = mx.symbol.Activation(name='res4b32_relu', data=res4b32 , act_type='relu')
    res4b33_branch2a = mx.symbol.Convolution(name='res4b33_branch2a', data=res4b32_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b33_branch2a = mx.symbol.BatchNorm(name='bn4b33_branch2a', data=res4b33_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b33_branch2a = bn4b33_branch2a
    res4b33_branch2a_relu = mx.symbol.Activation(name='res4b33_branch2a_relu', data=scale4b33_branch2a , act_type='relu')
    res4b33_branch2b = mx.symbol.Convolution(name='res4b33_branch2b', data=res4b33_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b33_branch2b = mx.symbol.BatchNorm(name='bn4b33_branch2b', data=res4b33_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b33_branch2b = bn4b33_branch2b
    res4b33_branch2b_relu = mx.symbol.Activation(name='res4b33_branch2b_relu', data=scale4b33_branch2b , act_type='relu')
    res4b33_branch2c = mx.symbol.Convolution(name='res4b33_branch2c', data=res4b33_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b33_branch2c = mx.symbol.BatchNorm(name='bn4b33_branch2c', data=res4b33_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b33_branch2c = bn4b33_branch2c
    res4b33 = mx.symbol.broadcast_add(name='res4b33', *[res4b32_relu,scale4b33_branch2c] )
    res4b33_relu = mx.symbol.Activation(name='res4b33_relu', data=res4b33 , act_type='relu')
    res4b34_branch2a = mx.symbol.Convolution(name='res4b34_branch2a', data=res4b33_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b34_branch2a = mx.symbol.BatchNorm(name='bn4b34_branch2a', data=res4b34_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b34_branch2a = bn4b34_branch2a
    res4b34_branch2a_relu = mx.symbol.Activation(name='res4b34_branch2a_relu', data=scale4b34_branch2a , act_type='relu')
    res4b34_branch2b = mx.symbol.Convolution(name='res4b34_branch2b', data=res4b34_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b34_branch2b = mx.symbol.BatchNorm(name='bn4b34_branch2b', data=res4b34_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b34_branch2b = bn4b34_branch2b
    res4b34_branch2b_relu = mx.symbol.Activation(name='res4b34_branch2b_relu', data=scale4b34_branch2b , act_type='relu')
    res4b34_branch2c = mx.symbol.Convolution(name='res4b34_branch2c', data=res4b34_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b34_branch2c = mx.symbol.BatchNorm(name='bn4b34_branch2c', data=res4b34_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b34_branch2c = bn4b34_branch2c
    res4b34 = mx.symbol.broadcast_add(name='res4b34', *[res4b33_relu,scale4b34_branch2c] )
    res4b34_relu = mx.symbol.Activation(name='res4b34_relu', data=res4b34 , act_type='relu')
    res4b35_branch2a = mx.symbol.Convolution(name='res4b35_branch2a', data=res4b34_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b35_branch2a = mx.symbol.BatchNorm(name='bn4b35_branch2a', data=res4b35_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b35_branch2a = bn4b35_branch2a
    res4b35_branch2a_relu = mx.symbol.Activation(name='res4b35_branch2a_relu', data=scale4b35_branch2a , act_type='relu')
    res4b35_branch2b = mx.symbol.Convolution(name='res4b35_branch2b', data=res4b35_branch2a_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b35_branch2b = mx.symbol.BatchNorm(name='bn4b35_branch2b', data=res4b35_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b35_branch2b = bn4b35_branch2b
    res4b35_branch2b_relu = mx.symbol.Activation(name='res4b35_branch2b_relu', data=scale4b35_branch2b , act_type='relu')
    res4b35_branch2c = mx.symbol.Convolution(name='res4b35_branch2c', data=res4b35_branch2b_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b35_branch2c = mx.symbol.BatchNorm(name='bn4b35_branch2c', data=res4b35_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale4b35_branch2c = bn4b35_branch2c
    res4b35 = mx.symbol.broadcast_add(name='res4b35', *[res4b34_relu,scale4b35_branch2c] )
    res4b35_relu = mx.symbol.Activation(name='res4b35_relu', data=res4b35 , act_type='relu')
    res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=res4b35_relu , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
    bn5a_branch1 = mx.symbol.BatchNorm(name='bn5a_branch1', data=res5a_branch1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale5a_branch1 = bn5a_branch1
    res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=res4b35_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
    bn5a_branch2a = mx.symbol.BatchNorm(name='bn5a_branch2a', data=res5a_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale5a_branch2a = bn5a_branch2a
    res5a_branch2a_relu = mx.symbol.Activation(name='res5a_branch2a_relu', data=scale5a_branch2a , act_type='relu')
    res5a_branch2b = mx.symbol.Convolution(name='res5a_branch2b', data=res5a_branch2a_relu , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn5a_branch2b = mx.symbol.BatchNorm(name='bn5a_branch2b', data=res5a_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale5a_branch2b = bn5a_branch2b
    res5a_branch2b_relu = mx.symbol.Activation(name='res5a_branch2b_relu', data=scale5a_branch2b , act_type='relu')
    res5a_branch2c = mx.symbol.Convolution(name='res5a_branch2c', data=res5a_branch2b_relu , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn5a_branch2c = mx.symbol.BatchNorm(name='bn5a_branch2c', data=res5a_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale5a_branch2c = bn5a_branch2c
    res5a = mx.symbol.broadcast_add(name='res5a', *[scale5a_branch1,scale5a_branch2c] )
    res5a_relu = mx.symbol.Activation(name='res5a_relu', data=res5a , act_type='relu')
    res5b_branch2a = mx.symbol.Convolution(name='res5b_branch2a', data=res5a_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn5b_branch2a = mx.symbol.BatchNorm(name='bn5b_branch2a', data=res5b_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale5b_branch2a = bn5b_branch2a
    res5b_branch2a_relu = mx.symbol.Activation(name='res5b_branch2a_relu', data=scale5b_branch2a , act_type='relu')
    res5b_branch2b = mx.symbol.Convolution(name='res5b_branch2b', data=res5b_branch2a_relu , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn5b_branch2b = mx.symbol.BatchNorm(name='bn5b_branch2b', data=res5b_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale5b_branch2b = bn5b_branch2b
    res5b_branch2b_relu = mx.symbol.Activation(name='res5b_branch2b_relu', data=scale5b_branch2b , act_type='relu')
    res5b_branch2c = mx.symbol.Convolution(name='res5b_branch2c', data=res5b_branch2b_relu , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn5b_branch2c = mx.symbol.BatchNorm(name='bn5b_branch2c', data=res5b_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale5b_branch2c = bn5b_branch2c
    res5b = mx.symbol.broadcast_add(name='res5b', *[res5a_relu,scale5b_branch2c] )
    res5b_relu = mx.symbol.Activation(name='res5b_relu', data=res5b , act_type='relu')
    res5c_branch2a = mx.symbol.Convolution(name='res5c_branch2a', data=res5b_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn5c_branch2a = mx.symbol.BatchNorm(name='bn5c_branch2a', data=res5c_branch2a , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale5c_branch2a = bn5c_branch2a
    res5c_branch2a_relu = mx.symbol.Activation(name='res5c_branch2a_relu', data=scale5c_branch2a , act_type='relu')
    res5c_branch2b = mx.symbol.Convolution(name='res5c_branch2b', data=res5c_branch2a_relu , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn5c_branch2b = mx.symbol.BatchNorm(name='bn5c_branch2b', data=res5c_branch2b , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale5c_branch2b = bn5c_branch2b
    res5c_branch2b_relu = mx.symbol.Activation(name='res5c_branch2b_relu', data=scale5c_branch2b , act_type='relu')
    res5c_branch2c = mx.symbol.Convolution(name='res5c_branch2c', data=res5c_branch2b_relu , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn5c_branch2c = mx.symbol.BatchNorm(name='bn5c_branch2c', data=res5c_branch2c , use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale5c_branch2c = bn5c_branch2c
    res5c = mx.symbol.broadcast_add(name='res5c', *[res5b_relu,scale5c_branch2c] )
    res5c_relu = mx.symbol.Activation(name='res5c_relu', data=res5c , act_type='relu')
    pool5 = mx.symbol.Pooling(name='pool5', data=res5c_relu , pooling_convention='full', pad=(0,0), kernel=(7,7), stride=(1,1), pool_type='avg')
    flatten_0=mx.symbol.Flatten(name='flatten_0', data=pool5)
    fc365 = mx.symbol.FullyConnected(name='fc365', data=flatten_0 , num_hidden=365, no_bias=False)
    prob = mx.symbol.SoftmaxOutput(name='prob', data=fc365 )

    return prob


def get_symbol():
    return resnet152()

