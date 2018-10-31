# build non local block
# refer to https://github.com/AlexHex7/Non-local_pytorch/blob/master/lib/non_local_simple_version.py
import sys
sys.path.insert(0, '/media/chencp/data_ssd2/work/incubator-mxnet/python/')
import mxnet as mx


def non_local_block(data, in_channels, inter_channels=None, lr_mult_nonlocal=1.0,
                    sub_sample=True, bn_non_local=True, bn_mom=0.99, name_prefix=''):
    # print(in_channels)
    # data, [b, c, h, w]
    if inter_channels is None:
        inter_channels = in_channels // 2
        if inter_channels == 0:
            inter_channels = 1

    g = mx.symbol.Convolution(name='{}g_conv'.format(name_prefix), data=data, num_filter=inter_channels, pad=(0, 0),
                              kernel=(1, 1), stride=(1, 1), lr_mult=lr_mult_nonlocal)
    theta = mx.symbol.Convolution(name='{}theta_conv'.format(name_prefix), data=data, num_filter=inter_channels,
                                  pad=(0, 0), kernel=(1, 1), stride=(1, 1), lr_mult=lr_mult_nonlocal)
    phi = mx.symbol.Convolution(name='{}phi_conv'.format(name_prefix), data=data, num_filter=inter_channels, pad=(0, 0),
                                kernel=(1, 1), stride=(1, 1), lr_mult=lr_mult_nonlocal)
    if sub_sample:
        g = mx.sym.Pooling(name='{}g_pool'.format(name_prefix), data=g, pooling_convention='full', pad=(0, 0),
                           kernel=(2, 2), stride=(2, 2), pool_type='max')
        phi = mx.sym.Pooling(name='{}phi_pool'.format(name_prefix), data=phi, pooling_convention='full', pad=(0, 0),
                             kernel=(2, 2), stride=(2, 2), pool_type='max')
    # g_x, [b, c/2, h*w/4]
    g_x = mx.symbol.Reshape(data=g, shape=(0, 0, -1), name='{}g_reshape'.format(name_prefix))
    phi_x = mx.sym.Reshape(data=phi, shape=(0, 0, -1), name='{}phi_reshape'.format(name_prefix))
    # theta_x, [b, c/2, h*w]
    theta_x = mx.sym.Reshape(data=theta, shape=(0, 0, -1), name='{}theta_reshape'.format(name_prefix))

    # f, [b, h*w, h*w/4]
    f = mx.sym.batch_dot(lhs=theta_x, rhs=phi_x, transpose_a=True, name='{}batch_dot_f'.format(name_prefix))
    f_dic_c = mx.sym.softmax(data=f, axis=2, name='{}softmax_f'.format(name_prefix))

    # y, [b, c/2, h*w]
    y = mx.sym.batch_dot(lhs=g_x, rhs=f_dic_c, transpose_b=True, name='{}batch_dot_y'.format(name_prefix))
    y = mx.sym.reshape_like(lhs=y, rhs=data, lhs_begin=-1, lhs_end=None, rhs_begin=-2, rhs_end=None,
                            name='{}reshape_like'.format(name_prefix))

    w = mx.symbol.Convolution(name='{}w_conv'.format(name_prefix), data=y, num_filter=in_channels,
                              kernel=(1, 1), pad=(0, 0), stride=(1, 1), lr_mult=lr_mult_nonlocal)
    if bn_non_local:
        # need to init as zeros
        w = mx.sym.BatchNorm(data=w, fix_gamma=False, eps=1e-5, momentum=bn_mom, name='{}bn'.format(name_prefix),
                             lr_mult=lr_mult_nonlocal)

    return data + w


def non_local_block_v2(data, in_channels, inter_channels=None, lr_mult_nonlocal=1.0,
                       sub_sample=True, bn_non_local=True, bn_mom=0.99, name_prefix=''):
    """modified for diffusion version of non local
      refer to paper, NIPS 2018

      diff: gather diff features instead of original features
      by chencp, on 2018.10.31
    """
    print('non local block with normarlized data')
    # print(in_channels)
    # data, [b, c, h, w]
    if inter_channels is None:
        inter_channels = in_channels // 2
        if inter_channels == 0:
            inter_channels = 1

    conv_w = mx.symbol.Variable('{}w_conv_weight'.format(name_prefix), shape=(in_channels, inter_channels, 1, 1))
    conv_b = mx.symbol.Variable('{}w_conv_bias'.format(name_prefix), shape=(in_channels))
    conv_g_w = mx.symbol.Variable('{}g_conv_weight'.format(name_prefix), shape=(inter_channels, in_channels, 1, 1))
    conv_g_b = mx.symbol.Variable('{}g_conv_bias'.format(name_prefix), shape=(inter_channels))

    conv_w_reshape = mx.sym.Reshape(data=conv_w, shape=(0, 0))
    conv_g_w_reshape = mx.sym.Reshape(data=conv_g_w, shape=(0, 0))

    g = mx.symbol.Convolution(name='{}g_conv'.format(name_prefix),
                              weight=conv_g_w, bias=conv_g_b,
                              data=data, num_filter=inter_channels, pad=(0, 0),
                              kernel=(1, 1), stride=(1, 1), lr_mult=lr_mult_nonlocal)
    theta = mx.symbol.Convolution(name='{}theta_conv'.format(name_prefix), data=data, num_filter=inter_channels,
                                  pad=(0, 0), kernel=(1, 1), stride=(1, 1), lr_mult=lr_mult_nonlocal)
    phi = mx.symbol.Convolution(name='{}phi_conv'.format(name_prefix), data=data, num_filter=inter_channels, pad=(0, 0),
                                kernel=(1, 1), stride=(1, 1), lr_mult=lr_mult_nonlocal)
    if sub_sample:
        g = mx.sym.Pooling(name='{}g_pool'.format(name_prefix), data=g, pooling_convention='full', pad=(0, 0),
                           kernel=(2, 2), stride=(2, 2), pool_type='max')
        phi = mx.sym.Pooling(name='{}phi_pool'.format(name_prefix), data=phi, pooling_convention='full', pad=(0, 0),
                             kernel=(2, 2), stride=(2, 2), pool_type='max')
    # g_x, [b, c/2, h*w/4]
    g_x = mx.symbol.Reshape(data=g, shape=(0, 0, -1), name='{}g_reshape'.format(name_prefix))
    phi_x = mx.sym.Reshape(data=phi, shape=(0, 0, -1), name='{}phi_reshape'.format(name_prefix))
    # theta_x, [b, c/2, h*w]
    theta_x = mx.sym.Reshape(data=theta, shape=(0, 0, -1), name='{}theta_reshape'.format(name_prefix))

    # f, [b, h*w, h*w/4]
    f = mx.sym.batch_dot(lhs=theta_x, rhs=phi_x, transpose_a=True, name='{}batch_dot_f'.format(name_prefix))
    f_dic_c = mx.sym.softmax(data=f, axis=2, name='{}softmax_f'.format(name_prefix))

    # y, [b, c/2, h*w]
    y = mx.sym.batch_dot(lhs=g_x, rhs=f_dic_c, transpose_b=True, name='{}batch_dot_y'.format(name_prefix))
    y = mx.sym.reshape_like(lhs=y, rhs=data, lhs_begin=-1, lhs_end=None, rhs_begin=-2, rhs_end=None,
                            name='{}reshape_like'.format(name_prefix))

    mat_dot = mx.sym.dot(lhs=conv_w_reshape, rhs=conv_g_w_reshape, name='{}low_rank_mat')
    mat_dot = mx.sym.Reshape(data=mat_dot, shape=(0, 0, 1, 1))
    norm_w = mx.sym.Convolution(data=data, weight=mat_dot, bias=None,
                                num_filter=in_channels, kernel=(1, 1), pad=(0, 0), stride=(1, 1),
                                name='{}norm_data'.format(name_prefix))

    w = mx.symbol.Convolution(name='{}w_conv'.format(name_prefix),
                              weight=conv_w, bias=conv_b,
                              data=y, num_filter=in_channels,
                              kernel=(1, 1), pad=(0, 0), stride=(1, 1), lr_mult=lr_mult_nonlocal)
    w = w - norm_w

    if bn_non_local:
        # need to init as zeros
        w = mx.sym.BatchNorm(data=w, fix_gamma=False, eps=1e-5, momentum=bn_mom, name='{}bn'.format(name_prefix),
                             lr_mult=lr_mult_nonlocal)

    return data + w


if __name__ == '__main__':
    data = mx.symbol.Variable(name='data')
    in_channels = 64
    sym = non_local_block_v2(data, in_channels)
    mx.viz.print_summary(symbol=sym, shape={"data": (1, in_channels, 224, 224)})
