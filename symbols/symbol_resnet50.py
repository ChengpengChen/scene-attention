'''
Reproducing paper:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
import sys
sys.path.insert(0, '/media/chencp/data_ssd2/work/incubator-mxnet/python/')
import mxnet as mx
from utils.symbol import Symbol
from symbols.symbol_nonlocalblock import non_local_block, non_local_block_v2


class ResnetClass(Symbol):
    """resnet wrap
    """
    def __init__(self):
        self.eps = 1e-5

    def residual_unit(self, data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.99, workspace=512,
                      memonger=False):
        """Return ResNet Unit symbol for building ResNet
        Parameters
        ----------
        data : str
            Input data
        num_filter : int
            Number of output channels
        bnf : int
            Bottle neck channels factor with regard to num_filter
        stride : tupe
            Stride used in convolution
        dim_match : Boolen
            True means channel number between input and output is the same, otherwise means differ
        name : str
            Base name of the operators
        workspace : int
            Workspace used in convolution operator
        """
        if bottle_neck:
            conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                                       pad=(0, 0),
                                       no_bias=1, workspace=workspace, name=name + '_conv1')
            bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=1e-5, momentum=bn_mom, name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
            conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride,
                                       pad=(1, 1),
                                       no_bias=1, workspace=workspace, name=name + '_conv2')
            bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=1e-5, momentum=bn_mom, name=name + '_bn2')
            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
            conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                       no_bias=1,
                                       workspace=workspace, name=name + '_conv3')
            bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=1e-5, momentum=bn_mom, name=name + '_bn3')
            # act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
            if dim_match:
                shortcut = data
            else:
                conv_shortcut = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride,
                                                   no_bias=1, workspace=workspace, name=name + '_downsample_0')
                shortcut = mx.sym.BatchNorm(data=conv_shortcut, fix_gamma=False, eps=1e-5,
                                            momentum=bn_mom, name=name + '_downsample_1')
            if memonger:
                shortcut._set_attr(mirror_stage='True')
            act3 = mx.sym.Activation(data=bn3 + shortcut, act_type='relu', name=name + '_relu3')

            return act3
        else:
            raise NotImplementedError
            # bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '.bn1')
            # act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '.relu1')
            # conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
            #                            no_bias=1, workspace=workspace, name=name + '.conv1')
            # bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '.bn2')
            # act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '.relu2')
            # conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
            #                            no_bias=1, workspace=workspace, name=name + '.conv2')
            # if dim_match:
            #     shortcut = data
            # else:
            #     shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=1,
            #                                   workspace=workspace, name=name + '.sc')
            # if memonger:
            #     shortcut._set_attr(mirror_stage='True')
            # return conv2 + shortcut

    def resnet(self, units, num_stage, filter_list, data_type, bottle_neck=True, bn_mom=0.99, workspace=1024, memonger=False,
               bn_data_layer=True, num_non_local_block=0, lr_mult_nonlocal=1.0, bn_non_local=False,
               nonlocal_sub_sample=True, nonlocal_norm=False):
        """Return ResNet symbol of cifar10 and imagenet
        Parameters
        ----------
        units : list
            Number of units in each stage
        num_stage : int
            Number of stage
        filter_list : list
            Channel size of each stage
        num_class : int
            Ouput size of symbol
        dataset : str
            Dataset type, only cifar10 and imagenet supports
        workspace : int
            Workspace used in convolution operator
        """
        non_local_block_func = non_local_block_v2 if nonlocal_norm else non_local_block
        num_unit = len(units)
        assert (num_unit == num_stage)
        data = mx.sym.Variable(name='data')  # remove bn on data, same as original paper
        if bn_data_layer:
            data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=1e-5, momentum=bn_mom, name='bn_data')
        if data_type == 'cifar10':
            body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                      no_bias=1, name="conv1", workspace=workspace)
        elif data_type == 'imagenet':
            body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                      no_bias=1, name="conv1", workspace=workspace)
            body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=1e-5, momentum=bn_mom, name='bn1')
            body = mx.sym.Activation(data=body, act_type='relu', name='relu')
            body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='maxpool')
        else:
            raise ValueError("do not support {} yet".format(data_type))
        for i in range(num_stage):
            body = self.residual_unit(body, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                                      name='layer%d_%d' % (i + 1, 0), bottle_neck=bottle_neck, workspace=workspace,
                                      memonger=memonger, bn_mom=bn_mom)
            # body = self.residual_unit(body, filter_list[i + 1], (1 if i == 0 or i == num_stage-1 else 2, 1 if i == 0 or i == num_stage-1 else 2), False,
            #                           name='layer%d_%d' % (i + 1, 0), bottle_neck=bottle_neck, workspace=workspace,
            #                           memonger=memonger, bn_mom=bn_mom)
            if (i == num_stage - 2 or i == num_stage - 3) and num_non_local_block == 10:
                body = non_local_block_func(data=body, in_channels=filter_list[i + 1],
                                            name_prefix='layer%d_%d' % (i + 1, 0) + '_nonlocal_',
                                            lr_mult_nonlocal=lr_mult_nonlocal, bn_non_local=bn_non_local,
                                            sub_sample=nonlocal_sub_sample)
            for j in range(units[i] - 1):
                body = self.residual_unit(body, filter_list[i + 1], (1, 1), True, name='layer%d_%d' % (i + 1, j + 1),
                                          bottle_neck=bottle_neck, workspace=workspace, memonger=memonger, bn_mom=bn_mom)
                if (i == num_stage - 2 or i == num_stage - 3) and \
                    (num_non_local_block == 10 or (num_non_local_block == 5 and j % 2 == 0)):
                    body = non_local_block_func(data=body, in_channels=filter_list[i+1],
                                                name_prefix='layer%d_%d' % (i + 1,  j + 1) + '_nonlocal_',
                                                lr_mult_nonlocal=lr_mult_nonlocal, bn_non_local=bn_non_local,
                                                sub_sample=nonlocal_sub_sample)

            # only add non local block as stage 4 for num_non_local_block=1
            if num_non_local_block == 1 and i == num_stage - 2:
                body = non_local_block_func(data=body, in_channels=filter_list[i+1],
                                            name_prefix='layer%d_%d' % (i + 1, units[i]-1) + '_nonlocal_',
                                            lr_mult_nonlocal=lr_mult_nonlocal, bn_non_local=bn_non_local,
                                            sub_sample=nonlocal_sub_sample)

        # output = mx.symbol.Pooling(data=body, global_pool=True, pool_type='avg', name='global_avg', kernel=(1, 1))
        # output = mx.symbol.FullyConnected(data=output, num_hidden=365, name='fc')

        return body

    def resnet_rois(self, units, num_stage, filter_list, data_type, bottle_neck=True, bn_mom=0.9, workspace=1024,
                    memonger=False, bn_data_layer=True):
        """Return ResNet symbol of cifar10 and imagenet, with extra rois input (same as rcnn)
        Parameters
        ----------
        units : list
            Number of units in each stage
        num_stage : int
            Number of stage
        filter_list : list
            Channel size of each stage
        num_class : int
            Ouput size of symbol
        dataset : str
            Dataset type, only cifar10 and imagenet supports
        workspace : int
            Workspace used in convolution operator
        """
        num_unit = len(units)
        assert (num_unit == num_stage)
        data = mx.sym.Variable(name='data')  # remove bn on data, same as original paper
        rois = mx.sym.Variable(name='rois')
        if bn_data_layer:
            data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=1e-5, momentum=bn_mom, name='bn_data')
        if data_type == 'cifar10':
            body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                      no_bias=1, name="conv1", workspace=workspace)
        elif data_type == 'imagenet':
            body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                      no_bias=1, name="conv1", workspace=workspace)
            body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=1e-5, momentum=bn_mom, name='bn1')
            body = mx.sym.Activation(data=body, act_type='relu', name='relu')
            body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='maxpool')
        else:
            raise ValueError("do not support {} yet".format(data_type))
        for i in range(num_stage-1):
            body = self.residual_unit(body, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                                      name='layer%d_%d' % (i + 1, 0), bottle_neck=bottle_neck, workspace=workspace,
                                      memonger=memonger, bn_mom=bn_mom)
            for j in range(units[i] - 1):
                body = self.residual_unit(body, filter_list[i + 1], (1, 1), True, name='layer%d_%d' % (i + 1, j + 1),
                                          bottle_neck=bottle_neck, workspace=workspace, memonger=memonger, bn_mom=bn_mom)

        # roi pooling
        conv_new_1 = mx.sym.Convolution(data=body, kernel=(1, 1), num_filter=256, name="conv_new_1")
        conv_new_1_relu = mx.sym.Activation(data=conv_new_1, act_type='relu', name='conv_new_1_relu')
        roi_pool = mx.symbol.ROIPooling(
            name='roi_pool', data=conv_new_1_relu, rois=rois, pooled_size=(7, 7), spatial_scale=0.0625)

        # stage 5, change stride of layer4_0 from (2, 2) to (1, 1)
        body = self.residual_unit(roi_pool, filter_list[num_stage], (1, 1), False,
                                  name='layer%d_%d' % (num_stage, 0), bottle_neck=bottle_neck, workspace=workspace,
                                  memonger=memonger, bn_mom=bn_mom)
        for j in range(units[num_stage-1] - 1):
            body = self.residual_unit(body, filter_list[num_stage], (1, 1), True, name='layer%d_%d' % (num_stage, j + 1),
                                      bottle_neck=bottle_neck, workspace=workspace, memonger=memonger, bn_mom=bn_mom)

        # output = mx.symbol.Pooling(data=body, global_pool=True, pool_type='avg', name='global_avg', kernel=(1, 1))
        # output = mx.symbol.FullyConnected(data=output, num_hidden=365, name='fc')

        return body

    def get_symbol(self, depth=50, bn_data_layer=True, bn_mom=0.99,
                   num_non_local_block=0, lr_mult_nonlocal=1.0, bn_non_local=False, nonlocal_sub_sample=True):
        if depth == 18:
            units = [2, 2, 2, 2]
        elif depth == 34:
            units = [3, 4, 6, 3]
        elif depth == 50:
            units = [3, 4, 6, 3]
        elif depth == 101:
            units = [3, 4, 23, 3]
        elif depth == 152:
            units = [3, 8, 36, 3]
        elif depth == 200:
            units = [3, 24, 36, 3]
        elif depth == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on detph {}, you can do it youself".format(depth))

        # note: the num_non_local_block is specific for resnet 50,
        # it should be to further design for other network (i.e., resnet101)
        assert num_non_local_block in [0, 1, 5, 10], 'the number of non_local_block should be in [0, 1, 5, 10]'

        symbol = self.resnet(units=units, num_stage=4, bn_mom=bn_mom,
                             filter_list=[64, 256, 512, 1024, 2048] if depth >= 50 else [64, 64, 128, 256, 512],
                             data_type="imagenet", bottle_neck=True if depth >= 50 else False, memonger=True,
                             bn_data_layer=bn_data_layer, num_non_local_block=num_non_local_block,
                             lr_mult_nonlocal=lr_mult_nonlocal, bn_non_local=bn_non_local,
                             nonlocal_sub_sample=nonlocal_sub_sample)
        self.sym = symbol
        return symbol

    def get_symbol_rois(self, depth=50, bn_data_layer=True):
        if depth == 18:
            units = [2, 2, 2, 2]
        elif depth == 34:
            units = [3, 4, 6, 3]
        elif depth == 50:
            units = [3, 4, 6, 3]
        elif depth == 101:
            units = [3, 4, 23, 3]
        elif depth == 152:
            units = [3, 8, 36, 3]
        elif depth == 200:
            units = [3, 24, 36, 3]
        elif depth == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on detph {}, you can do it youself".format(depth))

        symbol = self.resnet_rois(units=units, num_stage=4,
                                  filter_list=[64, 256, 512, 1024, 2048] if depth >= 50 else [64, 64, 128, 256, 512],
                                  data_type="imagenet", bottle_neck=True if depth >= 50 else False, memonger=True,
                                  bn_data_layer=bn_data_layer)
        self.sym = symbol
        return symbol


if __name__ == '__main__':
    # sym = ResnetClass().get_symbol(50, num_non_local_block=0)
    # mx.viz.print_summary(symbol=sym, shape={"data": (8, 3, 224, 224)})
    # a = mx.viz.plot_network(symbol=sym, shape={"data": (8, 3, 224, 224)}, title='resnet50_nonlocal_10block')
    # a.render()

    sym = ResnetClass().get_symbol_rois(50)
    mx.viz.print_summary(symbol=sym, shape={"data": (8, 3, 224, 224), "rois": (80, 5)})
    a = mx.viz.plot_network(symbol=sym, shape={"data": (8, 3, 224, 224), "rois": (80, 5)}, title='resnet50_rois')
    a.render()
