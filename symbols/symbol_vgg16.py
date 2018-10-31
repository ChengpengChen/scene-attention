import mxnet as mx
import math
from utils.symbol import Symbol


class vgg16(Symbol):
    def __init__(self):
        self.eps = 1e-5

    @staticmethod
    def get_vgg16_conv5(data):
        conv1_1 = mx.symbol.Convolution(name='conv1_1', data=data, num_filter=64, pad=(1, 1),
                                        kernel=(3, 3), stride=(1, 1))
        relu1_1 = mx.symbol.Activation(name='relu1_1', data=conv1_1, act_type='relu')
        conv1_2 = mx.symbol.Convolution(name='conv1_2', data=relu1_1, num_filter=64, pad=(1, 1),
                                        kernel=(3, 3), stride=(1, 1))
        relu1_2 = mx.symbol.Activation(name='relu1_2', data=conv1_2, act_type='relu')
        pool1 = mx.symbol.Pooling(name='pool1', data=relu1_2, pooling_convention='full', pad=(0, 0),
                                  kernel=(2, 2), stride=(2, 2), pool_type='max')

        conv2_1 = mx.symbol.Convolution(name='conv2_1', data=pool1, num_filter=128, pad=(1, 1),
                                        kernel=(3, 3), stride=(1, 1))
        relu2_1 = mx.symbol.Activation(name='relu2_1', data=conv2_1, act_type='relu')
        conv2_2 = mx.symbol.Convolution(name='conv2_2', data=relu2_1, num_filter=128, pad=(1, 1),
                                        kernel=(3, 3), stride=(1, 1))
        relu2_2 = mx.symbol.Activation(name='relu2_2', data=conv2_2, act_type='relu')
        pool2 = mx.symbol.Pooling(name='pool2', data=relu2_2, pooling_convention='full', pad=(0, 0),
                                  kernel=(2, 2), stride=(2, 2), pool_type='max')

        conv3_1 = mx.symbol.Convolution(name='conv3_1', data=pool2, num_filter=256, pad=(1, 1),
                                        kernel=(3, 3), stride=(1, 1))
        relu3_1 = mx.symbol.Activation(name='relu3_1', data=conv3_1, act_type='relu')
        conv3_2 = mx.symbol.Convolution(name='conv3_2', data=relu3_1, num_filter=256, pad=(1, 1),
                                        kernel=(3, 3), stride=(1, 1))
        relu3_2 = mx.symbol.Activation(name='relu3_2', data=conv3_2, act_type='relu')
        conv3_3 = mx.symbol.Convolution(name='conv3_3', data=relu3_2, num_filter=256, pad=(1, 1),
                                        kernel=(3, 3), stride=(1, 1))
        relu3_3 = mx.symbol.Activation(name='relu3_3', data=conv3_3, act_type='relu')
        pool3 = mx.symbol.Pooling(name='pool3', data=relu3_3, pooling_convention='full', pad=(0, 0),
                                  kernel=(2, 2), stride=(2, 2), pool_type='max')

        conv4_1 = mx.symbol.Convolution(name='conv4_1', data=pool3, num_filter=512, pad=(1, 1),
                                        kernel=(3, 3), stride=(1, 1))
        relu4_1 = mx.symbol.Activation(name='relu4_1', data=conv4_1, act_type='relu')
        conv4_2 = mx.symbol.Convolution(name='conv4_2', data=relu4_1, num_filter=512, pad=(1, 1),
                                        kernel=(3, 3), stride=(1, 1))
        relu4_2 = mx.symbol.Activation(name='relu4_2', data=conv4_2, act_type='relu')
        conv4_3 = mx.symbol.Convolution(name='conv4_3', data=relu4_2, num_filter=512, pad=(1, 1),
                                        kernel=(3, 3), stride=(1, 1))
        relu4_3 = mx.symbol.Activation(name='relu4_3', data=conv4_3, act_type='relu')
        pool4 = mx.symbol.Pooling(name='pool4', data=relu4_3, pooling_convention='full', pad=(0, 0),
                                  kernel=(2, 2), stride=(2, 2), pool_type='max')

        conv5_1 = mx.symbol.Convolution(name='conv5_1', data=pool4, num_filter=512, pad=(1, 1),
                                        kernel=(3, 3), stride=(1, 1))
        relu5_1 = mx.symbol.Activation(name='relu5_1', data=conv5_1, act_type='relu')
        conv5_2 = mx.symbol.Convolution(name='conv5_2', data=relu5_1, num_filter=512, pad=(1, 1),
                                        kernel=(3, 3), stride=(1, 1))
        relu5_2 = mx.symbol.Activation(name='relu5_2', data=conv5_2, act_type='relu')
        conv5_3 = mx.symbol.Convolution(name='conv5_3', data=relu5_2, num_filter=512, pad=(1, 1),
                                        kernel=(3, 3), stride=(1, 1))
        relu5_3 = mx.symbol.Activation(name='relu5_3', data=conv5_3, act_type='relu')

        return relu5_3

    @staticmethod
    def attention_module_multi_head(roi_feat, position_embedding,
                                    nongt_dim, fc_dim, feat_dim=1024,
                                    dim=(1024, 1024, 1024),
                                    group=16, index=1):
        """ Attetion module with vectorized version

        Args:
            roi_feat: [num_img, num_rois, feat_dim]
            position_embedding: [num_img, num_rois, nongt_dim, emb_dim]
            num_rois: number of rois per image, for reshape
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:

        Returns:
            output: [num_img, num_rois, ovr_feat_dim, output_dim]
        """
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)
        nongt_roi_feat = mx.symbol.slice_axis(data=roi_feat, axis=1, begin=0, end=nongt_dim)

        # position_feat_1, [num_img, num_rois, nongt_dim, fc_dim]
        position_feat_1 = mx.sym.FullyConnected(name='pair_pos_fc1_' + str(index),
                                                data=position_embedding,
                                                num_hidden=fc_dim,
                                                flatten=False)
        position_feat_1_relu = mx.sym.Activation(data=position_feat_1, act_type='relu')
        # aff_weight, [num_img, num_rois, fc_dim, nongt_dim]
        aff_weight = mx.sym.transpose(position_feat_1_relu, axes=(0, 1, 3, 2))

        # multi head
        assert dim[0] == dim[1], 'Matrix multiply requires same dimensions!'
        # q_data, [num_img, num_rois, dim[0]]
        q_data = mx.sym.FullyConnected(name='query_' + str(index),
                                       data=roi_feat,
                                       num_hidden=dim[0],
                                       flatten=False)
        # q_data, [num_img, num_rois, group, dim_group[0]]
        q_data_batch = mx.sym.Reshape(q_data, shape=(0, 0, group, dim_group[0]))
        q_data_batch = mx.sym.transpose(q_data_batch, axes=(0, 2, 1, 3))
        # q_data_batch, [num_img*group, num_rois, dim_group[0]]
        q_data_batch = mx.sym.Reshape(q_data_batch, shape=(-3, -2))

        # k_data, [num_img, nongt_dim, dim[1]]
        k_data = mx.symbol.FullyConnected(name='key_' + str(index),
                                          data=nongt_roi_feat,
                                          num_hidden=dim[1],
                                          flatten=False)
        # k_data, [num_img, nongt_dim, group, dim_group[1]]
        k_data_batch = mx.sym.Reshape(k_data, shape=(0, 0, group, dim_group[1]))
        k_data_batch = mx.sym.transpose(k_data_batch, axes=(0, 2, 1, 3))
        # k_data_batch, [num_img*group, nongt_dim, dim_group[1]]
        k_data_batch = mx.sym.Reshape(k_data_batch, shape=(-3, -2))

        # v_data, [num_img, nongt_dim, feat_dim]
        v_data = nongt_roi_feat
        # v_data =  mx.symbol.FullyConnected(name='value_'+str(index)+'_'+str(gid), data=roi_feat, num_hidden=dim_group[2])
        aff = mx.symbol.batch_dot(lhs=q_data_batch, rhs=k_data_batch, transpose_a=False, transpose_b=True)
        # aff_scale, [num_img*group, num_rois, nongt_dim]
        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        aff_scale = mx.sym.Reshape(data=aff_scale, shape=(-4, -1, group, -2))
        # aff_scale, [num_img, num_rois, group, nongt_dim]
        aff_scale = mx.sym.transpose(aff_scale, axes=(0, 2, 1, 3))

        assert fc_dim == group, 'fc_dim != group'
        # weighted_aff, [num_img, num_rois, fc_dim, nongt_dim]
        weighted_aff = mx.sym.log(mx.sym.maximum(left=aff_weight, right=1e-6)) + aff_scale
        aff_softmax = mx.symbol.softmax(data=weighted_aff, axis=3, name='softmax_' + str(index))
        # [num_img, num_rois * fc_dim, nongt_dim]
        aff_softmax_reshape = mx.sym.Reshape(aff_softmax, shape=(0, -3, nongt_dim))
        # output_t, [num_img, num_rois * fc_dim, feat_dim]
        output_t = mx.symbol.batch_dot(lhs=aff_softmax_reshape, rhs=v_data,
                                       transpose_a=False, transpose_b=False)
        # linear_out, [num_img, num_rois * fc_dim, dim_group[2]]
        linear_out = mx.sym.FullyConnected(data=output_t,
                                           name='linear_out_' + str(index),
                                           num_hidden=dim_group[2],
                                           flatten=False)
        # output, [num_img, num_rois, fc_dim, dim_group[2]]
        output = mx.sym.Reshape(linear_out, shape=(0, -4, -1, fc_dim, dim_group[2]))
        # output, [num_img, num_rois, dim[2](fc_dim*dim_group[2])]
        output = mx.sym.Reshape(output, shape=(0, 0, feat_dim))
        return output

    def get_symbol_vgg16_attention(self, num_class, batch_images, num_rois, nongt_dim=-1):

        nongt_dim = nongt_dim if nongt_dim > 0 else num_rois

        data = mx.sym.Variable(name="data")
        rois = mx.sym.Variable(name="rois")  # [num_img * roi_per_img, 5], for roi pooling

        relu5_3 = self.get_vgg16_conv5(data)
        conv_new_1 = mx.sym.Convolution(data=relu5_3, kernel=(1, 1), num_filter=256, name="conv_new_1")
        conv_new_1_relu = mx.sym.Activation(data=conv_new_1, act_type='relu', name='conv_new_1_relu')

        roi_pool = mx.symbol.ROIPooling(
            name='roi_pool', data=conv_new_1_relu, rois=rois, pooled_size=(7, 7), spatial_scale=0.0625)

        # from utils.pos_embedding import extract_position_embedding, extract_position_matrix
        # sliced_rois = mx.sym.slice_axis(rois, axis=2, begin=1, end=None)
        # position_matrix = extract_position_matrix(sliced_rois)
        # # [num_rois, nongt_dim, 64]
        # position_embedding = extract_position_embedding(position_matrix, feat_dim=64)
        import utils.pos_embedding
        position_embedding = mx.sym.Custom(rois=rois,
                                           op_type='position_embedding_py',
                                           feat_dim=64,
                                           batch_images=batch_images,
                                           name='pos_embedding')
        # 2 fc
        fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1', data=roi_pool, num_hidden=1024)
        fc_new_1_reshape = mx.symbol.Reshape(data=fc_new_1, name='fc_new_1_reshape', shape=(-4, batch_images, -1, 0))
        # attention, [num_img, num_rois, feat_dim]
        attention_1 = self.attention_module_multi_head(fc_new_1_reshape, position_embedding,
                                                       nongt_dim=nongt_dim, fc_dim=16, feat_dim=1024,
                                                       index=1, group=16,
                                                       dim=(1024, 1024, 1024))
        fc_all_1 = fc_new_1_reshape + attention_1
        fc_all_1_relu = mx.sym.Activation(data=fc_all_1, act_type='relu', name='fc_all_1_relu')

        fc_new_2 = mx.symbol.FullyConnected(name='fc_new_2', data=fc_all_1_relu, num_hidden=1024, flatten=False)
        attention_2 = self.attention_module_multi_head(fc_new_2, position_embedding,
                                                       nongt_dim=nongt_dim, fc_dim=16, feat_dim=1024,
                                                       index=2, group=16,
                                                       dim=(1024, 1024, 1024))
        fc_all_2 = fc_new_2 + attention_2
        fc_all_2_relu = mx.sym.Activation(data=fc_all_2, act_type='relu', name='fc_all_2_relu')

        fc8a = mx.symbol.FullyConnected(name='fc_cls', data=fc_all_2_relu, num_hidden=num_class)
        prob = mx.symbol.SoftmaxOutput(name='prob', data=fc8a)

        return prob

    def get_symbol_vgg16(self):
        data = mx.symbol.Variable(name='data')

        relu5_3 = self.get_vgg16_conv5(data)
        pool5 = mx.symbol.Pooling(name='pool5', data=relu5_3, pooling_convention='full', pad=(0, 0),
                                  kernel=(2, 2), stride=(2, 2), pool_type='max')
        flatten_0 = mx.symbol.Flatten(name='flatten_0', data=pool5)

        fc6 = mx.symbol.FullyConnected(name='fc6', data=flatten_0, num_hidden=4096)
        relu6 = mx.symbol.Activation(name='relu6', data=fc6, act_type='relu')
        drop6 = mx.symbol.Dropout(name='drop6', data=relu6, p=0.500000)

        fc7 = mx.symbol.FullyConnected(name='fc7', data=drop6, num_hidden=4096)
        relu7 = mx.symbol.Activation(name='relu7', data=fc7, act_type='relu')
        drop7 = mx.symbol.Dropout(name='drop7', data=relu7, p=0.500000)

        fc8a = mx.symbol.FullyConnected(name='fc8a', data=drop7, num_hidden=365)
        prob = mx.symbol.SoftmaxOutput(name='prob', data=fc8a)

        return prob


if __name__ == '__main__':
    # sym = vgg16().get_symbol_vgg16()
    # mx.viz.print_summary(sym, {'data': (1, 3, 224, 224)})
    # a = mx.viz.plot_network(symbol=sym, shape={"data": (1, 3, 224, 224)}, title='vgg16')
    # a.render()
    sym = vgg16().get_symbol_vgg16_attention(num_class=397, batch_images=2, num_rois=30)
    mx.viz.print_summary(sym, {'data': (2, 3, 224, 224), 'rois': (60, 5)})
    a = mx.viz.plot_network(symbol=sym, shape={"data": (2, 3, 224, 224), 'rois': (60, 5)}, title='vgg16_att')
    a.render()
