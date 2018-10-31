"""position embedding"""

import mxnet as mx
import numpy as np


def extract_position_embedding(position_mat, feat_dim, wave_length=1000):
    # position_mat, [num_rois, nongt_dim, 4]
    feat_range = mx.sym.arange(0, feat_dim / 8)
    dim_mat = mx.sym.broadcast_power(lhs=mx.sym.full((1,), wave_length),
                                     rhs=(8. / feat_dim) * feat_range)
    dim_mat = mx.sym.Reshape(dim_mat, shape=(1, 1, 1, -1))
    position_mat = mx.sym.expand_dims(100.0 * position_mat, axis=3)
    div_mat = mx.sym.broadcast_div(lhs=position_mat, rhs=dim_mat)
    sin_mat = mx.sym.sin(data=div_mat)
    cos_mat = mx.sym.cos(data=div_mat)
    # embedding, [num_rois, nongt_dim, 4, feat_dim/4]
    embedding = mx.sym.concat(sin_mat, cos_mat, dim=3)
    # embedding, [num_rois, nongt_dim, feat_dim]
    embedding = mx.sym.Reshape(embedding, shape=(0, 0, feat_dim))
    return embedding


def extract_position_matrix(bbox, nongt_dim=-1):
    """ Extract position matrix

    Args:
        bbox: [num_boxes, 4]

    Returns:
        position_matrix: [num_boxes, nongt_dim, 4]
    """
    xmin, ymin, xmax, ymax = mx.sym.split(data=bbox,
                                          num_outputs=4, axis=1)
    # [num_fg_classes, num_boxes, 1]
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    # [num_fg_classes, num_boxes, num_boxes]
    delta_x = mx.sym.broadcast_minus(lhs=center_x,
                                     rhs=mx.sym.transpose(center_x))
    delta_x = mx.sym.broadcast_div(delta_x, bbox_width)
    delta_x = mx.sym.log(mx.sym.maximum(mx.sym.abs(delta_x), 1e-3))
    delta_y = mx.sym.broadcast_minus(lhs=center_y,
                                     rhs=mx.sym.transpose(center_y))
    delta_y = mx.sym.broadcast_div(delta_y, bbox_height)
    delta_y = mx.sym.log(mx.sym.maximum(mx.sym.abs(delta_y), 1e-3))
    delta_width = mx.sym.broadcast_div(lhs=bbox_width,
                                       rhs=mx.sym.transpose(bbox_width))
    delta_width = mx.sym.log(delta_width)
    delta_height = mx.sym.broadcast_div(lhs=bbox_height,
                                        rhs=mx.sym.transpose(bbox_height))
    delta_height = mx.sym.log(delta_height)
    concat_list = [delta_x, delta_y, delta_width, delta_height]
    for idx, sym in enumerate(concat_list):
        sym = mx.sym.slice_axis(sym, axis=1, begin=0, end=nongt_dim) if nongt_dim > 0 else sym
        concat_list[idx] = mx.sym.expand_dims(sym, axis=2)
    position_matrix = mx.sym.concat(*concat_list, dim=2)
    return position_matrix


class PositionEmbeddingPyOperator(mx.operator.CustomOp):
    def __init__(self, feat_dim, batch_images, nongt_dim=-1, wave_length=1000):
        super(PositionEmbeddingPyOperator, self).__init__()
        self._feat_dim = feat_dim
        self._batch_images = batch_images
        self._nongt_dim = nongt_dim
        self._wave_length = wave_length
        self.eps = 1e-3

    def forward(self, is_train, req, in_data, out_data, aux):
        all_rois = in_data[0].asnumpy()   # [num_img * roi_per_img, 5]
        assert all_rois.shape[0] % self._batch_images == 0, \
            'rois number should be divided by num_img, check the rois loading'

        # [num_img, roi_per_img, roi_per_img, fea_dim]
        position_encoding = self.pos_encoding(all_rois, fea_dim=self._feat_dim, nongt_dim=self._nongt_dim)
        self.assign(out_data[0], req[0], mx.nd.array(position_encoding))

    def pos_encoding(self, rois_ori, fea_dim, nongt_dim=-1):
        """
        calculate the relative-position encoding for each location pair
        note: ignore the _feat_stride because of the zero order transformation
        :param rois: rois with [bbox_num, 5], (im_inds, x1, y1, x2, y2)
        :param fea_dim: the dimension of the output of each location pair
        :return: encoding of the relative-position, [im_num, bbox_num, bbox_num, fea_dim]
        """
        # change form to ctr_x, ctr_y, widths, heights
        rois = rois_ori.copy()
        rois[:, 3] = rois[:, 3] - rois[:, 1] + 1
        rois[:, 4] = rois[:, 4] - rois[:, 2] + 1
        rois[:, 1] = rois[:, 1] + 0.5 * rois[:, 3]
        rois[:, 2] = rois[:, 2] + 0.5 * rois[:, 4]

        rois_num_total = rois.shape[0]
        im_num = np.alen(np.unique(rois[:, 0]))
        rois_num = rois_num_total // im_num

        assert fea_dim % 8 == 0, 'fea_dim must be divided by 8'

        pos_encode_list = []

        for im_i in np.arange(im_num):
            rois_im_i = rois[im_i * rois_num:(im_i + 1) * rois_num, 1:]  # [rois_num, 4]
            pos_encode_tmp = np.zeros(shape=[rois_num, rois_num, 4], dtype=np.float32)

            pos_encode_tmp[:, :, 0] = np.divide(rois_im_i[:, 0][:, np.newaxis] - rois_im_i[:, 0][np.newaxis],
                                                rois_im_i[:, 2][:, np.newaxis])
            pos_encode_tmp[:, :, 0] = np.maximum(np.abs(pos_encode_tmp[:, :, 0]), self.eps)
            pos_encode_tmp[:, :, 1] = np.divide(rois_im_i[:, 1][:, np.newaxis] - rois_im_i[:, 1][np.newaxis],
                                                rois_im_i[:, 3][:, np.newaxis])
            pos_encode_tmp[:, :, 1] = np.maximum(np.abs(pos_encode_tmp[:, :, 1]), self.eps)
            pos_encode_tmp[:, :, 2] = np.divide(rois_im_i[:, 2][np.newaxis], rois_im_i[:, 2][:, np.newaxis])
            pos_encode_tmp[:, :, 3] = np.divide(rois_im_i[:, 3][np.newaxis], rois_im_i[:, 3][:, np.newaxis])

            pos_encode_tmp = pos_encode_tmp if nongt_dim < 0 else pos_encode_tmp[:, :nongt_dim]
            pos_encode_tmp = np.log(pos_encode_tmp)
            # print(pos_encode_tmp[:2, :2, :])

            pos_encode_tmp = self._pos_encode(pos_encode_tmp, fea_dim)
            # pos_encode_tmp = np.reshape(pos_encode_tmp, [rois_num, rois_num, -1])

            pos_encode_list.append(pos_encode_tmp[np.newaxis])

        pos_encode_total = np.concatenate(pos_encode_list)

        return np.float32(pos_encode_total)

    @staticmethod
    def _pos_encode(position_mat, feat_dim, wave_length=1000):
        """
        return sin and cos encodeing of the position code, refer to code in mxnet
        """
        # position_mat, [num_rois, nongt_dim, 4]
        rois_num = position_mat.shape[0]
        nongt_dim = position_mat.shape[1]
        feat_range = np.arange(0, feat_dim / 8)
        dim_mat = np.power(np.full((1,), wave_length),
                           (8. / feat_dim) * feat_range)
        dim_mat = np.reshape(dim_mat, [1, 1, 1, -1])
        position_mat = np.expand_dims(100.0 * position_mat, axis=3)
        div_mat = np.divide(position_mat, dim_mat)
        sin_mat = np.sin(div_mat)
        cos_mat = np.cos(div_mat)
        # embedding, [num_rois, nongt_dim, 4, feat_dim/4]
        embedding = np.concatenate([sin_mat, cos_mat], axis=3)
        # embedding, [num_rois, nongt_dim, feat_dim]
        embedding = np.reshape(embedding, [rois_num, nongt_dim, feat_dim])

        return embedding

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)


@mx.operator.register('position_embedding_py')
class PositionEmbeddingPyProp(mx.operator.CustomOpProp):
    def __init__(self, feat_dim, batch_images, nongt_dim=-1, wave_length=1000):
        super(PositionEmbeddingPyProp, self).__init__(need_top_grad=False)
        self._feat_dim = int(feat_dim)
        self._batch_images = int(batch_images)
        self._nongt_dim = int(nongt_dim)
        self._wave_length = int(wave_length)
        self.eps = 1e-3

    def list_arguments(self):
        return ['rois']

    def list_outputs(self):
        return ['pos_encoding']

    def infer_shape(self, in_shape):
        rois_shape = in_shape[0]
        # print rois_shape[0], self._batch_images
        # print type(rois_shape[0]), type(self._batch_images)
        rois_per_img = rois_shape[0] // self._batch_images
        pos_embedding = (self._batch_images, rois_per_img, rois_per_img, self._feat_dim)

        return [rois_shape], [pos_embedding]

    def create_operator(self, ctx, shapes, dtypes):
        return PositionEmbeddingPyOperator(self._feat_dim, self._batch_images, self._nongt_dim, self._wave_length)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
