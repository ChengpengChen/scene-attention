"""
create data loader
by ccp, on 2018.08.18
"""

from __future__ import print_function, division, absolute_import
import os.path as osp
import cv2
import glob
import random

import mxnet as mx
import numpy as np

from threading import Thread
from collections import defaultdict

import os
try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from mxnet.io import DataBatch, DataIter

from utils.random_erasing import random_erasing
from utils.bbox_utils import _clip_boxes, _vis_detections, _sample_rois


def pop(x, size):
    return [x.pop(0) for _ in range(size)]


# to generate the mask dir
image_root_sun397 = '/media/chencp/data_ssd2/datasets/SUN397'
image_root_mit67 = '/media/chencp/data_ssd2/datasets/indoor67'


class ROIdatalayer(DataIter):
    def __init__(self, data_dir, img_list_file, roi_file, batch_size, image_size=224, rand_crop=True, shuffle=False,
                 rand_flip=False, random_erasing=False, force_resize=0, resize_shorter=0, roi_aug_dict=dict(),
                 num_worker=4, random_seed=None):
        """
        provide image and roi data
        :param dataset: mit67 or sun397
        :param img_list_file: TestImages.label or TrainImages.label
        :param roi_file: pickle file containing roi data
        :param flip: flip the image
        :param random_erasing: erase image
        :param roi_num: number of rois for training
        :param nms: to process rois in an image
        :param enlarge_factor: enlarge the roi to include more context
        :param num_worker:
        :param random_seed:
        """
        # if dataset == 'mit67':
        #     self.img_list_file = os.path.join(image_root_mit67, 'list', img_list_file)
        #     self.roi_file = os.path.join(image_root_mit67, 'roi_pkl', roi_file)
        #     self.image_root = os.path.join(image_root_mit67, 'Images')
        # elif dataset == 'sun397':
        #     self.img_list_file = os.path.join(image_root_sun397, 'list', img_list_file)
        #     self.roi_file = os.path.join(image_root_sun397, 'roi_pkl', roi_file)
        #     self.image_root = os.path.join(image_root_sun397, 'image')
        # else:
        #     print('only mit67 or sun397 support yet')
        #     raise NotImplementedError
        self.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
        self.img_list_file = osp.join(self.ROOT_DIR, img_list_file)
        self.roi_file = osp.join(self.ROOT_DIR, roi_file)
        assert os.path.exists(self.img_list_file), 'image list not exist: %s' % self.img_list_file
        assert os.path.exists(self.roi_file), 'roi file not exist: %s' % self.roi_file

        if random_seed is None:
            random_seed = random.randint(0, 2 ** 32 - 1)
        np.random.RandomState(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.random_seed = random_seed

        self.data_dir = osp.join(self.ROOT_DIR, data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.rand_flip = rand_flip
        self.rand_crop = rand_crop
        self.random_erasing = random_erasing
        self.force_resize = force_resize
        self.resize_shorter = resize_shorter
        self.shuffle = shuffle

        if 'roi_num' not in roi_aug_dict:
            roi_aug_dict['roi_num'] = 20
        if 'nms_thresh' not in roi_aug_dict:
            roi_aug_dict['nms_thresh'] = 0.7
        if 'enlarge_factor' not in roi_aug_dict:
            roi_aug_dict['enlarge_factor'] = 0
        if 'drop_small' not in roi_aug_dict:
            roi_aug_dict['drop_small_scale'] = 16
        self.roi_aug_dict = roi_aug_dict
        self.cursor = 0
        self.size = 0

        resize_set = {self.resize_shorter, self.force_resize}
        assert len(resize_set) == 2 and 0 in resize_set, "resize_shorter and force_resize are mutually exclusive!"

        if self.force_resize > 0:
            if self.force_resize < self.image_size:
                raise ValueError("each edge of force_resize must be larger than or equal to target image size")

        if self.resize_shorter > 0:
            if self.resize_shorter < self.image_size:
                raise ValueError("resize_shorter must be larger than or equal to minimum of target image side size")

        print("Data pre-processing..")
        # roi_dict: {'image_name': (x1, x2, y1, y2, score)}
        self.img_list, self.roi_dict, self.roi_dict_pro = self._preprocess(img_list_file, roi_file, self.roi_aug_dict)
        print("loaded!")
        self.data_num = len(self.img_list)

        # multi-thread primitive
        self.result_queue = Queue(maxsize=8 * num_worker)
        self.index_queue = Queue()
        self.workers = None

        self._thread_start(num_worker)

        self.reset()

    @staticmethod
    def _preprocess(img_list_file, roi_file, roi_aug_dict):
        with open(img_list_file) as fin:
            img_list = []
            for line in iter(fin.readline, ''):
                line = line.strip().split(' ')
                # label = mx.nd.array([float(i) for i in line[1:]])
                label = int(line[1])
                img_name = line[0]
                img_list.append((img_name, label))

        with open(roi_file, 'rb') as fid:
            roi_dict = pickle.load(fid)
        roi_dict_pro = _sample_rois(roi_dict, roi_aug_dict)
        #
        # roi_dict_pro = dict()
        # for img, rois in roi_dict.items():
        #     # todo: sample rois, e.g. applying nms and random sample
        #     # drop score for each roi
        #     ind = np.random.randint(0, len(rois), roi_num)
        #     rois_new = rois[ind, :4]
        #     roi_dict_pro[img] = rois_new
        #     print(rois_new[0])

        return img_list, roi_dict, roi_dict_pro

    def reset(self):
        self.cursor = 0
        self.index_queue.queue.clear()

        # insert queue
        img_list = self.img_list[:]
        if self.shuffle:
            random.shuffle(img_list)

        sample_range = list(range(0, len(img_list), self.batch_size))
        if len(img_list) % self.batch_size != 0:
            sample_range = sample_range[:-1]
        self.size = len(sample_range)
        data = []
        labels = []
        for i in sample_range:
            start = i
            stop = start + self.batch_size
            for img, label in img_list[start:stop]:
                data.append(img if img[0] != '/' else img[1:])
                labels.append(label)

            self.index_queue.put([data[:], labels[:]])

            assert len(data) == self.batch_size
            assert len(labels) == self.batch_size

            del data[:]
            del labels[:]

        self.roi_dict_pro = _sample_rois(self.roi_dict, self.roi_aug_dict)

    def _thread_start(self, num_worker):
        self.workers = [Thread(target=self._worker, args=(self.random_seed + i,)) for i in range(num_worker)]

        for worker in self.workers:
            worker.daemon = True
            worker.start()

    def _worker(self, seed):
        np.random.RandomState(seed)
        np.random.seed(seed)
        random.seed(seed)

        while True:
            indices = self.index_queue.get()
            result = self._get_batch(indices=indices)

            if result is None:
                return

            self.result_queue.put(result)

    def _get_batch(self, indices):
        img_names = indices[0]
        labels = indices[1]

        data = []
        rois = []
        im_index = 0
        for img_name in img_names:
            # load rois
            roi = self.roi_dict_pro[img_name[:-4]]
            assert roi.shape[0] == self.roi_aug_dict['roi_num'], 'rois number error'
            # print(rois)

            # load images
            img_path = os.path.join(self.data_dir, img_name)
            img = cv2.imread(img_path)
            cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB, dst=img)
            img = img.astype(np.float32)
            ori_h, ori_w = img.shape[:2]
            # _vis_detections(img, roi, 'iter_ori_img')

            if self.force_resize > 0:
                img = cv2.resize(img, (self.force_resize, self.force_resize), interpolation=cv2.INTER_LINEAR)
                scales = (np.float(self.force_resize) / ori_h, np.float(self.force_resize) / ori_w)
                roi = roi * np.array([[scales[1], scales[0], scales[1], scales[0]]])
                # _vis_detections(img, roi, 'iter_resize_img')
            else:
                if ori_h < ori_w:
                    h, w = self.resize_shorter, int(round(self.resize_shorter / ori_h * ori_w))
                else:
                    h, w = int(round(self.resize_shorter / ori_w * ori_h)), self.resize_shorter

                if h != ori_h or w != ori_w:
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
                    scales = (np.float(h) / ori_h, np.float(w) / ori_w)
                    roi = roi * np.array([[scales[1], scales[0], scales[1], scales[0]]])

            h, w = img.shape[:2]
            x, y = int(round((w - self.image_size) / 2)), int(round((h - self.image_size) / 2))

            # random crop
            if self.rand_crop:
                x = random.randint(0, w - self.image_size)
                y = random.randint(0, h - self.image_size)

            if (self.image_size, self.image_size) != (h, w):
                img = img[y:y + self.image_size, x:x + self.image_size, :]
                roi -= np.array([[x, y, x, y]])
                roi = _clip_boxes(roi, (self.image_size, self.image_size))
                # _vis_detections(img, roi, 'iter_crop_img')

            # random mirror
            if self.rand_flip and random.randint(0, 1) == 1:
                img = cv2.flip(img, flipCode=1)
                roi[:, 0], roi[:, 2] = self.image_size - roi[:, 2] - 1, self.image_size - roi[:, 0] - 1
                # _vis_detections(img, roi, 'iter_flip_img')

            if self.random_erasing:
                img = random_erasing(img)

            roi = _clip_boxes(roi, (self.image_size, self.image_size))
            roi = np.hstack((im_index*np.ones(shape=(roi.shape[0], 1)), roi))
            im_index += 1

            rois.append(roi)
            data.append(img)

        data = mx.nd.array(np.stack(data, axis=0).transpose([0, 3, 1, 2]))
        rois = mx.nd.array(np.stack(rois, axis=0))
        labels = mx.nd.array(labels)
        assert data.shape[0] == labels.shape[0] == rois.shape[0] == self.batch_size

        data_all = [mx.nd.array(data), mx.nd.array(rois)]
        label_all = [mx.nd.array(labels)]

        return data_all, label_all

    @property
    def provide_data(self):
        data = [('data', (self.batch_size, 3, self.image_size, self.image_size)),
                ('rois', (self.batch_size, self.roi_aug_dict['roi_num'], 4))]
        return data

    @property
    def provide_label(self):
        labels = [('softmax_label', (self.batch_size,))]
        return labels

    def next(self):
        if self.cursor >= self.size:
            raise StopIteration

        data, label = self.result_queue.get()

        self.cursor += 1

        return DataBatch(data=data, label=label, provide_data=self.provide_data, provide_label=self.provide_label)


if __name__ == '__main__':
    import time

    data_dir = '/media/chencp/data_ssd2/datasets/indoor67/Images'
    img_list = '/media/chencp/data_ssd2/datasets/indoor67/list/temp.label'
    roi_file = '/media/chencp/data_ssd2/datasets/indoor67/roi_pkl/mit67_bbox_test.pkl'
    batch_size = 2
    roi_aug_dict = dict()
    roi_aug_dict['roi_num'] = 10
    roi_aug_dict['nms'] = 0.3
    roi_aug_dict['enlarge_factor'] = 0.2
    roi_aug_dict['drop_small_scale'] = 16

    iterator = ROIdatalayer(data_dir=data_dir, img_list_file=img_list, roi_file=roi_file, batch_size=batch_size,
                            image_size=256, rand_crop=True, shuffle=False, rand_flip=True, random_erasing=False,
                            force_resize=256, resize_shorter=0, roi_aug_dict=roi_aug_dict,
                            num_worker=1, random_seed=None)

    tic = time.time()
    tmp = []
    for i in range(2):
        print(i)
        batch = iterator.next()
        tmp.append(batch.label[0].asnumpy().tolist())
        print(batch.data[0].shape)
        print(batch.data[1].shape)
        print(batch.data[1])
        print(batch.label[0].shape)
        print(batch.label[0])
        imgs = batch.data[0].transpose([0, 2, 3, 1]).asnumpy()

        print(imgs.dtype)
        for j in range(imgs.shape[0]):
            img = imgs[j]
            cv2.imwrite("../save_img/%d-%d.jpg" % (i, j), img[:, :, [2, 1, 0]].astype(np.uint8))
            # import pdb
            # pdb.set_trace()
    iterator.reset()
    for i in range(2):
        print(i)
        batch = iterator.next()
        tmp.append(batch.label[0].asnumpy().tolist())
        print(batch.data[0].shape)
        print(batch.data[1].shape)
        print(batch.data[1])
        print(batch.label[0].shape)
        print(batch.label[0])
        imgs = batch.data[0].transpose([0, 2, 3, 1]).asnumpy()

        print(imgs.dtype)
        for j in range(imgs.shape[0]):
            img = imgs[j]
            cv2.imwrite("../save_img/%d-%d.jpg" % (i, j), img[:, :, [2, 1, 0]].astype(np.uint8))
            # import pdb
            # pdb.set_trace()
    print((time.time() - tic) / 5)
