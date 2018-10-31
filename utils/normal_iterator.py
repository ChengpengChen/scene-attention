"""
get a iter to get rgb data
refer to roi_iterator

by ccp, on 2018.08.25
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


def pop(x, size):
    return [x.pop(0) for _ in range(size)]


mean_torch = np.reshape(np.array([0.485, 0.456, 0.406]), [1, 1, 3])
std_torch = np.reshape(np.array([0.229, 0.224, 0.225]), [1, 1, 3])


class NormalIterator(DataIter):
    def __init__(self, data_dir, img_list_file, batch_size, image_size=224, resize_shorter=0, shuffle=True,
                 rand_mirror=False, rand_crop=False, random_erasing=False, force_resize=0,
                 num_worker=4, data_norm=False, random_seed=None):
        self.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
        self.img_list_file = osp.join(self.ROOT_DIR, img_list_file)
        assert os.path.exists(self.img_list_file), 'image list not exist: %s' % self.img_list_file
        if random_seed is None:
            random_seed = random.randint(0, 2 ** 32 - 1)
        np.random.RandomState(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.random_seed = random_seed

        self.data_dir = osp.join(self.ROOT_DIR, data_dir)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.image_size = image_size
        self.rand_mirror = rand_mirror
        self.rand_crop = rand_crop
        self.random_erasing = random_erasing
        self.resize_shorter = resize_shorter
        self.num_worker = num_worker
        self.data_norm = data_norm
        self.force_resize = force_resize
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

        print("Data loading..")
        self.img_list = self._preprocess(self.img_list_file)
        print("Data loaded!")

        # self.num_id = len(self.id2imgs)
        # print(self.num_id)

        # multi-thread primitive
        self.result_queue = Queue(maxsize=8 * num_worker)
        self.index_queue = Queue()
        self.workers = None

        self._thread_start(num_worker)

        self.reset()

    @staticmethod
    def _preprocess(img_list):
        with open(img_list) as fin:
            img_list = []
            for line in iter(fin.readline, ''):
                line = line.strip().split(' ')
                # label = mx.nd.array([float(i) for i in line[1:]])
                label = int(line[1])
                img_name = line[0]
                img_list.append((img_name, label))

        return img_list

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

        # Loading
        data = []
        for img_name in img_names:
            img_path = os.path.join(self.data_dir, img_name)
            # print('img_path: {}'.format(img_path))
            # print('file exist: {}'.format('Yes' if osp.exists(img_path) else 'No'))
            img = cv2.imread(img_path)
            cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB, dst=img)
            img = img.astype(np.float32)
            ori_h, ori_w = img.shape[:2]
            # import pdb
            # pdb.set_trace()

            if self.force_resize > 0:
                img = cv2.resize(img, (self.force_resize, self.force_resize), interpolation=cv2.INTER_LINEAR)
            else:
                if ori_h < ori_w:
                    h, w = self.resize_shorter, int(round(self.resize_shorter / ori_h * ori_w))
                else:
                    h, w = int(round(self.resize_shorter / ori_w * ori_h)), self.resize_shorter

                if h != ori_h or w != ori_w:
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

            h, w = img.shape[:2]
            x, y = int(round((w - self.image_size) / 2)), int(round((h - self.image_size) / 2))

            # random crop
            if self.rand_crop:
                x = random.randint(0, w - self.image_size)
                y = random.randint(0, h - self.image_size)

            if (self.image_size, self.image_size) != (h, w):
                img = img[y:y + self.image_size, x:x + self.image_size, :]

            # random mirror
            if self.rand_mirror and random.randint(0, 1) == 1:
                img = cv2.flip(img, flipCode=1)

            if self.random_erasing:
                img = random_erasing(img)

            # data norm with mean and std from pytorch
            if self.data_norm:
                img /= 255
                img -= mean_torch
                img /= std_torch

            data.append(img)

        data = np.stack(data, axis=0).transpose([0, 3, 1, 2])
        data = mx.nd.array(data)
        labels = mx.nd.array(labels)

        assert data.shape[0] == labels.shape[0] == self.batch_size

        return [data], [labels]

    @property
    def provide_data(self):
        return [('data', (self.batch_size, 3, self.image_size, self.image_size))]

    @property
    def provide_label(self):
        return [('softmax_label', (self.batch_size,))]

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

    def next(self):
        if self.cursor >= self.size:
            raise StopIteration

        data, label = self.result_queue.get()

        self.cursor += 1

        return DataBatch(data=data, label=label, provide_data=self.provide_data, provide_label=self.provide_label)


if __name__ == '__main__':
    import time

    data_dir = 'dataset/SUN397/Images'
    img_list = 'dataset/SUN397/list/TrainImages.label'
    train = NormalIterator(data_dir=data_dir, img_list_file=img_list, batch_size=50, image_size=224,
                           rand_mirror=True, rand_crop=True, random_erasing=True, shuffle=False,
                           resize_shorter=0, force_resize=256, num_worker=1)
    tic = time.time()
    tmp = []
    for i in range(397):
        print(i)
        batch = train.next()
        # print(batch.data[0].shape)
        # print(batch.label[0].shape)
        # imgs = batch.data[0].transpose([0, 2, 3, 1]).asnumpy()
        #
        # print(imgs.dtype)
        # for j in range(imgs.shape[0]):
        #     img = imgs[j]
        #     cv2.imwrite("save_img/%d-%d.jpg" % (i, j), img[:, :, [2, 1, 0]].astype(np.uint8))
    print((time.time() - tic) / 397)
