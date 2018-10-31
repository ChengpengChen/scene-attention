"""
use model.score to eval the model with validation data

"""
import mxnet as mx
import numpy as np
from utils.normal_iterator import NormalIterator


def get_iterators(data_dir, img_list, image_size, batch_size, aug_dict, data_norm):
    rand_mirror = aug_dict.get("rand_mirror", False)
    rand_crop = aug_dict.get("rand_crop", False)
    random_erasing = aug_dict.get("random_erasing", False)
    resize_shorter = aug_dict.get("resize_shorter", 0)
    force_resize = aug_dict.get("force_resize", 0)
    #
    # train = NormalIterator(data_dir=data_dir, img_list_file=img_list[0], batch_size=batch_size, image_size=image_size,
    #                        rand_mirror=rand_mirror, rand_crop=rand_crop, random_erasing=random_erasing, shuffle=True,
    #                        resize_shorter=resize_shorter, force_resize=force_resize,
    #                        data_norm=data_norm, random_seed=seed)

    val = NormalIterator(data_dir=data_dir, img_list_file=img_list, batch_size=batch_size, image_size=image_size,
                         rand_mirror=False, rand_crop=False, random_erasing=False, shuffle=False,
                         resize_shorter=resize_shorter, force_resize=force_resize,
                         data_norm=data_norm, random_seed=None)

    return val


if __name__ == '__main__':
    gpu_id = 0
    context = mx.gpu(gpu_id)
    dataset = 'sun397'

    # iterator
    if dataset == 'mit67':
        data_dir = 'dataset/MIT67/Images'
        img_list = 'dataset/MIT67/list/TestImages.label'
        batch_size = 20
    else:
        data_dir = 'dataset/SUN397/Images'
        img_list = 'dataset/SUN397/list/TestImages.label'
        batch_size = 1

    image_size = 224
    aug_dict = dict()
    aug_dict['force_resize'] = 256
    data_norm = False
    data_shapes = [('data', (batch_size, 3, image_size, image_size))]

    val_iter = get_iterators(data_dir, img_list, image_size, batch_size, aug_dict, data_norm)

    # module
    prefix = 'baseline_resnet50_sun397_bs16_drop0.5_0903'
    epoch_idx = 50

    load_model_prefix = "models/%s" % prefix if dataset is None \
        else "models/%s" % (prefix)
    symbol, arg_params, aux_params = mx.model.load_checkpoint(load_model_prefix, epoch_idx)
    softmax_output = symbol.get_internals()["softmax_output"]
    model = mx.mod.Module(symbol=softmax_output, context=context, data_names=['data'], label_names=['softmax_label'])
    model.bind(data_shapes=val_iter.provide_data,
               label_shapes=val_iter.provide_label, for_training=False, force_rebind=True)

    model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

    acc = mx.metric.Accuracy(output_names=["softmax_output"], label_names=["softmax_label"], name="acc")
    ce_loss = mx.metric.CrossEntropy(output_names=["softmax_output"], label_names=["softmax_label"], name="ce")
    metric_list = [acc]
    metric = mx.metric.CompositeEvalMetric(metrics=metric_list)

    # eval
    res = model.score(val_iter, metric)
    for name, val in res:
        print('Epoch[%d] Validation-%s=%f' % (epoch_idx, name, val))


