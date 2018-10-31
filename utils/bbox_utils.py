"""
include the func to process bounding boxes
"""
import os
import numpy as np

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes


def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i, :] = boxes[i, :] / scales[int(inds[i])]

  return boxes


def _vis_detections(im, dets, im_name=''):
    """Draw detected bounding boxes."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    if dets.ndim == 1:
        dets = np.reshape(dets, [1, 4])
    # bbox = dets
    for bbox in dets:
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
    if im_name:
        fig.savefig(os.path.join('tmp', im_name), format='jpg')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def _check_bbox_transform():
    import cv2
    im_path = '/media/chencp/data_ssd2/datasets/indoor67/Images/artstudio/artistic_studio_05_19_altavista.jpg'

    im = cv2.imread(im_path)
    # cv2.cvtColor(im, code=cv2.COLOR_BGR2RGB, dst=im)
    # im = im.astype(np.float32)

    print('image original shape: {}'.format(im.shape))  # (210, 292, 3)
    rois = np.array([
        [120, 10, 200, 200],
        [70, 5, 100, 120],
        [20, 100, 160, 200],
        [100, 70, 120, 210]
    ])
    print('displaying the original image and bbox')
    print(rois[:, 2].max(), rois[:, 3].max())
    _vis_detections(im, rois, im_name='ori_img')

    # resize
    h, w = 256, 300
    ori_h, ori_w = im.shape[:2]
    img = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
    scales = (np.float(h) / ori_h, np.float(w) / ori_w)
    rois = rois * np.array([[scales[1], scales[0], scales[1], scales[0]]])

    print(rois[:, 2].max(), rois[:, 3].max())
    print('displaying the resized image and bbox')
    _vis_detections(img, rois, im_name='resize_img')

    # crop
    new_w, new_h = (224, 188)
    h, w = img.shape[:2]
    x, y = int(round((w - new_w) / 2)), int(round((h - new_h) / 2))
    if (new_h, new_w) != (h, w):
        img = img[y:y + new_h, x:x + new_w, :]
        rois -= np.array([[x, y, x, y]])
        rois = _clip_boxes(rois, (new_h, new_w))
    print(rois[:, 2].max(), rois[:, 3].max())
    print('displaying the croped image and bbox')
    _vis_detections(img, rois, im_name='crop_img')

    # flip
    img = cv2.flip(img, flipCode=1)
    rois[:, 0], rois[:, 2] = new_w - rois[:, 2] - 1, new_w - rois[:, 0] - 1

    print(rois[:, 2].max(), rois[:, 3].max())
    print('displaying the fliped image and bbox')
    _vis_detections(img, rois, im_name='flip_img')


def _sample_rois(roi_dict, roi_aug_dict):
    # not put rois into the queue, only preprocess it in dict (nms and select rois with roi num)
    roi_dict_pro = dict()
    for img, rois in roi_dict.items():
        # todo: sample rois, e.g. applying nms and random sample
        # drop score for each roi
        width = rois[:, 2] - rois[:, 0] + 1
        height = rois[:, 3] - rois[:, 1] + 1
        area = width * height
        if roi_aug_dict['enlarge_factor'] > 0:
            # enlarge the bbox to include more context, no need to clip
            width *= 1 + roi_aug_dict['enlarge_factor']
            height *= 1 + roi_aug_dict['enlarge_factor']
            x_center = (rois[:, 2] + rois[:, 0] - 1) / 2
            y_center = (rois[:, 3] + rois[:, 1] - 1) / 2
            rois[:, 0] = x_center - width / 2
            rois[:, 2] = x_center + width / 2
            rois[:, 1] = y_center - height / 2
            rois[:, 3] = y_center + height / 2

        if roi_aug_dict['drop_small_scale '] > 0:
            ind = np.where(area > roi_aug_dict[roi_aug_dict['drop_small_scale ']**2])[0]
            rois = rois[ind]

        if rois.shape[0] >= roi_aug_dict['roi_num']:
            scores = rois[:, -1][:]
            scores = scores * np.sqrt(area)  # prefer to large region
            ind = np.random.choice(len(roi_dict), size=(roi_aug_dict['roi_num'],), replace=False, p=scores)
            rois_new = rois[ind, :4]
        else:
            # no enough rois, need to generate some to compensate it
            rois_new = rois[:, :4]
            append_num = roi_aug_dict['roi_num'] - rois.shape[0]
            rois_generate = _rois_gen(append_num)


        roi_dict_pro[img] = rois_new
        # print(rois_new[0])
    return roi_dict_pro


if __name__ == '__main__':
    pass
