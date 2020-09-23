import os
import numpy as np
import cv2
from sklearn.model_selection import KFold


def split_data(data_list, fold, output_path, seed, split_via_case):
    kf = KFold(n_splits=fold, random_state=seed, shuffle=True)
    kf_dict = {}
    if split_via_case:
        case2img = {}
        for d in data_list:
            c = d.split('_')[0]
            if c not in case2img:
                case2img[c] = []
            case2img[c].append(d)
        data_list = [k for k, _ in case2img.items()]
    for i, (train_index, test_index) in enumerate(kf.split(data_list)):
        if i not in kf_dict:
            kf_dict[i] = {}
        kf_dict[i]['train'] = [data_list[x] for x in train_index]
        kf_dict[i]['val'] = [data_list[x] for x in test_index]

    for k, v in kf_dict.items():
        with open(os.path.join(output_path, 'train_list_{}'.format(k)), 'w') as f:
            for i in v['train']:
                if split_via_case:
                    for l in case2img[i]:
                        f.write(l)
                else:
                    f.write(i)
        with open(os.path.join(output_path, 'val_list_{}'.format(k)), 'w') as f:
            for i in v['val']:
                if split_via_case:
                    for l in case2img[i]:
                        f.write(l)
                else:
                    f.write(i)


def grey2heat(grey):
    heat_stages_grey = np.array((0, 64, 128, 192, 256))
    heat_stages_color = np.array(((0, 0, 0), (0, 0, 64), (0, 255, 0), (255, 255, 0), (255, 0, 0)))
    #     heat_stages_grey = np.array((0, 128, 129, 192, 256))
    #     heat_stages_color = np.array(((0, 0, 0), (0, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 0)))
    np.clip(grey, 0, 255)

    for i in range(1, len(heat_stages_grey)):
        if heat_stages_grey[i] > grey >= heat_stages_grey[i - 1]:
            weight = (grey - heat_stages_grey[i - 1]) / float(heat_stages_grey[i] - heat_stages_grey[i - 1])
            color = weight * heat_stages_color[i] + (1 - weight) * heat_stages_color[i - 1]
            break
    return color.astype(np.int)


def fusion(im1, im2, weight=0.5):
    f_im = im1 * weight + im2 * (1 - weight)
    f_im = f_im.astype(np.uint8)
    return f_im


def weighted_sigmoid(arr, w=1):
    return 1. / (1 + np.exp(-arr * w))


def get_raw_img(inp):
    # inp = inp.view(3, 512, 512)
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = inp*255
    inp = np.clip(inp, 0, 255)
    inp = inp.astype(np.int)
    return inp


def save_heat_map(images, feature_maps, weights, preds, filenames, output, gt={}):
    if not os.path.exists(os.path.join(output, 'CAM')):
        os.makedirs(os.path.join(output, 'CAM'))
    for i, img in enumerate(images):
        raw_img = get_raw_img(img)
        H, W = raw_img.shape[:2]
        grey_map = feature_maps[i] * weights[preds[i]]
        grey_map = grey_map.sum(axis=0)
        grey_map = (grey_map - grey_map.min()) / (grey_map.max() - grey_map.min())
        grey_map *= 255
        # grey_map = weighted_sigmoid(grey_map, 0.1) * 255
        h, w = grey_map.shape
        heat_map = np.zeros((h, w, 3), np.uint8)
        for k in range(h):
            for j in range(w):
                heat_map[k, j] = grey2heat(grey_map[k, j])

        heat_map = cv2.resize(heat_map, (H, W))
        fusion_img = fusion(raw_img, heat_map, 0.7)

        result = np.zeros((512, 1044, 3), dtype=np.uint8)
        result[:, :512, :] = heat_map
        result[:, 512:532, :] = 255
        result[:, 532:, :] = fusion_img

        cv2.putText(result, 'pred: {}'.format(preds[i]), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if len(gt) != 0:
            cv2.putText(result, 'target: {}'.format(gt[filenames[i]]), (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imwrite(os.path.join(output, 'CAM', filenames[i]), result[:, :, ::-1])
