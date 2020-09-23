import torch
import numpy as np
import time
import os
import pandas as pd
import shutil
import copy
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch.nn.functional as F

from data.utils import save_heat_map
from utils import save_roc
from tools.save_scores import save_result


def mask2bbox(attention_maps):
    height = attention_maps.shape[2]
    width = attention_maps.shape[3]
    thetas = []
    for i in range(attention_maps.shape[0]):
        mask = attention_maps[i][0]
        max_activate = mask.max()
        min_activate = 0.1 * max_activate
        #         mask = (mask >= min_activate)
        itemindex = torch.nonzero(mask >= min_activate)
        ymin = itemindex[:, 0].min().item() / height - 0.05
        ymax = itemindex[:, 0].max().item() / height + 0.05
        xmin = itemindex[:, 1].min().item() / width - 0.05
        xmax = itemindex[:, 1].max().item() / width + 0.05
        a = xmax - xmin
        e = ymax - ymin
        # crop weight=height
        pad = abs(a - e) / 2.
        if a <= e:
            a = e
            xmin -= pad
        else:
            e = a
            ymin -= pad
        c = 2 * xmin - 1 + a
        f = 2 * ymin - 1 + e
        theta = np.asarray([[a, 0, c], [0, e, f]], dtype=np.float32)
        thetas.append(theta)
    thetas = np.asarray(thetas, np.float32)
    return thetas


def avg_f1(targets, preds):
    conf_mat = confusion_matrix(targets, preds)
    conf_mat = conf_mat.astype(np.int)

    TP = conf_mat[1, 1]
    TN = conf_mat[0, 0]
    TPR = TP / (conf_mat[1, :].sum()+1e-7)
    TNR = TN / (conf_mat[0, :].sum()+1e-7)

    f1 = (2*TPR*TNR) / (TPR+TNR+1e-7)

    return conf_mat, TPR, TNR, f1


def evalute(epoch, epochs, model, model_name, loader, dataset, criterion, threshold=0.5, write_log='', _type='Val',
            draw_roc=False, title=None, name=None, save_pred=False, save_scores=False):
    begin = time.time()
    if write_log != '':
        logs = open(write_log, 'a')
    model.eval()
    with torch.no_grad():
        running_corrects = 0
        running_loss = 0.0
        all_preds = []
        all_targets = []
        all_scores = []
        for i, (images, labels) in enumerate(loader):
            images = images.cuda()
            labels = labels.cuda()

            if model_name != 'ws_dan_resnet50':
                outputs = model(images)
                loss = criterion[0](outputs, labels)
            else:
                outputs, _, attention_map = model(images)
                batch_loss = criterion[0](outputs, labels)

                # mask crop
                attention_map = torch.mean(attention_map, dim=1).unsqueeze(1)
                attention_map = F.interpolate(attention_map, size=(images.size(2), images.size(3)))
                thetas = mask2bbox(attention_map)
                thetas = torch.from_numpy(thetas).cuda()
                grid = F.affine_grid(thetas, images.size(), align_corners=True)
                crop_images = F.grid_sample(images, grid, align_corners=True)
                outputs1, _, _ = model(crop_images)
                mask_loss = criterion[0](outputs1, labels)

                loss = (batch_loss + mask_loss) / 2
                outputs = outputs + outputs1

            outputs = torch.softmax(outputs, dim=1)
            # _, preds = torch.max(outputs, 1)
            preds = torch.tensor([1 if item[1] >= threshold else 0 for item in outputs]).cuda()
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.append(preds.cpu().numpy().copy())
            all_targets.append(labels.cpu().numpy().copy())
            all_scores.append(outputs.cpu().numpy().copy())
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_scores = np.concatenate(all_scores)

        auc = roc_auc_score(all_targets, all_scores[:, 1])
        cm, sens, spec, f1 = avg_f1(all_targets, all_preds)
        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = running_corrects.double() / len(loader.dataset)
        log = '{} Epoch: {}/{}, Loss: {:.4f}, ' \
              'Acc: {:.0f}/{}, {:.4f}, Auc: {:.4f}, ' \
              'sens: {:.4f}, spec: {:.4f}, f1: {:.4f}, ' \
              'Time: {:.0f}s'.format(_type, epoch+1, epochs, epoch_loss,
                                     running_corrects, len(loader.dataset),
                                     epoch_acc, auc, sens, spec, f1, time.time()-begin)

        print(log)
        # print(cm)
        if write_log != '':
            logs.write(log+'\n')

        if draw_roc:
            save_roc(all_targets, all_scores[:, 1], title, name)

        if save_scores:
            true_positives, true_negatives, missing_report, mistake_report = save_result(dataset, all_targets, all_scores, all_preds, name)

        if save_pred:
            return epoch_acc.item(), auc.item(), sens.item(), spec.item(), f1.item(), cm, all_preds, \
                   true_positives, true_negatives, missing_report, mistake_report
        else:
            return epoch_acc.item(), auc.item(), sens.item(), spec.item(), f1.item(), cm


def inference(model, loader, output, weights, gt={}):
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_scores = []
        for i, (images, filenames) in enumerate(loader):
            images = images.cuda()

            outputs, feature = model(images, True)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy().copy())
            outputs = torch.softmax(outputs, dim=1)
            all_scores.append(outputs.cpu().numpy().copy())
            save_heat_map(images, feature.cpu().numpy().copy(), weights, preds.cpu().numpy().copy(), filenames, output, gt)

        all_preds = np.concatenate(all_preds)
        all_scores = np.concatenate(all_scores)
        return all_preds, all_scores
