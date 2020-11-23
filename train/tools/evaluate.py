import torch
import numpy as np
import time
from sklearn.metrics import confusion_matrix, roc_auc_score

from data.utils import save_heat_map
from utils import save_roc


def avg_f1(targets, preds):
    conf_mat = confusion_matrix(targets, preds)
    conf_mat = conf_mat.astype(np.int)

    TP = conf_mat[1, 1]
    TN = conf_mat[0, 0]
    TPR = TP / (conf_mat[1, :].sum()+1e-7)
    TNR = TN / (conf_mat[0, :].sum()+1e-7)

    f1 = (2*TPR*TNR) / (TPR+TNR+1e-7)

    return conf_mat, TPR, TNR, f1


def evaluate(epoch, epochs, model, loader, criterion, write_log='', _type='Val', draw_roc=False, title=None, name=None):
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

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.append(preds.cpu().numpy().copy())
            all_targets.append(labels.cpu().numpy().copy())
            all_scores.append(torch.softmax(outputs, dim=1).cpu().numpy().copy())
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
        if write_log != '':
            logs.write(log+'\n')

        if draw_roc:
            save_roc(all_targets, all_scores[:, 1], title, name)

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
