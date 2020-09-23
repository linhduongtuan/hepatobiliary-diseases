import torch
import numpy as np
import time
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

from tools import evalute, avg_f1


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(model, loader, criterion, optimizer, schedler, config,
          train_log='train_log', val_log='val_log', saved_model='./model'):
    epochs = config['model']['epochs']
    num_samples = config['num_samples'] if config['balanced'] else loader[0].dataset.__len__()
    gpu_ids = config['gpu_ids']
    mixup = config['mixup']
    best_via_auc = {
        'acc': 0.0,
        'auc': 0.0,
        'sensitivity': 0.0,
        'specificity': 0.0,
        'f1': 0.0,
        'confusion matrix': ''
    }
    best_via_f1 = {
        'acc': 0.0,
        'auc': 0.0,
        'sensitivity': 0.0,
        'specificity': 0.0,
        'f1': 0.0,
        'confusion matrix': ''
    }
    for epoch in range(epochs):
        begin = time.time()
        logs = open(train_log, 'a')
        model.train()
        running_corrects = 0
        running_loss = 0.0
        all_preds = []
        all_targets = []
        all_scores = []
        schedler.step()
        for i, (images, labels) in enumerate(loader[0]):
            start = time.time()
            images = images.cuda()
            labels = labels.cuda()

            if mixup:
                images, targets_a, targets_b, lam = mixup_data(images, labels)
                images, targets_a, targets_b = map(Variable, (images,
                                                              targets_a, targets_b))

            outputs = model(images)
            if mixup:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            if mixup:
                running_corrects += (lam * preds.eq(targets_a.data).cpu().sum().float()
                                     + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float())
            else:
                running_corrects += torch.sum(preds == labels.data)
            all_preds.append(preds.detach().cpu().numpy().copy())
            all_targets.append(labels.cpu().numpy().copy())
            all_scores.append(torch.softmax(outputs, dim=1).detach().cpu().numpy().copy())

            if i % 10 == 0:
                print('Epoch: {}/{}, Iter: {}/{:.0f}, Loss: {:.4f}, Time: {:.4f}s/batch'
                      .format(epoch + 1, epochs, i, len(loader[0]) + 1, loss.item(), time.time() - start))
        epoch_loss = running_loss / len(loader[0])
        epoch_acc = running_corrects.double() / num_samples

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_scores = np.concatenate(all_scores)
        auc = roc_auc_score(all_targets, all_scores[:, 1])
        cm, sens, spec, f1 = avg_f1(all_targets, all_preds)
        log = 'Train Epoch: {}/{}, Loss: {:.4f}, ' \
              'Acc: {:.0f}/{}, {:.4f}, Auc: {:.4f}, ' \
              'sens: {:.4f}, spec: {:.4f}, f1: {:.4f}, ' \
              'Time: {:.0f}s'.format(epoch + 1, epochs, epoch_loss,
                                     running_corrects, num_samples,
                                     epoch_acc, auc,
                                     sens, spec, f1, time.time() - begin)

        print(log)
        logs.write(log + '\n')

        val_acc, val_auc, val_sens, val_spec, val_f1, val_cm = evalute(epoch, epochs, model,
                                                                       loader[1], criterion,
                                                                       write_log=val_log)
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict() if len(gpu_ids) > 1 else model.state_dict(),
            'val_acc': round(val_acc, 4),
            'val_auc': round(val_auc, 4),
            'val_f1': round(val_f1, 4),
            'val_sens': round(val_sens, 4),
            'val_spec': round(val_spec, 4),
            'conf_mat': val_cm
        }, '{}_latest.pkl'.format(saved_model))
        if best_via_f1['f1'] < val_f1:
            best_via_f1['acc'] = round(val_acc, 4)
            best_via_f1['auc'] = round(val_auc, 4)
            best_via_f1['sensitivity'] = round(val_sens, 4)
            best_via_f1['specificity'] = round(val_spec, 4)
            best_via_f1['f1'] = round(val_f1, 4)
            best_via_f1['confusion matrix'] = '[[{}, {}], [{}, {}]]'.format(val_cm[0, 0], val_cm[0, 1],
                                                                            val_cm[1, 0], val_cm[1, 1])
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if len(gpu_ids) > 1 else model.state_dict(),
                'val_acc': round(val_acc, 4),
                'val_auc': round(val_auc, 4),
                'val_f1': round(val_f1, 4),
                'val_sens': round(val_sens, 4),
                'val_spec': round(val_spec, 4),
                'conf_mat': val_cm
            }, '{}_best_f1.pkl'.format(saved_model))

        if best_via_auc['auc'] < val_auc:
            best_via_auc['acc'] = round(val_acc, 4)
            best_via_auc['auc'] = round(val_auc, 4)
            best_via_auc['sensitivity'] = round(val_sens, 4)
            best_via_auc['specificity'] = round(val_spec, 4)
            best_via_auc['f1'] = val_f1
            best_via_auc['confusion matrix'] = '[[{}, {}], [{}, {}]]'.format(val_cm[0, 0], val_cm[0, 1],
                                                                             val_cm[1, 0], val_cm[1, 1])
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if len(gpu_ids) > 1 else model.state_dict(),
                'val_acc': round(val_acc, 4),
                'val_auc': round(val_auc, 4),
                'val_f1': round(val_f1, 4),
                'val_sens': round(val_sens, 4),
                'val_spec': round(val_spec, 4),
                'conf_mat': val_cm
            }, '{}_best_auc.pkl'.format(saved_model))

    return best_via_auc, best_via_f1
