import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np


def save_fig(train_log, val_log, title, _type, name, margin, point_type):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.plot(train_log, label='training '+_type)
    plt.plot(val_log, label='val '+_type)
    if point_type == 'min':
        x0 = np.argmin(train_log)
        y0 = train_log[np.argmin(train_log)]
        x1 = np.argmin(val_log)
        y1 = val_log[np.argmin(val_log)]
        optimal = 'min'
    else:
        x0 = np.argmax(train_log)
        y0 = train_log[np.argmax(train_log)]
        x1 = np.argmax(val_log)
        y1 = val_log[np.argmax(val_log)]
        optimal = 'max'

    plt.plot([x0, x1], [y0, y1], 'o')
    plt.annotate('{}: ({}, {})'.format(optimal, x0, y0), xy=(x0, y0), xytext=(x0+margin, y0+margin), arrowprops=dict(arrowstyle="->"))
    plt.annotate('{}: ({}, {})'.format(optimal, x1, y1), xy=(x1, y1), xytext=(x1+margin, y1+margin), arrowprops=dict(arrowstyle="->"))
    #     plt.annotate(s="Nothing",  xytext=(x0+1, y0+1), xy=(x0, y0), arrowprops=dict(arrowstyle="->"))
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(_type)
    plt.savefig(name)


def plot_log(train_log, val_log, title, name, i1=4, i2=9, i3=15, margin1=0.01, margin2=0.01):
    train_logs = open(train_log).readlines()
    train_loss = [float(x.strip().split(' ')[i1].replace(',', '')) for x in train_logs]
    train_auc = [float(x.strip().split(' ')[i2].replace(',', '')) for x in train_logs]
    train_f1 = [float(x.strip().split(' ')[i3].replace(',', '')) for x in train_logs]

    val_logs = open(val_log).readlines()
    val_loss = [float(x.strip().split(' ')[i1].replace(',', '')) for x in val_logs]
    val_auc = [float(x.strip().split(' ')[i2].replace(',', '')) for x in val_logs]
    val_f1 = [float(x.strip().split(' ')[i3].replace(',', '')) for x in val_logs]

    save_fig(train_loss, val_loss, title+'_loss', 'loss', name+'_loss', margin1, 'min')
    save_fig(train_auc, val_auc, title+'_auc', 'auc', name+'_auc', margin2, 'max')
    save_fig(train_f1, val_f1, title+'_f1', 'f1', name+'_f1', margin2, 'max')
