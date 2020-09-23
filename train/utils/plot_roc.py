import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def save_roc(targets, scores, title, name):
    plt.figure()
    fpr, tpr, _ = roc_curve(targets, scores)
    roc_auc = auc(fpr, tpr)
    #     plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    plt.legend(loc='best')
    plt.savefig(name)
