import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import pandas as pd


def save_roc(targets, scores, title, name):
    plt.figure()
    fpr, tpr, _ = roc_curve(targets, scores)
    roc_auc = auc(fpr, tpr)

    #roc_auc = round(roc_auc, 2)
    #roc_auc = str(roc_auc)
    #roc_auc = roc_auc[0] + '·' + roc_auc[2:]

    #     plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr,color='steelblue', label='ROC curve (area = %0.2f)' % roc_auc)
    #plt.plot(fpr, tpr, color='steelblue', label='{} (AUC={},{}-{})'.format('Hepatobiliary disease', roc_auc,'0·65','0·71'))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    #plt.title(title)
    plt.legend(loc='lower right',prop = {'size':7.5})
    #plt.legend(loc='best')
    plt.savefig(name + '.tiff')


def all_roc_plot(root,name,diseases_types,colors,confidence_set):
    for j,txt_file in enumerate(os.listdir(root)):
        if txt_file == 'gray_iris':
            continue
        results = open(root + txt_file).read().split('\n')
        labels = []
        scores = []
        for i,row in enumerate(results):
            row = row.split(',')
            if i == 0 or i == len(results) - 1:
                continue
            labels.append(int(row[2]))
            scores.append(float(row[4]))

        #plt.figure()
        fpr, tpr, _ = roc_curve(labels, scores)

        roc_auc = auc(fpr, tpr)
        roc_auc = round(roc_auc,2)
        roc_auc = str(roc_auc)
        if len(roc_auc) != 4:
            roc_auc = roc_auc + '0'
        roc_auc = roc_auc[0] + '·' + roc_auc[2:]

        plt.plot([0, 1], [0, 1], 'k--')
        #plt.tick_params(labelsize=5)
       # plt.plot(fpr, tpr,color=colors[j], label='{} (AUC=%0.2f,{}-{})'.format(diseases_types[j],confidence_set[j][0],confidence_set[j][1]) % roc_auc)
        plt.plot(fpr, tpr, color=colors[j], label='{} (AUC={},{}-{})'.format(diseases_types[j], roc_auc,confidence_set[j][0],
                                                                                confidence_set[j][1]))
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    #plt.title(title)
    plt.legend(loc='best',prop = {'size':7.5})
    #plt.savefig(name + '.tiff')
    plt.savefig(name + '.pdf', format='PDF', transparent=True, dpi=300, pad_inches=0)



if __name__ == '__main__':
    root_slitlamp = 'D:\ZOC_DATA\liver_disease2\models\\slitlamp\\all_scores\\'
    root_fundus = 'D:\ZOC_DATA\liver_disease2\models\\fundus\\all_scores\\'
    name = 'fundus_0+6diseases'
    diseases_types = ['Hepatobiliary Diseases','Liver cancer','Liver cirrhosis','Chronic viral Hepatitis',
                      'Nonalcoholic fatty liver disease','Cholelithiasis',
                      'Hepatic cyst']
    colors = ['crimson','chocolate','steelblue','rosybrown','darkkhaki','cadetblue','lightslategrey']
    # slitlamp
    #confidence_set = [['0·71','0·76'],['0·91','0·94'], ['0·88','0·91'], ['0·66','0·71'], ['0·60','0·66'], ['0·55','0·61'], ['0·63','0·68']]
    #all_roc_plot(root_slitlamp, name, diseases_types, colors, confidence_set)

    # fundus
    confidence_set=[['0·65','0·71'],['0·81','0·86'],['0·81','0·86'],['0·58','0·65'],['0·67','0·73'],['0·65','0·71'],['0·65','0·72']]
    all_roc_plot(root_fundus,name,diseases_types,colors,confidence_set)


    #from PIL import Image
    #im = Image.open('./slitlamp_0+6diseases.tiff')
    #im.save('slitlamp_0+6diseases1.tiff', dpi=(300.0, 300.0))
    #plt.savefig()
    #im.savefig('slitlamp_0+6diseases1.tiff', format='PDF', transparent=True, dpi=300, pad_inches=0)