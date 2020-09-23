# -*- coding: utf-8 -*-
import os
import pandas as pd
import shutil
import numpy as np
import tools.gene_folder as gf


def save_result(dataset, targets, pred_scores, pred_label, root):
    """
    说明：该函数把预测的结果和真实的标签进行整合输出，并把真正例，真负例，误报以及漏报的图像输出到对应的文件夹
    :param dataset: 测试集
    :param targets: 图像的真实标签
    :param pred_scores: 图像的预测得分
    :param pred_label: 图像的预测标签
    :param root: 保存根路径
    :return: 无
    """
    scores = pd.DataFrame(columns=['ID', 'Image', 'Label', 'Score[0]', 'Score[1]', 'Pred_Label'])
    scores['ID'] = dataset.z                                                                                            # 患者编号
    scores['Image'] = dataset.t                                                                                         # 图像名称
    scores['Label'] = targets                                                                                           # 真实标签
    scores['Score[0]'] = np.around(pred_scores[:, 0], decimals=4)                                                       # 预测类别0的得分
    scores['Score[1]'] = np.around(pred_scores[:, 1], decimals=4)                                                       # 预测类别1的得分
    scores['Pred_Label'] = pred_label                                                                                   # 预测类别
    scores.to_csv(os.path.join(os.path.dirname(root), 'scores.txt'), sep=',', index=False)

    truePositivesList = []
    trueNegativesList = []
    missingReportList = []
    mistakeReportList = []

    truePositivesRoot, trueNegativesRoot, missingReportRoot, mistakeReportRoot = gf.gene_report_folder(root)

    for i in range(len(scores)):
        path = dataset.x[i]                                                                                             # 图像的保存路径
        true_label = scores.loc[i, 'Label']                                                                             # 真实标签
        pred_label = scores.loc[i, 'Pred_Label']                                                                        # 预测标签
        if (true_label == 1) and (pred_label == 1):
            shutil.copy(path, truePositivesRoot)                                                                        # 保存真正例
            truePositivesList.append(path)
        elif (true_label == 0) and (pred_label == 0):
            shutil.copy(path, trueNegativesRoot)                                                                        # 保存真负例
            trueNegativesList.append(path)
        elif (true_label == 1) and (pred_label == 0):
            shutil.copy(path, missingReportRoot)                                                                        # 保存漏报
            missingReportList.append(path)
        elif (true_label == 0) and (pred_label == 1):
            shutil.copy(path, mistakeReportRoot)                                                                        # 保存误报
            mistakeReportList.append(path)
        else:
            raise ArithmeticError('true label or pred label abnormal!')

    return truePositivesList, trueNegativesList, missingReportList, mistakeReportList
