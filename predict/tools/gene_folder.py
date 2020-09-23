# -*- coding: utf-8 -*-
import os
import shutil


def reset_folder(root_path):
    """
    说明：该函数用来清空文件夹，并重新创建文件夹，防止重新执行时有上次残留的数据
    :param root_path: 文件夹路径
    :return: 无
    """
    if os.path.exists(root_path):
        shutil.rmtree(root_path)
    os.makedirs(root_path)


def gene_report_folder(root_path):
    """
    说明：该函数用来生成保留图像的文件夹
    :param root_path: 文件夹根路径
    :return: 创建的文件夹路径
    """
    truePositivesFolder = os.path.join(os.path.dirname(root_path), 'true_positives_report(true_1, pred_1)', 'orig')     # 真正例路径
    trueNegativesFolder = os.path.join(os.path.dirname(root_path), 'true_negatives_report(true_0, pred_0)', 'orig')     # 真负例路径
    missingReportFolder = os.path.join(os.path.dirname(root_path), 'missing_report(true_1, pred_0)', 'orig')            # 漏报路径
    mistakeReportFolder = os.path.join(os.path.dirname(root_path), 'mistake_report(true_0, pred_1)', 'orig')            # 误报路径

    # 清空文件夹并重新创建空的文件夹
    reset_folder(truePositivesFolder)
    reset_folder(trueNegativesFolder)
    reset_folder(missingReportFolder)
    reset_folder(mistakeReportFolder)

    return truePositivesFolder, trueNegativesFolder, missingReportFolder, mistakeReportFolder


class gene_heatmap_folder(object):
    """
    说明：该类用来生成热度图对应的文件夹
    """
    def __init__(self, root_path):
        self.path = root_path

        # 真正例下热度图文件夹路径
        self.guidedBPTPFolder = ""
        self.camHeatmapTPFolder = ""
        self.camHeatmapOnImageTPFolder = ""
        self.gradCamHeatmapTPFolder = ""
        self.gradCamHeatmapOnImageTPFolder = ""
        self.gradCamGuidedGradCamTPFolder = ""

        # 真负例下热度图文件夹路径
        self.guidedBPTNFolder = ""
        self.camHeatmapTNFolder =""
        self.camHeatmapOnImageTNFolder = ""
        self.gradCamHeatmapTNFolder = ""
        self.gradCamHeatmapOnImageTNFolder = ""
        self.gradCamGuidedGradCamTNFolder = ""

        # 漏报下热度图文件夹路径
        self.guidedBPMissFolder = ""
        self.camHeatmapMissFolder = ""
        self.camHeatmapOnImageMissFolder = ""
        self.gradCamHeatmapMissFolder = ""
        self.gradCamHeatmapOnImageMissFolder = ""
        self.gradCamGuidedGradCamMissFolder = ""

        # 误报下热度图文件夹路径
        self.guidedBPMistFolder = ""
        self.camHeatmapMistFolder = ""
        self.camHeatmapOnImageMistFolder = ""
        self.gradCamHeatmapMistFolder = ""
        self.gradCamHeatmapOnImageMistFolder = ""
        self.gradCamGuidedGradCamMistFolder = ""

        self.heatmap_folder(self.path)

    def type_folder(self, root_path, type):
        """
        说明：该函数生成真正例，真负例，漏报和误报下的热度图文件夹
        :param root_path: 根路径
        :param type: 可取真正例，真负例，漏报和误报四种类型
        :return: 路径名
        """
        guidedBPFolder = os.path.join(os.path.dirname(root_path), type, 'guided_backprop')
        camHeatmapFolder = os.path.join(os.path.dirname(root_path), type, 'cam_heatmap')
        camHeatmapOnImageFolder = os.path.join(os.path.dirname(root_path), type, 'cam_heatmap_on_image')
        gradCamHeatmapFolder = os.path.join(os.path.dirname(root_path), type, 'grad_cam_heatmap')
        gradCamHeatmapOnImageFolder = os.path.join(os.path.dirname(root_path), type, 'grad_cam_heatmap_on_image')
        gradCamGuidedGradCamFolder = os.path.join(os.path.dirname(root_path), type, 'grad_cam_guided_grad_cam')

        reset_folder(guidedBPFolder)
        reset_folder(camHeatmapFolder)
        reset_folder(camHeatmapOnImageFolder)
        reset_folder(gradCamHeatmapFolder)
        reset_folder(gradCamHeatmapOnImageFolder)
        reset_folder(gradCamGuidedGradCamFolder)

        return guidedBPFolder, camHeatmapFolder, camHeatmapOnImageFolder, \
               gradCamHeatmapFolder, gradCamHeatmapOnImageFolder, gradCamGuidedGradCamFolder

    def heatmap_folder(self, root_path):
        """
        说明：该函数生成热度图对应的文件夹
        :param root_path: 根路径
        :return: 文件夹路径
        """
        # 生成真正例下热度图文件夹
        self.guidedBPTPFolder, self.camHeatmapTPFolder, self.camHeatmapOnImageTPFolder, self.gradCamHeatmapTPFolder, \
        self.gradCamHeatmapOnImageTPFolder, self.gradCamGuidedGradCamTPFolder = self.type_folder(root_path, 'true_positives_report(true_1, pred_1)')

        # 生成真负例下热度图文件夹
        self.guidedBPTNFolder, self.camHeatmapTNFolder, self.camHeatmapOnImageTNFolder, self.gradCamHeatmapTNFolder, \
        self.gradCamHeatmapOnImageTNFolder, self.gradCamGuidedGradCamTNFolder = self.type_folder(root_path, 'true_negatives_report(true_0, pred_0)')

        # 生成漏报下热度图文件夹
        self.guidedBPMissFolder, self.camHeatmapMissFolder, self.camHeatmapOnImageMissFolder, self.gradCamHeatmapMissFolder, \
        self.gradCamHeatmapOnImageMissFolder, self.gradCamGuidedGradCamMissFolder = self.type_folder(root_path, 'missing_report(true_1, pred_0)')

        # 生成误报下热度图文件夹
        self.guidedBPMistFolder, self.camHeatmapMistFolder, self.camHeatmapOnImageMistFolder, self.gradCamHeatmapMistFolder, \
        self.gradCamHeatmapOnImageMistFolder, self.gradCamGuidedGradCamMistFolder = self.type_folder(root_path, 'mistake_report(true_0, pred_1)')
