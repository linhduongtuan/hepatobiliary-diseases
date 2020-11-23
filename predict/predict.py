import argparse
import os
import json
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import torch.utils.data as _data

from data import LiverSingleDataset
from model import resnet50, resnet101, inception_v3, vgg16, vgg19_bn, ws_dan_resnet50
from tools import evalute
from utils import gene_grad_cam, gene_guided_bp
import tools.gene_folder as gf


parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='config/slitlamp_test/config1.json', help='config file')
args = parser.parse_args()


class Predict(object):
    """

    """

    def __init__(self, config):
        if not isinstance(config, str):
            self.config = config
        else:
            assert os.path.exists(config)
            self.config = json.load(open(config))
        assert 'name' in self.config
        assert 'data_path' in self.config
        assert 'balanced' in self.config
        assert 'num_samples' in self.config
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config['gpu_ids']

        self.date = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

        # create project folder
        if not os.path.exists(os.path.join('./results/', self.config['name'])):
            os.makedirs(os.path.join('./results/', self.config['name']))

        if self.config['model']['name'] == 'resnet50':
            self.model = resnet50(pretrained=True, num_classes=self.config['model']['num_classes'])
        elif self.config['model']['name'] == 'resnet101':
            self.model = resnet101(pretrained=True, num_classes=self.config['model']['num_classes'])
        elif self.config['model']['name'] == 'inception_v3':
            self.model = inception_v3(pretrained=True, num_classes=self.config['model']['num_classes'])
        elif self.config['model']['name'] == 'vgg16':
            self.model = vgg16(pretrained=True, num_classes=self.config['model']['num_classes'])
        elif self.config['model']['name'] == 'vgg19_bn':
            self.model = vgg19_bn(pretrained=True, num_classes=self.config['model']['num_classes'])
        elif self.config['model']['name'] == 'ws_dan_resnet50':
            self.model = ws_dan_resnet50(pretrained=True, num_classes=self.config['model']['num_classes'], num_attentions=self.config['model']['num_attentions'])
        self.model.cuda()
        if len(self.config['gpu_ids']) > 1:
            self.model = nn.DataParallel(self.model)
        #self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 3.0]).cuda())                               #数据不均衡时可修改损失函数权重
        self.criterion_attention = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['model']['init_lr'], momentum=0.9,
                                   weight_decay=1e-4)
        # self.exp_lr_schedler = lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['model']['milestones'], gamma=0.1)
        # self.exp_lr_schedler = lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.9)
        self.exp_lr_schedler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.6)

    def load_state(self, _type='test'):
        if self.config['model']['name'] == 'resnet50':
            self.model = resnet50(pretrained=False, num_classes=self.config['model']['num_classes'])
        elif self.config['model']['name'] == 'resnet101':
            self.model = resnet101(pretrained=False, num_classes=self.config['model']['num_classes'])
        elif self.config['model']['name'] == 'inception_v3':
            self.model = inception_v3(pretrained=False, num_classes=self.config['model']['num_classes'])
        elif self.config['model']['name'] == 'vgg16':
            self.model = vgg16(pretrained=False, num_classes=self.config['model']['num_classes'])
        elif self.config['model']['name'] == 'vgg19_bn':
            self.model = vgg19_bn(pretrained=False, num_classes=self.config['model']['num_classes'])
        elif self.config['model']['name'] == 'ws_dan_resnet50':
            self.model = ws_dan_resnet50(pretrained=True, num_classes=self.config['model']['num_classes'], num_attentions=self.config['model']['num_attentions'])

        if _type == 'test':
            # checkpoints = os.path.join('./zhongshan/new_test_file_20200119', self.config['name'], 'checkpoints',
            #                            self.config['test']['checkpoint'])
            checkpoints = os.path.join('.', self.config['test']['checkpoint'])
        elif _type == 'test_batch':
            checkpoints = self.config['test']['checkpoint']
        else:
            checkpoints = os.path.join('./results', self.config['name'], 'checkpoints',
                                       self.config['inference']['checkpoint'])
        self.model.load_state_dict(torch.load(checkpoints)['state_dict'])
        self.model.cuda()
        if len(self.config['gpu_ids']) > 1:
            self.model = nn.DataParallel(self.model)

        # for param in self.model.named_parameters():
        #     print(param)

        # print ('{}, Acc: {:.4f}, Auc: {:.4f}, ' \
        #        'sens: {:.4f}, spec: {:.4f}, f1: {:.4f}'\
        #        .format("Val", torch.load(checkpoints)['val_acc'], torch.load(checkpoints)['val_auc'],
        #                torch.load(checkpoints)['val_sens'], torch.load(checkpoints)['val_spec'],
        #                torch.load(checkpoints)['val_f1']))

    def test(self):
        self.load_state()
        os.makedirs(os.path.join('./results', self.config['name'], 'test'), exist_ok=True)
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config['image']['height'], self.config['image']['width'])),
            transforms.CenterCrop((self.config['image']['height'], self.config['image']['width'])),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        testset = LiverSingleDataset(transform=test_transform, root=self.config['data_path'], sclera=self.config['sclera'],
                                     _type=self.config['type'], data_file=self.config['test_list'])
        testloader = _data.DataLoader(testset, shuffle=False, batch_size=self.config['model']['batch_size'],
                                      num_workers=self.config['model']['num_workers'])

        if self.config['type'] == 'fundus':
            testset_orig = LiverSingleDataset(transform=test_transform, root=self.config['data_path_orig'], sclera=self.config['sclera'],
                                              _type=self.config['type'], data_file=self.config['test_list'])
            testloader_vis = _data.DataLoader(testset_orig, shuffle=False, batch_size=1, num_workers=0)
        elif self.config['type'] == 'slitlamp':
            testloader_vis = _data.DataLoader(testset, shuffle=False, batch_size=1, num_workers=0)
        else:
            raise ValueError('type must be fundus or slitlamp')

        title = self.config['name'] + '_' + self.config['model']['name']
        name = os.path.join('./results', self.config['name'], 'test', self.config['test']['checkpoint'].replace('.pkl', ''))
        os.makedirs(os.path.dirname(name), exist_ok=True)
        '''
        test_acc, test_auc, test_sens, test_spec, test_f1, test_cm, test_preds, \
        test_auc_confidence, test_sens_confidence, test_spec_confidence, \
        test_true_positives, test_true_negatives, test_missing_report, test_mistake_report = evalute(0, 1, self.model, self.config['model']['name'],
                                                                                                     testloader, testset, [self.criterion, self.criterion_attention],
                                                                                                     self.config['model']['category_threshold'], _type='Test',
                                                                                                     draw_roc=True, title=title, name=name, save_pred=True,
                                                                                                     save_scores=False)
        '''
        test_acc, test_auc, test_sens, test_spec, test_f1, test_cm, test_preds, \
        test_auc_confidence, test_sens_confidence, test_spec_confidence = evalute(0, 1, self.model, self.config['model']['name'],
                                                                                                     testloader, testset, [self.criterion, self.criterion_attention],
                                                                                                     self.config['model']['category_threshold'], _type='Test',
                                                                                                     draw_roc=True, title=title, name=name, save_pred=True,
                                                                                                     save_scores=False)
        #'''

        test_result = {
            'acc': round(test_acc, 4),
            'auc': round(test_auc, 4),
            'test_auc_confidence': test_auc_confidence,
            'sensitivity': round(test_sens, 4),
            'test_sens_confidence': test_sens_confidence,
            'specificity': round(test_spec, 4),
            'test_spec_confidence': test_spec_confidence,
            'f1': round(test_f1, 4),
            'confusion matrix': '[[{}, {}], [{}, {}]]'.format(test_cm[0, 0], test_cm[0, 1], test_cm[1, 0], test_cm[1, 1])
        }
        if not os.path.exists(os.path.join(os.path.dirname(name), 'test_auc{}'.format(str(round(test_auc,2))))):
            os.makedirs(os.path.join(os.path.dirname(name), 'test_auc{}'.format(str(round(test_auc,2)))))
        with open(os.path.join(os.path.dirname(name), 'test_auc{}'.format(str(round(test_auc,2))),'test_results.json'), 'w') as f:
            json.dump(test_result, f, indent=4)


        ''''
        # print('start drawing heatmap...')
        #
        ghf = gf.gene_heatmap_folder(name)

        if len(self.config['gpu_ids']) > 1:
            weights = self.model.module.fc.weight.data
        else:
            weights = self.model.fc.weight.data
        weights = weights.cpu().numpy()
        #
        gbs = gene_guided_bp(self.model, testloader_vis)
        cam_hms, cam_hois, grad_cams, grad_cam_hms, grad_cam_hois = gene_grad_cam(self.model, testloader_vis, weights)

        for i, gb in enumerate(gbs):
            gc = grad_cams[i]
            ggc = np.multiply(np.stack([gc, gc, gc]), gb)
            ggc = ggc - ggc.min()
            ggc /= ggc.max()
            ggc = np.transpose(ggc, (1, 2, 0))
            ggc = ggc * 255
            ggc = ggc.astype(np.uint8)

              #     gb = gb - gb.min()
            gb /= gb.max()
            gb = np.transpose(gb, (1, 2, 0))
            gb = gb * 255
            gb = gb.astype(np.uint8)
        #
            if testset.x[i] in test_true_positives:
                cam_hms[i].save(os.path.join(ghf.camHeatmapTPFolder, testset.x[i].split('/')[-1].replace('.jpg', '.png')))
                cam_hois[i].save(os.path.join(ghf.camHeatmapOnImageTPFolder, testset.x[i].split('/')[-1].replace('.jpg', '.png')))
                # cv2.imwrite(os.path.join(ghf.gradCamGuidedGradCamTPFolder, testset.x[i].split('/')[-1]), ggc[:, :, ::-1])
                # cv2.imwrite(os.path.join(ghf.guidedBPTPFolder, testset.x[i].split('/')[-1]), gb[:, :, ::-1])
                grad_cam_hms[i].save(os.path.join(ghf.gradCamHeatmapTPFolder, testset.x[i].split('/')[-1].replace('.jpg', '.png')))
                grad_cam_hois[i].save(os.path.join(ghf.gradCamHeatmapOnImageTPFolder, testset.x[i].split('/')[-1].replace('.jpg', '.png')))
            elif testset.x[i] in test_true_negatives:
                cam_hms[i].save(os.path.join(ghf.camHeatmapTNFolder, testset.x[i].split('/')[-1].replace('.jpg', '.png')))
                cam_hois[i].save(os.path.join(ghf.camHeatmapOnImageTNFolder, testset.x[i].split('/')[-1].replace('.jpg', '.png')))
                # cv2.imwrite(os.path.join(ghf.gradCamGuidedGradCamTNFolder, testset.x[i].split('/')[-1]), ggc[:, :, ::-1])
                # cv2.imwrite(os.path.join(ghf.guidedBPTNFolder, testset.x[i].split('/')[-1]), gb[:, :, ::-1])
                grad_cam_hms[i].save(os.path.join(ghf.gradCamHeatmapTNFolder, testset.x[i].split('/')[-1].replace('.jpg', '.png')))
                grad_cam_hois[i].save(os.path.join(ghf.gradCamHeatmapOnImageTNFolder, testset.x[i].split('/')[-1].replace('.jpg', '.png')))
            elif testset.x[i] in test_missing_report:
                cam_hms[i].save(os.path.join(ghf.camHeatmapMissFolder, testset.x[i].split('/')[-1].replace('.jpg', '.png')))
                cam_hois[i].save(os.path.join(ghf.camHeatmapOnImageMissFolder, testset.x[i].split('/')[-1].replace('.jpg', '.png')))
                # cv2.imwrite(os.path.join(ghf.gradCamGuidedGradCamMissFolder, testset.x[i].split('/')[-1]), ggc[:, :, ::-1])
                # cv2.imwrite(os.path.join(ghf.guidedBPMissFolder, testset.x[i].split('/')[-1]), gb[:, :, ::-1])
                grad_cam_hms[i].save(os.path.join(ghf.gradCamHeatmapMissFolder, testset.x[i].split('/')[-1].replace('.jpg', '.png')))
                grad_cam_hois[i].save(os.path.join(ghf.gradCamHeatmapOnImageMissFolder, testset.x[i].split('/')[-1].replace('.jpg', '.png')))
            elif testset.x[i] in test_mistake_report:
                cam_hms[i].save(os.path.join(ghf.camHeatmapMistFolder, testset.x[i].split('/')[-1].replace('.jpg', '.png')))
                cam_hois[i].save(os.path.join(ghf.camHeatmapOnImageMistFolder, testset.x[i].split('/')[-1].replace('.jpg', '.png')))
                # cv2.imwrite(os.path.join(ghf.gradCamGuidedGradCamMistFolder, testset.x[i].split('/')[-1]), ggc[:, :, ::-1])
                # cv2.imwrite(os.path.join(ghf.guidedBPMistFolder, testset.x[i].split('/')[-1]), gb[:, :, ::-1])
                grad_cam_hms[i].save(os.path.join(ghf.gradCamHeatmapMistFolder, testset.x[i].split('/')[-1].replace('.jpg', '.png')))
                grad_cam_hois[i].save(os.path.join(ghf.gradCamHeatmapOnImageMistFolder, testset.x[i].split('/')[-1].replace('.jpg', '.png')))
            else:
                 raise RuntimeError("image name is wrong!")
            '''


def main(config):
    predict = Predict(config)
    print('start test')
    predict.test()


if __name__ == '__main__':
    main(args.config)
