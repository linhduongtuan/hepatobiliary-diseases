import json
import os
import numpy as np
import torch
import time
#import timm
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import torch.utils.data as _data
from torch.utils.data.sampler import WeightedRandomSampler

from data import LiverSingleDataset, LiverInferenceDataset
from model import resnet50, resnet101, inception_v3, vgg16, vgg19_bn
from tools import train, inference, evalute
from utils import plot_log


class Project(object):
    """

    """

    def __init__(self, config):
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
        #elif self.config['model']['name'] == 'efficientnet_b0':
            #self.model = timm.create_model('tf_efficientnet_b0', pretrained=True)
            #in_features = self.model.classifier.in_features
            #self.model.classifier = nn.Linear(in_features, self.config['model']['num_classes'])

        self.model.cuda()
        if len(self.config['gpu_ids']) > 1:
            self.model = nn.DataParallel(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['model']['init_lr'],momentum=0.9)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['model']['init_lr'],betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
                                   
        self.exp_lr_schedler = lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['model']['milestones'],
                                                        gamma=0.2)

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
        elif self.config['model']['name'] == 'efficientnet_b0':
            self.model = timm.create_model('tf_efficientnet_b0', pretrained=True)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, self.config['model']['num_classes'])

        if _type == 'test':
            checkpoints = os.path.join('./results', self.config['name'], 'checkpoints',
                                       self.config['test']['checkpoint'])
        else:
            checkpoints = os.path.join('./results', self.config['name'], 'checkpoints',
                                       self.config['inference']['checkpoint'])
        self.model.load_state_dict(torch.load(checkpoints)['state_dict'])
        self.model.cuda()
        if len(self.config['gpu_ids']) > 1:
            self.model = nn.DataParallel(self.model)

    @staticmethod
    def get_weight(dataset):
        label_num = {}
        for label in dataset.y:
            if label not in label_num:
                label_num[label] = 0
            label_num[label] += 1
        label_weight = {}
        for k, v in label_num.items():
            label_weight[k] = dataset.__len__() / v
        return label_weight

    def train(self):
        os.makedirs(os.path.join('./results', self.config['name'], 'logs'), exist_ok=True)
        os.makedirs(os.path.join('./results', self.config['name'], 'checkpoints'), exist_ok=True)
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config['image']['height'], self.config['image']['width'])),
            transforms.CenterCrop((self.config['image']['height'], self.config['image']['width'])),
            transforms.RandomHorizontalFlip(0.5),
                 #transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                 #transforms.ColorJitter(hue=.05, saturation=.05),
                 #transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config['image']['height'], self.config['image']['width'])),
            transforms.CenterCrop((self.config['image']['height'], self.config['image']['width'])),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        result = {
            'best_via_auc': {},
            'best_via_f1': {}
        }
        trainset = LiverSingleDataset(transform=train_transform, root=self.config['data_path'],
                                      _type=self.config['type'], data_file=self.config['train_list'])
        if self.config['balanced']:
            print('balancing data...')
            weighted = self.get_weight(trainset)
            train_weights = [weighted[label] for _, label in trainset]
            train_sampler = WeightedRandomSampler(train_weights,
                                                  num_samples=self.config['num_samples'],
                                                  replacement=True)
            trainloader = _data.DataLoader(trainset, sampler=train_sampler,
                                           batch_size=self.config['model']['batch_size'],
                                           num_workers=self.config['model']['num_workers'])
        else:
            trainloader = _data.DataLoader(trainset, shuffle=True,
                                           batch_size=self.config['model']['batch_size'],
                                           num_workers=self.config['model']['num_workers'])

        valset = LiverSingleDataset(transform=test_transform, root=self.config['data_path'],
                                    _type=self.config['type'], data_file=self.config['val_list'])
        valloader = _data.DataLoader(valset, shuffle=False,
                                     batch_size=self.config['model']['batch_size'],
                                     num_workers=self.config['model']['num_workers'])

        train_log = os.path.join('./results', self.config['name'], 'logs',
                                 'train_{}_log'.format(self.config['model']['name']))
        val_log = os.path.join('./results', self.config['name'], 'logs',
                               'val_{}_log'.format(self.config['model']['name']))
        saved_model = os.path.join('./results', self.config['name'], 'checkpoints', self.config['model']['name'])
        best_via_auc, best_via_f1 = train(self.model, [trainloader, valloader],
                                          self.criterion, self.optimizer, self.exp_lr_schedler,
                                          config=self.config, train_log=train_log,
                                          val_log=val_log, saved_model=saved_model)
        result['best_via_auc'] = best_via_auc
        result['best_via_f1'] = best_via_f1
        with open(os.path.join('./results', self.config['name'], 'val_results.json'), 'w') as f:
            json.dump(result, f, indent=4)

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
        testset = LiverSingleDataset(transform=test_transform, root=self.config['data_path'],
                                     _type=self.config['type'], data_file=self.config['val_list'])
        testloader = _data.DataLoader(testset, shuffle=False,
                                      batch_size=self.config['model']['batch_size'],
                                      num_workers=self.config['model']['num_workers'])

        title = self.config['name'] + '_' + self.config['model']['name']
        name = os.path.join('./results', self.config['name'], 'test',
                            self.config['test']['checkpoint'].replace('.pkl', ''))
        test_acc, test_auc, test_sens, test_spec, test_f1, test_cm = evalute(0, 0, self.model,
                                                                             testloader, self.criterion,
                                                                             _type='Test', draw_roc=True, title=title,
                                                                             name=name)
        test_result = {
            'acc': round(test_acc, 4),
            'auc': round(test_auc, 4),
            'sensitivity': round(test_sens, 4),
            'specificity': round(test_spec, 4),
            'f1': round(test_f1, 4),
            'confusion matrix': '[[{}, {}], [{}, {}]]'.format(test_cm[0, 0], test_cm[0, 1],
                                                              test_cm[1, 0], test_cm[1, 1])
        }

        with open(os.path.join('./results', self.config['name'], 'test', 'test_results.json'), 'w') as f:
            json.dump(test_result, f, indent=4)

    def inference(self, target_file):
        self.load_state()
        os.makedirs(os.path.join('./results', self.config['name'], 'inference'), exist_ok=True)
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config['image']['height'], self.config['image']['width'])),
            transforms.CenterCrop((self.config['image']['height'], self.config['image']['width'])),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        testset = LiverInferenceDataset(transform=test_transform, data_path=self.config['inference']['data_path'],
                                        sclera=self.config['sclera'])
        testloader = _data.DataLoader(testset, shuffle=False,
                                      batch_size=self.config['model']['batch_size'],
                                      num_workers=self.config['model']['num_workers'])

        if len(self.config['gpu_ids']) > 1:
            weights = self.model.module.fc.weight.data
        else:
            weights = self.model.fc.weight.data
        weights = weights.cpu().numpy()
        weights = weights[:, :, np.newaxis, np.newaxis]

        gt = []
        if target_file != '':
            gt = json.load(open(target_file))

        preds, scores = inference(self.model, testloader,
                                  os.path.join('./results', self.config['name'], 'inference'), weights, gt)
        result = {}
        for i, name in enumerate(testset.x):
            name = name.split('/')[-1]
            result[name] = '({:.4f}, {:.4f})'.format(scores[i][0], scores[i][1])

        result_name = os.path.join('./results', self.config['name'],
                                   'inference/{}_{}.json'.format(self.config['name'],
                                                                 self.config['inference']['data_path'].split('/')[-1]))
        with open(result_name, 'w') as f:
            json.dump(result, f, indent=4)

    def plot_log(self):
        train_log = os.path.join('./results', self.config['name'], 'logs',
                                 'train_{}_log'.format(self.config['model']['name']))
        val_log = os.path.join('./results', self.config['name'], 'logs',
                               'val_{}_log'.format(self.config['model']['name']))
        title = self.config['name'] + '_' + self.config['model']['name']
        name = os.path.join('./results', self.config['name'], 'logs', self.config['model']['name'])
        plot_log(train_log, val_log, title, name)
