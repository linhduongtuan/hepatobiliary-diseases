import torch.utils.data as data
import os
import cv2


class LiverSingleDataset(data.Dataset):
    def __init__(self,
                 transform=None,
                 _type=None,
                 sclera='',
                 root='data/slitlamp',
                 data_file='../../data/im_list/train_list'
                 ):
        self.transform = transform
        self.x = []
        self.y = []
        if sclera and _type != 'slitlamp':
            raise ValueError("sclera must with slitlamp, but fund _type: {}".format(_type))
        self.sclera = sclera
        if sclera:
            self.scleras = os.listdir(sclera)
        self.load_data_list(_type, root, data_file)

    def load_data_list(self, _type, root, data_file):
        if _type not in ['slitlamp', 'fundus']:
            raise ValueError("_type need be slitlamp or fundus, but fund: {}".format(_type))

        if not os.path.exists(root):
            raise ValueError("root '{}' dose not exist".format(root))

        if not os.path.exists(data_file):
            raise ValueError("data_file '{}' dose not exist".format(data_file))

        case2file = {}
        for f in os.listdir(root):
            c = f.split('_')[1]
            if c not in case2file:
                case2file[c] = []
            case2file[c].append(os.path.join(root, f))
        with open(data_file) as f:
            for line in f.readlines():
                lines = line.strip().split(', ')
                if lines[0] in case2file:
                    for i in case2file[lines[0]]:
                        self.x.append(i)
                        self.y.append(int(lines[1]))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        filepath = self.x[item]
        label = self.y[item]

        # image = Image.open(filepath)
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.sclera:
            filename = filepath.split('/')[-1]
            sclera_file = filename.replace('jpg', 'png').replace('JPG', 'png')
            if sclera_file not in self.scleras:
                raise ValueError('{} not in {}'.format(sclera_file, self.sclera))
            sclera = cv2.imread(os.path.join(self.sclera, sclera_file), 0)
            sclera = cv2.resize(sclera, (image.shape[1], image.shape[0]))

            _, mask = cv2.threshold(sclera, 10, 255, cv2.THRESH_BINARY)
            image = cv2.bitwise_and(image, image, mask=mask)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class LiverDataset(data.Dataset):
    def __init__(self,
                 transform=None,
                 _type=None,
                 sclera=False,
                 root='/datastore/users/yang.zhou/multi_data/',
                 data_file='../../data/im_list/train_list'
                 ):
        self.transform = transform
        self.x = []
        self.y = []
        if sclera and _type != 'slitlamp':
            raise ValueError("sclera must with slitlamp, but fund _type: {}".format(_type))
        self.sclera = sclera
        if sclera:
            self.scleras = os.listdir(sclera)
        self.load_data_list(_type, root, data_file)

    def load_data_list(self, _type, root, data_file):
        if _type not in ['slitlamp', 'fundus']:
            raise ValueError("_type need be slitlamp or fundus, but fund: {}".format(_type))
        with open(data_file) as f:
            for line in f.readlines():
                lines = line.strip().split(', ')
                if os.path.exists(root+'{}_{}_OS.jpg'.format(_type, lines[0])):
                    self.x.append(root+'{}_{}_OS.jpg'.format(_type, lines[0]))
                    self.y.append(int(lines[1]))
                if os.path.exists(root+'{}_{}_OD.jpg'.format(_type, lines[0])):
                    self.x.append(root+'{}_{}_OD.jpg'.format(_type, lines[0]))
                    self.y.append(int(lines[1]))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        filepath = self.x[item]
        label = self.y[item]

        # image = Image.open(filepath)
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.sclera:
            filename = filepath.split('/')[-1]
            sclera_file = filename.replace('jpg', 'png')
            if sclera_file not in self.scleras:
                raise ValueError('{} not in {}'.format(sclera_file, self.sclera))
            sclera = cv2.imread(os.path.join(self.sclera, sclera_file), 0)
            sclera = cv2.resize(sclera, (image.shape[1], image.shape[0]))

            _, mask = cv2.threshold(sclera, 10, 255, cv2.THRESH_BINARY)
            image = cv2.bitwise_and(image, image, mask=mask)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class LiverInferenceDataset(data.Dataset):
    def __init__(self,
                 transform=None,
                 sclera=None,
                 data_path='./inference',
                 ):
        self.transform = transform
        self.data_path = data_path
        self.x = os.listdir(data_path)
        self.x = [os.path.join(data_path, x) for x in self.x]
        self.sclera = sclera
        if sclera:
            self.scleras = os.listdir(sclera)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        filepath = self.x[item]

        # image = Image.open(filepath)
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        filename = filepath.split('/')[-1]
        if self.sclera:
            sclera_file = filename.replace('jpg', 'png')
            if sclera_file not in self.scleras:
                raise ValueError('{} not in {}'.format(sclera_file, self.sclera))
            sclera = cv2.imread(os.path.join(self.sclera, sclera_file), 0)
            sclera = cv2.resize(sclera, (image.shape[1], image.shape[0]))

            _, mask = cv2.threshold(sclera, 10, 255, cv2.THRESH_BINARY)
            image = cv2.bitwise_and(image, image, mask=mask)

        if self.transform is not None:
            image = self.transform(image)

        return image, filename
