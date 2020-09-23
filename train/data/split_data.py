import os
from sklearn.model_selection import train_test_split

import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, default='data_list/raw_list', help='input list')
parser.add_argument('--output', '-o', type=str, default='', help='output folder')
parser.add_argument('--split_via_label', '-svl', type=str2bool, default=True, help='split via per label')
args = parser.parse_args()


def split_data(input_file, output, split_via_label):
    if not os.path.exists(input_file):
        raise ValueError('file {} not exist'.format(input_file))
    os.makedirs(output, exist_ok=True)
    case_list = open(input_file).readlines()

    if split_via_label:
        positive = []
        negtive = []
        for c in case_list:
            _case = c.split(', ')[0]
            labels = c.strip().replace(_case+', ', '')
            if labels != '0':
                positive.append(_case+', 1')
            else:
                negtive.append(_case+', 0')
        train_pos, val_pos = train_test_split(positive, test_size=0.4, random_state=42)
        val_pos, test_pos = train_test_split(val_pos, test_size=0.5, random_state=42)
        train_neg, val_neg = train_test_split(negtive, test_size=0.4, random_state=42)
        val_neg, test_neg = train_test_split(val_neg, test_size=0.5, random_state=42)

        train = train_pos + train_neg
        val = val_pos + val_neg
        test = test_pos + test_neg

    else:
        case_list_fusion_label = []
        for c in case_list:
            _case = c.split(', ')[0]
            labels = c.strip().replace(_case+', ', '')
            if labels != '0':
                case_list_fusion_label.append(_case+', 1')
            else:
                case_list_fusion_label.append(_case+', 0')
        train, val = train_test_split(case_list_fusion_label, test_size=0.4, random_state=42)
        val, test = train_test_split(val, test_size=0.5, random_state=42)

    with open(os.path.join(output, 'train_list'), 'w') as f:
        for i in train:
            f.write(i+'\n')

    with open(os.path.join(output, 'val_list'), 'w') as f:
        for i in val:
            f.write(i+'\n')

    with open(os.path.join(output, 'test_list'), 'w') as f:
        for i in test:
            f.write(i+'\n')


if __name__ == '__main__':
    split_data(args.input, args.output, args.split_via_label)
