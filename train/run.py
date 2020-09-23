from project import Project
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='config/config.json', help='config file')
args = parser.parse_args()


def main(config):
    project = Project(config)
    # print('split data')
    # project.split_data()
    print('start training')
    project.train()
    print('start testing')
    project.test()

    print('train finished, start plotting log')
    project.plot_log()


if __name__ == '__main__':
    main(args.config)
