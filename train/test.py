from project.project import Project
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='config/config.json', help='config file')
args = parser.parse_args()


def main(config):
    project = Project(config)
    print('start test')
    project.test()


if __name__ == '__main__':
    main(args.config)
