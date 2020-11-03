### auto train

python run.py -c config3.json

#### Preparing the training dataset
`python data/split_data.py -i data/data_list/case_file -o data/data_list/multi_data_list -svl True(split using every class，default value is True)`

#### Usage

- Configuration file like `config/config.json`
- root of images `data_path`
- Project name `name`
- Training `python run.py -c config/{your config file}.config`
- Testing `python test.py -c config/{your config file}.config`
- Evaluating `python inference.py -c config/{your config file}.config -t target.json` `-t`optional parameter,target.json {image_name: target},ignoring this if you dont have ground truth

### config

```
name: project name
balanced: balanced sampling（1:1）
num_samples： sampling number of every epoch
type: eye images data type（'fundus', 'slitlamp'）
gpu_ids: GPU id（multiple gpu like '0, 1'）
sclera: if masking slitlamp image eyelids（if so, providing masking images path）
model: {
    name: model name(resnet50, resnet101, inception_v3, vgg16_bn, vgg19_bn)
}
test: {
    checkpoint: model path during testing
}
inference: {
    checkpoint: model path during inference
    data_path: data path
}
```

#### result

- save in `results/project_name`
- `checkpoints` model saving path
- `logs`log files saving path
- `val_result.json` best result on valid dataset
- `test`testing saving path
- `inference`inference result the activation mapping fig path
