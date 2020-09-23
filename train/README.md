### auto train

python run.py -c config3.json

#### 准备数据集
`python data/split_data.py -i data/data_list/case_file -o data/data_list/multi_data_list -svl True(是否根据每一类单独划分，默认为True)`

#### 使用

- 参考`config/config.json`，创建一份项目配置
- 修改配置文件中`data_path`为图片所在目录
- 修改配置文件中`name`为当前项目名称
- 训练 `python run.py -c config/{创建的配置文件}.config`
- 测试 `python test.py -c config/{创建的配置文件}.config`
- 推理 `python inference.py -c config/{创建的配置文件}.config -t target.json` `-t`参数可选，target.json为{image_name: target}，如果没有ground truth可以不输入

### config

```
name: 项目名称
balanced: 是否均衡采样（按1:1采样）
num_samples： 每个epoch采样数量
type: 数据类型（'fundus', 'slitlamp'）
gpu_ids: 使用GPU id号（使用多个为'0, 1'）
sclera: 是否提取巩膜（如需提取输入巩膜图片对应地址）
model: {
    name: 模型名称(resnet50, resnet101, inception_v3, vgg16_bn, vgg19_bn)
}
test: {
    checkpoint: 测试时使用的模型
}
inference: {
    checkpoint: 推理时使用的模型
    data_path: 需要推理的图片路径（）
}
```

#### 结果

- 结果保存在 `results/project_name`
- `checkpoints`目录下保存模型文件
- `logs`目录下保存训练日志文件
- `val_result.json` 为验证集上最好结果的总结
- `test`目录保存测试结果
- `inference`目录保存推理结果和类激活图
