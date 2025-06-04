本项目继承自李沐衡同学的IDS项目https://github.com/MooreFoss/IDS

李沐衡同学做出了一些基于机器学习的入侵检测尝试，这里我做出了以下的一些尝试性分析与优化，力求提升模型性能

## 0 问题分析

### 0.1 入侵检测

- 基础语义信息：编码形式较简单，易解决
- 时序信息：网络入侵本质上是时间序列事件，时序特征是其重要的特征信息

综上，本次优化希望着重解决模型对时序信息的感知能力

### 0.2 原有方法分析

原方法基于CNN、LSTM实现

- 优点：
  - 时序特征：有一定时序特征提取能力
  - 特征提取能力：CNN擅长局部特征提取
- 局限性：
  - 时序特征：LSTM有限记忆长度，长序列信息易丢失
  - 训练、推理效率：受LSTM限制，只能顺序计算，无法并行处理，导致训练、推理效率低

综上可以看出，原方法虽然具备较好的局部特征提取能力，但其对时序信息的理解能力不足



因此，为优化模型，笔者尝试利用基于Transformer的方法，构建具有强大时序信息感知能力的模型



## 1 Get Start

测试时使用的python版本是3.9

### 1.1 安装与cuda版本适配的pytorch

### 1.2 其他软件环境

```
pip install -r requirements.txt
```

### 1.3 数据配置

#### 1.3.1 开源数据

使用开源数据集UNSW_NB15，直接在[官网](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15?select=UNSW_NB15_testing-set.csv)下载并解压到自定义路径并使用即可

#### 1.3.2 实采数据

数据采集方式见[get_data.md](https://github.com/bhsh0112/IDS/blob/main/get_data.md)

### 1.4 代码运行

#### 训练

```
python transformerBase_train.py
```

可以在代码549~550行修改数据输入路径

#### 验证

```
python transformerBase_eval.py
```

可以在代码第195、196行分别修改数据输入路径和权重输入路径

