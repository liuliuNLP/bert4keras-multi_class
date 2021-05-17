本项目是基于苏神(https://github.com/bojone/bert4keras) 的 bert4keras 框架下进行开发的多标签分类任务。

本项目是通过bert、albert、electra三个预训练模型对下游任务进行fine-tune来实现文本多标签分类任务。

### 维护者

- 溜溜NLP

### 数据集

[全网新闻分类数据(SogouCA)](http://www.sogou.com/labs/resource/ca.php)

本次实验采用的sougou小分类数据集，共有5个类别，分别为体育、健康、军事、教育、汽车。

下面链接帮我们整理了各种任务的数据集，有需要的可以自行下载。

各种任务的数据集下载：https://github.com/CLUEbenchmark/CLUEDatasetSearch

### 模型下载

本项目只使用了三种预训练模型，各位如果想使用其他模型跑效果，可自行下载。

bert: https://github.com/google-research/bert

electra_180g_small: https://github.com/ymcui/Chinese-ELECTRA

albert_small_zh_google: https://github.com/brightmart/albert_zh

### 模型效果

- bert

模型参数: batch_size = 64, maxlen = 128, epoch=50

使用bert预训练模型，评估结果如下:

```
   micro avg     0.9488    0.8606    0.9025      1657
   macro avg     0.9446    0.8084    0.8589      1657
weighted avg     0.9460    0.8606    0.8955      1657
 samples avg     0.8932    0.8795    0.8799      1657

accuracy:  0.828437917222964
hamming loss:  0.0031631919482386773
```



- electra_180g_small

模型参数: batch_size = 64, maxlen = 128, epoch=50

使用electra_180g_small预训练模型，评估结果如下:

```
   micro avg     0.9471    0.9294    0.9382      1657
   macro avg     0.9416    0.9105    0.9208      1657
weighted avg     0.9477    0.9294    0.9362      1657
 samples avg     0.9436    0.9431    0.9379      1657

accuracy:  0.8931909212283045
hamming loss:  0.0020848310567936736
```



- albert_small_zh_google

模型参数: batch_size = 64, maxlen = 128, epoch=50

使用albert_small_zh_google预训练模型，评估结果如下:

```
   micro avg     0.9471    0.9294    0.9382      1657
   macro avg     0.9416    0.9105    0.9208      1657
weighted avg     0.9477    0.9294    0.9362      1657
 samples avg     0.9436    0.9431    0.9379      1657

accuracy:  0.8931909212283045
hamming loss:  0.0020848310567936736
```

