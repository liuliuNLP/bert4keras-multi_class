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

模型参数: batch_size = 32, maxlen = 300, epoch=50

使用bert预训练模型，评估结果如下:

```
	  precision	   recall	  f1-score	 support
汽车	0.987025948	0.990981964	0.989	        998
体育	0.994877049	0.974899598	0.984787018	996
健康	0.991596639	0.946840522	0.968701898	997
教育	0.927289896	0.984954865	0.955252918	997
军事	0.972918756	0.971943888	0.972431078	998

accuracy	0.973926996	0.973926996	0.973926996	0.973926996
macro avg	0.974741658	0.973924167	0.974034583	4986
weighted avg	0.974739717	0.973926996	0.974035106	4986

```



- electra_180g_small

模型参数: batch_size = 32, maxlen = 300, epoch=50

使用electra_180g_small预训练模型，评估结果如下:

```
  	  precision	   recall	  f1-score	support
体育	0.995967742	0.991967871	0.993963783	996
健康	0.951124145	0.975927783	0.963366337	997
教育	0.981012658	0.932798395	0.956298201	997
汽车	0.953623188	0.988977956	0.970978849	998
军事	0.99291498	0.982965932	0.987915408	998
accuracy	0.97452868	0.97452868	0.97452868	0.97452868
macro avg	0.974928543	0.974527588	0.974504515	4986
weighted avg	0.974923657	0.97452868	0.974502595	4986

```



- albert_small_zh_google

模型参数: batch_size = 32, maxlen = 300, epoch=50

使用albert_small_zh_google预训练模型，评估结果如下:

```
	  precision	   recall	  f1-score	 support
军事	0.993	        0.99498998	0.993993994	998
教育	0.980632008	0.964894684	0.972699697	997
健康	0.977159881	0.986960883	0.982035928	997
汽车	0.986055777	0.991983968	0.989010989	998
体育	0.996981891	0.99497992	0.995979899	996
accuracy	0.986762936	0.986762936	0.986762936	0.986762936
macro avg	0.986765911	0.986761887	0.986744101	4986
weighted avg	0.98676497	0.986762936	0.986744158	4986


```



### 致谢

苏神：https://github.com/bojone/bert4keras
