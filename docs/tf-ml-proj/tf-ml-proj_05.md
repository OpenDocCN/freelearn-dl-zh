# 第五章：语音转文本与主题提取使用自然语言处理

由于语音数据的复杂性和多样性，识别和理解口语语言是一个具有挑战性的问题。过去已经部署了几种不同的技术来识别口语单词。大多数方法的适用范围非常有限，因为它们无法识别各种单词、口音和语调，以及口语语言的某些方面，如单词之间的停顿。一些常见的语音识别建模技术包括**隐马尔可夫模型**（**HMM**）、**动态时间规整**（**DTW**）、**长短期记忆网络**（**LSTM**）和**连接时序分类**（**CTC**）。

本章将介绍语音转文本的各种选项，以及 Google 的 TensorFlow 团队使用语音命令数据集的预构建模型。我们将讨论以下主题：

+   语音转文本框架和工具包

+   Google 语音命令数据集

+   基于卷积神经网络的语音识别架构

+   一个 TensorFlow 语音命令示例

下载并遵循本章的代码：[`github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/)。

# 语音转文本框架和工具包

许多基于云的 AI 提供商提供语音转文本作为服务：

+   亚马逊提供的语音识别服务被称为**Amazon Transcribe**。Amazon Transcribe 支持将存储在 Amazon S3 中的音频文件转录为四种不同格式：`.flac`、`.wav`、`.mp4` 和 `.mp3`。它允许最大长度为两小时，最大大小为 1 GB 的音频文件。转录结果以 JSON 文件格式保存在 Amazon S3 存储桶中。

+   Google 将语音转文本作为其 Google Cloud ML 服务的一部分提供。Google Cloud Speech to Text 支持 `FLAC`、`Linear16`、`MULAW`、`AMR`、`AMR_WB` 和 `OGG_OPUS` 文件格式。

+   微软在其 Azure 认知服务平台中提供语音转文本 API，称为语音服务 SDK。语音服务 SDK 与其他 Microsoft API 集成，用于转录录制的音频。它仅支持单声道 WAV 或 PCM 文件格式，且采样率为 6 kHz。

+   IBM 在其 Watson 平台中提供语音转文本 API。Watson Speech to Text 支持八种音频格式：BASIC、FLAC、L16、MP3、MULAW、OGG、WAV 和 WEBM。音频文件的最大大小和时长根据使用的格式而异。转录结果以 JSON 文件返回。

除了对各种国际口语语言和广泛的全球词汇表的支持外，这些云服务还在不同程度上支持以下功能：

+   **多通道识别**：识别多个通道中记录的多个参与者

+   **说话者分离**：预测特定说话者的语音

+   **自定义模型和模型选择**：插入您自己的模型并从大量预构建模型中选择

+   不当内容过滤和噪声过滤

还有许多用于语音识别的开源工具包，如 Kaldi。

Kaldi (http:/kaldi-asr.org) 是一个流行的开源语音转文本识别库。它是用 C++编写的，且可以从[`github.com/kaldi-asr/kaldi`](https://github.com/kaldi-asr/kaldi)获取。Kaldi 可以通过其 C++ API 集成到您的应用程序中，也支持使用 NDK、clang++和 OpenBLAS 在 Android 上运行。

# Google 语音命令数据集

Google 语音命令数据集由 TensorFlow 和 AIY 团队创建，旨在展示使用 TensorFlow API 的语音识别示例。该数据集包含 65,000 个一秒钟长的音频片段，每个片段包含由成千上万的不同发音者所说的 30 个不同单词之一。

Google 语音命令数据集可以从以下链接下载：[`download.tensorflow.org/data/speech_commands_v0.02.tar.gz`](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)。

这些音频片段是在真实环境中使用手机和笔记本电脑录制的。35 个单词包含噪声词，另外 10 个命令词是机器人环境中最有用的，列举如下：

+   是的

+   否

+   向上

+   向下

+   向左

+   向右

+   向前

+   关闭

+   停止

+   继续

关于如何准备语音数据集的更多细节，请参见以下链接：

+   [`arxiv.org/pdf/1804.03209.pdf`](https://arxiv.org/pdf/1804.03209.pdf)

+   [`ai.googleblog.com/2017/08/launching-speech-commands-dataset.html`](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html)

使用这个数据集，因此，本章中的示例问题被称为关键字检测任务。

# 神经网络架构

该示例使用的网络包含三个模块：

+   一个特征提取模块，将音频片段处理为特征向量

+   一个深度神经网络模块，为输入特征向量帧中的每个单词生成软最大概率

+   一个后验处理模块，将帧级后验分数合并成每个关键字的单一分数

# 特征提取模块

为了简化计算，传入的音频信号通过语音活动检测系统，信号被划分为语音部分和非语音部分。语音活动检测器使用一个 30 分量对角协方差 GMM 模型。该模型的输入是 13 维 PLP 特征、其增量和二阶增量。GMM 的输出传递给一个状态机进行时间平滑处理。

该 GMM-SM 模块的输出是信号中的语音部分和非语音部分。

信号中的语音部分进一步处理以生成特征。声学特征基于 40 维对数滤波器组能量计算，每 10 毫秒计算一次，窗口为 25 毫秒。信号中还加入了 10 个未来帧和 30 个过去帧。

关于特征提取器的更多细节可以从原始论文中获得，相关链接在进一步阅读部分提供。

# 深度神经网络模块

DNN 模块是使用卷积神经网络（CNN）架构实现的。代码实现了多种不同变体的 ConvNet，每个变体产生不同的准确度，并且训练所需的时间不同。

构建模型的代码提供在 `models.py` 文件中。它允许根据命令行传递的参数创建四种不同的模型：

+   `single_fc`：该模型仅有一个全连接层。

+   `conv`：该模型是一个完整的 CNN 架构，包含两对卷积层和最大池化层，后跟一个全连接层。

+   `low_latency_conv`：该模型有一个卷积层，后面跟着三个全连接层。顾名思义，与 `conv` 架构相比，它的参数和计算量较少。

+   `low_latency_svdf`：该模型遵循论文《*压缩深度神经网络*》中的架构和层。

    *使用秩约束拓扑的网络* 可从 [`research.google.com/pubs/archive/43813.pdf`](https://research.google.com/pubs/archive/43813.pdf) 获得。

+   `tiny_conv`：该模型只有一个卷积层和一个全连接层。

如果命令行未传递架构，则默认架构为 `conv`。在我们的运行中，架构在使用默认准确率和默认步数 18,000 训练模型时，显示了以下训练、验证和测试集的准确率：

| 架构 | 准确率（%） |
| --- | --- |
| 训练集 | 验证集 | 测试集 |
| `conv`（默认） | 90 | 88.5 | 87.7 |
| `single_fc` | 50 | 48.5 | 48.2 |
| `low_latenxy_conv` | 22 | 21.6 | 23.6 |
| `low_latency_svdf` | 7 | 8.9 | 8.6 |
| `tiny_conv` | 55 | 65.7 | 65.4 |

由于网络架构使用的是更适合图像数据的 CNN 层，因此语音文件通过将短时间段的音频信号转换为频率强度的向量，转化为单通道图像。

从前面的观察可以看出，缩短的架构在相同超参数下给出的准确率较低，但运行速度更快。因此，可以运行更多的迭代轮次，或者可以增加学习率以获得更高的准确率。

现在让我们来看一下如何训练和使用这个模型。

# 训练模型

1.  移动到从仓库克隆代码的文件夹，并使用以下命令训练模型：

```py
python tensorflow/examples/speech_commands/train.py
```

你将开始看到训练的输出，如下所示：

```py
I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:897] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties:
name: Quadro P5000 major: 6 minor: 1 memoryClockRate(GHz): 1.506
pciBusID: 0000:01:00.0
totalMemory: 15.90GiB freeMemory: 14.63GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0
I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0: N
I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14168 MB memory) -> physical GPU (device: 0, name: Quadro P5000, pci bus id: 0000:01:00.0, compute capability: 6.1)
```

1.  一旦训练迭代开始，代码会打印出学习率、训练集的准确率以及交叉熵损失，如下所示：

```py
INFO:tensorflow:Training from step: 1
INFO:tensorflow:Step #1: rate 0.001000, accuracy 12.0%, cross entropy 2.662751
INFO:tensorflow:Step #2: rate 0.001000, accuracy 6.0%, cross entropy 2.572391
INFO:tensorflow:Step #3: rate 0.001000, accuracy 11.0%, cross entropy 2.547692
INFO:tensorflow:Step #4: rate 0.001000, accuracy 8.0%, cross entropy 2.615582
INFO:tensorflow:Step #5: rate 0.001000, accuracy 5.0%, cross entropy 2.592372
```

1.  代码还会每 100 步保存一次模型，因此如果训练中断，可以从最近的检查点重新开始：

```py
INFO:tensorflow:Saving to "/tmp/speech_commands_train/conv.ckpt-100"
```

训练运行了几个小时，共 18,000 步，最终会打印出最终的训练学习率、准确度、损失和混淆矩阵，如下所示：

```py
INFO:tensorflow:Step #18000: rate 0.000100, accuracy 90.0%, cross entropy 0.420554
INFO:tensorflow:Confusion Matrix:
 [[368 2 0 0 1 0 0 0 0 0 0 0]
 [ 3 252 9 6 13 15 13 18 17 1 13 11]
 [ 0 1 370 12 2 2 7 2 0 0 0 1]
 [ 3 8 4 351 8 7 6 0 0 0 3 16]
 [ 3 4 0 0 324 1 3 0 1 5 7 2]
 [ 4 3 4 19 1 330 3 0 0 1 3 9]
 [ 2 2 12 2 4 0 321 7 0 0 2 0]
 [ 3 7 1 1 2 0 4 344 1 0 0 0]
 [ 5 10 0 0 9 1 1 0 334 3 0 0]
 [ 4 2 1 0 33 0 2 2 7 317 4 1]
 [ 5 2 0 0 15 0 1 1 0 2 323 1]
 [ 4 17 0 33 2 8 0 1 0 2 3 302]]
```

从输出中可以观察到，尽管代码一开始的学习率为 0.001，但它在训练结束时将学习率降低到 0.001。由于有 12 个命令词，它还会输出一个 12 x 12 的混淆矩阵。

代码还会打印验证集的准确度和混淆矩阵，如下所示：

```py
INFO:tensorflow:Step 18000: Validation accuracy = 88.5% (N=4445)
INFO:tensorflow:Saving to "/tmp/speech_commands_train/conv.ckpt-18000"
INFO:tensorflow:set_size=4890
INFO:tensorflow:Confusion Matrix:
 [[404 2 0 0 0 0 0 0 0 0 2 0]
 [ 1 283 10 3 14 15 15 22 12 4 10 19]
 [ 0 7 394 4 1 3 9 0 0 0 1 0]
 [ 0 8 7 353 0 7 9 1 0 0 0 20]
 [ 2 4 1 0 397 6 2 0 1 6 5 1]
 [ 1 8 1 36 2 342 6 1 0 0 0 9]
 [ 1 2 14 1 4 0 386 4 0 0 0 0]
 [ 1 9 0 2 1 0 10 368 3 0 1 1]
 [ 2 13 0 0 7 10 1 0 345 15 3 0]
 [ 1 8 0 0 34 0 3 1 14 329 7 5]
 [ 0 1 1 0 11 3 0 0 1 2 387 5]
 [ 3 16 2 58 6 9 3 2 0 1 1 301]]
```

1.  最后，代码打印出测试集的准确度，如下所示：

```py
INFO:tensorflow:Final test accuracy = 87.7% (N=4890)
```

就是这样。模型已经训练完成，可以通过 TensorFlow 导出并用于服务，或者嵌入到其他桌面、Web 或移动应用中。

# 总结

在本章中，我们学习了一个将音频数据转为文本的项目。现在有许多开源 SDK 和商业付费云服务，可以将音频记录和文件转换为文本数据。作为示例项目，我们使用了 Google 的语音命令数据集和 TensorFlow 基于深度学习的示例，将音频文件转换为语音命令进行识别。

在下一章，我们将继续这个旅程，构建一个使用高斯过程预测股票价格的项目，这是一个广泛用于预测的算法。

# 问题

1.  混淆矩阵在训练结束时的解释是什么？

1.  创建一个你和家人朋友录制的声音数据集。使用这个数据运行模型并观察准确率如何。

1.  在你自己的数据集上重新训练模型，并检查你自己训练、验证和测试集的准确度。

1.  尝试修改 `train.py` 中的不同选项，并在博客中分享你的发现。

1.  向 `models.py` 文件中添加不同的架构，看看你是否能为语音数据集或自己录制的数据集创建一个更好的架构。

# 进一步阅读

以下链接有助于深入了解语音转文本：

+   [`arxiv.org/pdf/1804.03209.pdf`](https://arxiv.org/pdf/1804.03209.pdf)

+   [`ai.googleblog.com/2017/08/launching-speech-commands-dataset.html`](https://arxiv.org/pdf/1804.03209.pdf)

+   [`research.google.com/pubs/archive/43813.pdf`](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html)

+   [`cloud.google.com/speech-to-text/docs/`](https://research.google.com/pubs/archive/43813.pdf)

+   [`docs.aws.amazon.com/transcribe/latest/dg/what-is-transcribe.html`](https://docs.aws.amazon.com/transcribe/latest/dg/what-is-transcribe.html)

+   [`docs.microsoft.com/en-us/azure/cognitive-services/speech-service/`](https://docs.aws.amazon.com/transcribe/latest/dg/what-is-transcribe.html)

+   [`nlp.stanford.edu/projects/speech.shtml`](https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/)
