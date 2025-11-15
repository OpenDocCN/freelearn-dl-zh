# 前言

近年来，深度学习在视觉、语音、自然语言处理与理解以及其他数据丰富领域的难题中取得了前所未有的成功。企业、大学、政府和研究机构对该领域的关注加速了技术的进步。本书涵盖了深度学习中的一些重要进展。通过介绍原理背景、深入挖掘概念背后的直觉、使用 Keras 实现方程和算法，并分析结果，本书对高级理论进行了详细讲解。

**人工智能**（**AI**）现阶段仍然远未成为一个完全被理解的领域。作为 AI 子领域的深度学习，也处于同样的境地。虽然它远未成熟，但许多现实世界的应用，如基于视觉的检测与识别、产品推荐、语音识别与合成、节能、药物发现、金融和营销，已经在使用深度学习算法。未来还会发现并构建更多的应用。本书的目标是解释高级概念，提供示例实现，并让读者作为该领域的专家，识别出目标应用。

一个尚不完全成熟的领域是一把双刃剑。一方面，它为发现和开发提供了许多机会。深度学习中有很多尚未解决的问题，这为成为市场先行者——无论是在产品开发、出版还是行业认可上——提供了机会。另一方面，在任务关键型环境中，很难信任一个尚未完全理解的领域。我们可以放心地说，如果有人问，很少有机器学习工程师会选择乘坐由深度学习系统控制的自动驾驶飞机。要获得这种程度的信任，还需要做大量的工作。本书中讨论的高级概念很有可能在建立这种信任基础方面发挥重要作用。

每一本关于深度学习的书都不可能完全涵盖整个领域，本书也不例外。鉴于时间和篇幅的限制，我们本可以触及一些有趣的领域，如检测、分割与识别、视觉理解、概率推理、自然语言处理与理解、语音合成以及自动化机器学习。然而，本书相信，选取并讲解某些领域能够使读者能够进入未覆盖的其他领域。

当读者准备继续阅读本书时，需要牢记，他们选择了一个充满激动人心的挑战并且能够对社会产生巨大影响的领域。我们很幸运，拥有一份让我们每天早晨醒来都期待的工作。

# 本书适合谁阅读

本书面向希望深入理解深度学习高级主题的机器学习工程师和学生。每个讨论都附有 Keras 中的代码实现。本书适合那些希望了解如何将理论转化为在 Keras 中实现的工作代码的读者。除了理解理论外，代码实现通常是将机器学习应用于现实问题时最具挑战性的任务之一。

# 本书内容概览

第一章，*使用 Keras 介绍高级深度学习*，涵盖了深度学习的关键概念，如优化、正则化、损失函数、基础层和网络及其在 Keras 中的实现。本章还回顾了深度学习和 Keras 的使用，采用顺序 API。

第二章，*深度神经网络*，讨论了 Keras 的功能 API。探讨了两种广泛使用的深度网络架构，ResNet 和 DenseNet，并在 Keras 中通过功能 API 进行了实现。

第三章，*自编码器*，介绍了一个常见的网络结构——自编码器，用于发现输入数据的潜在表示。讨论并在 Keras 中实现了两个自编码器的应用示例：去噪和着色。

第四章，*生成对抗网络（GANs）*，讨论了深度学习中的一项重要进展。GAN 用于生成看似真实的新合成数据。本章解释了 GAN 的原理。本文讨论并实现了两种 GAN 示例：DCGAN 和 CGAN，均在 Keras 中实现。

第五章，*改进的 GANs*，介绍了改进基础 GAN 的算法。这些算法解决了训练 GAN 时的难题，并提高了合成数据的感知质量。讨论并在 Keras 中实现了 WGAN、LSGAN 和 ACGAN。

第六章，*解耦表示 GANs*，讨论了如何控制 GAN 生成的合成数据的属性。如果潜在表示解耦，这些属性就可以被控制。介绍了两种解耦表示的技术：InfoGAN 和 StackedGAN，并在 Keras 中进行了实现。

第七章，*跨域 GANs*，涵盖了 GAN 的一个实际应用，即将图像从一个领域转移到另一个领域，通常称为跨域迁移。讨论了广泛使用的跨域 GAN——CycleGAN，并在 Keras 中实现。本章还展示了 CycleGAN 执行图像着色和风格迁移的过程。

第八章，*变分自编码器（VAE）*，讨论了深度学习中的另一项重大进展。与 GAN 类似，VAE 是一种生成模型，用于生成合成数据。与 GAN 不同，VAE 侧重于可解码的连续潜在空间，适用于变分推断。VAE 及其变种 CVAE 和 β-VAE 在 Keras 中进行了实现。

第九章，*深度强化学习*，解释了强化学习和 Q 学习的原理。介绍了实现 Q 学习的两种技术，Q 表更新和深度 Q 网络（DQN）。在 OpenAI gym 环境中，演示了使用 Python 和 DQN 实现 Q 学习。

第十章，*策略梯度方法*，解释了如何使用神经网络学习强化学习中的决策策略。涵盖了四种方法，并在 Keras 和 OpenAI gym 环境中实现，分别是 REINFORCE、带基线的 REINFORCE、演员-评论家和优势演员-评论家。本章中的示例展示了在连续动作空间中应用策略梯度方法。

# 为了最大限度地从本书中受益

+   **深度学习与 Python**：读者应具备基本的深度学习知识，并了解如何在 Python 中实现深度学习。虽然有使用 Keras 实现深度学习算法的经验会有所帮助，但并不是必须的。第一章，*使用 Keras 介绍深度学习进阶*，回顾了深度学习的概念及其在 Keras 中的实现。

+   **数学**：本书中的讨论假设读者具备大学水平的微积分、线性代数、统计学和概率论基础知识。

+   **GPU**：本书中的大多数 Keras 实现需要 GPU。如果没有 GPU，执行许多代码示例将不可行，因为所需时间过长（可能需要几个小时甚至几天）。本书中的示例尽量使用合理的数据大小，以减少高性能计算机的使用。本书假设读者至少可以访问 NVIDIA GTX 1060。

+   **编辑**：本书中的代码示例是在`vim`编辑器下，使用 Ubuntu Linux 16.04 LTS、Ubuntu Linux 17.04 和 macOS High Sierra 操作系统编辑的。任何支持 Python 的文本编辑器都可以使用。

+   **TensorFlow**：Keras 需要一个后端。本书中的代码示例使用 Keras 和 TensorFlow 后端编写。请确保正确安装了 GPU 驱动和`tensorflow`。

+   **GitHub**：我们通过示例和实验来学习。请从本书的 GitHub 仓库中 `git pull` 或 `fork` 代码包。获取代码后，检查它。运行它。修改它。再次运行它。通过调整代码示例进行所有创造性的实验。这是理解章节中所有理论的唯一方法。我们也非常感激你在书籍 GitHub 仓库上给予星标。

## 下载示例代码文件

本书的代码包托管在 GitHub 上，地址是：

[`github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras`](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras)

我们还在 [`github.com/PacktPublishing/`](https://github.com/PacktPublishing/) 提供了其他来自我们丰富书籍和视频目录的代码包，快去看看吧！

## 下载彩色图片

我们还提供了一个 PDF 文件，其中包含本书中使用的截图/图表的彩色图片。你可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/9781788629416_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781788629416_ColorImages.pdf)。

## 使用的约定

本书中的代码示例使用 Python 编写，具体来说是 `python3`。配色方案基于 vim 语法高亮。考虑以下示例：

```py
def encoder_layer(inputs,
                  filters=16,
                  kernel_size=3,
                  strides=2,
                  activation='relu',
                  instance_norm=True):
    """Builds a generic encoder layer made of Conv2D-IN-LeakyReLU
    IN is optional, LeakyReLU may be replaced by ReLU

    """

    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same')

    x = inputs
    if instance_norm:
        x = InstanceNormalization()(x)
    if activation == 'relu':
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    return x
```

尽可能包含文档字符串。至少使用文本注释来 最小化空间的使用。

所有命令行代码执行格式如下：

```py
$ python3 dcgan-mnist-4.2.1.py

```

示例代码文件命名格式为：`algorithm-dataset-chapter.section.number.py`。命令行示例是第四章第二节的第一段代码，使用的是 DCGAN 算法和 MNIST 数据集。在某些情况下，执行命令行并未明确写出，但默认是：

```py
$ python3 name-of-the-file-in-listing

```

```py
The file name of the code example is included in the Listing caption.

```

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：发送电子邮件至 `<feedback@packtpub.com>`，并在邮件主题中注明书名。如果你对本书的任何部分有疑问，请通过 `<questions@packtpub.com>` 给我们发送邮件。

**勘误**：尽管我们已尽最大努力确保内容的准确性，但错误还是难免发生。如果你在本书中发现错误，请告知我们。请访问，[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择你的书籍，点击“勘误提交表单”链接，并填写相关信息。

**盗版**：如果你在互联网上发现任何形式的非法复制品，我们将非常感激你提供相关网址或网站名称。请通过 `<copyright@packtpub.com>` 联系我们，并附上相关材料的链接。

**如果你有兴趣成为作者**：如果你对某个领域有专长，并且有兴趣编写或为书籍贡献内容，请访问 [`authors.packtpub.com`](http://authors.packtpub.com)。

## 评论

请留下评论。当你阅读并使用过这本书后，为什么不在你购买它的站点上留下评论呢？潜在的读者可以看到并参考你公正的意见做出购买决策，我们在 Packt 能够了解你对我们产品的看法，作者也可以看到你对他们书籍的反馈。谢谢！

了解更多关于 Packt 的信息，请访问 [packtpub.com](http://packtpub.com)。
