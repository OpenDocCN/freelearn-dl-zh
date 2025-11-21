# 第十章：从 TensorFlow 1 迁移到 TensorFlow 2

由于 TensorFlow 2 最近才发布，因此大多数在线项目仍然是为 TensorFlow 1 构建的。尽管第一版已经具备了许多有用的功能，如 AutoGraph 和 Keras API，但建议你迁移到最新版本的 TensorFlow，以避免技术债务。幸运的是，TensorFlow 2 提供了一个自动迁移工具，能够将大多数项目转换为其最新版本。该工具几乎不需要额外的努力，并且会输出功能正常的代码。然而，要将代码迁移到符合 TensorFlow 2 规范的版本，需要一些细心和对两个版本的了解。在本节中，我们将介绍迁移工具，并将 TensorFlow 1 的概念与其 TensorFlow 2 对应概念进行比较。

# 自动迁移

安装 TensorFlow 2 后，可以通过命令行使用迁移工具。要转换项目目录，请运行以下命令：

```py
$ tf_upgrade_v2 --intree ./project_directory --outtree ./project_directory_updated
```

以下是示例项目中命令日志的样本：

```py
INFO line 1111:10: Renamed 'tf.placeholder' to 'tf.compat.v1.placeholder'
 INFO line 1112:10: Renamed 'tf.layers.dense' to 'tf.compat.v1.layers.dense'
 TensorFlow 2.0 Upgrade Script
 -----------------------------
 Converted 21 files
 Detected 1 issues that require attention
 ----------------------------------------------------------------------
 ----------------------------------------------------------------------
 File: project_directory/test_tf_converter.py
 ----------------------------------------------------------------------
 project_directory/test_tf_converter.py:806:10: WARNING: tf.image.resize_bilinear called with align_corners argument requires manual check: align_corners is not supported by tf.image.resize, the new default transformation is close to what v1 provided. If you require exactly the same transformation as before, use compat.v1.image.resize_bilinear.
  Make sure to read the detailed log 'report.txt'
```

转换工具会详细列出它对文件所做的所有更改。在极少数情况下，当它检测到需要手动处理的代码行时，会输出带有更新说明的警告。

大多数过时的调用都已移至 `tf.compat.v1`。事实上，尽管许多概念已经废弃，TensorFlow 2 仍然通过此模块提供对旧 API 的访问。然而，请注意，调用 `tf.contrib` 会导致转换工具失败并生成错误：

```py
ERROR: Using member tf.contrib.copy_graph.copy_op_to_graph in deprecated module tf.contrib. tf.contrib.copy_graph.copy_op_to_graph cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository, or fork the required code.
```

# 迁移 TensorFlow 1 代码

如果工具运行没有任何错误，代码可以按原样使用。然而，迁移工具使用的 `tf.compat.v1` 模块被认为已废弃。调用此模块时会输出废弃警告，并且该模块的内容将不再由社区更新。因此，建议重构代码，使其更加符合 TensorFlow 2 的规范。在接下来的部分中，我们将介绍 TensorFlow 1 的概念，并解释如何将它们迁移到 TensorFlow 2。在以下示例中，将使用 `tf1` 来代替 `tf`，表示使用 TensorFlow 1.13。

# 会话

由于 TensorFlow 1 默认不使用即时执行（eager execution），因此操作的结果不会直接显示。例如，当对两个常量求和时，输出对象是一个操作：

```py
import tensorflow as tf1 # TensorFlow 1.13

a = tf1.constant([1,2,3])
b = tf1.constant([1,2,3])
c = a + b
print(c) # Prints <tf.Tensor 'add:0' shape=(3,) dtype=int32
```

为了计算结果，你需要手动创建 `tf1.Session`。会话负责以下任务：

+   管理内存

+   在 CPU 或 GPU 上运行操作

+   如有必要，跨多台机器运行

使用会话最常见的方法是通过 Python 中的 `with` 语句。与其他不受管理的资源一样，`with` 语句确保我们使用完会话后会正确关闭。如果会话没有关闭，可能会继续占用内存。因此，TensorFlow 1 中的会话通常是这样实例化和使用的：

```py
with tf1.Session() as sess:
 result = sess.run(c)
print(result) # Prints array([2, 4, 6], dtype=int32)
```

你也可以显式关闭会话，但不推荐这么做：

```py
sess = tf1.Session()
result = sess.run(c)
sess.close()
```

在 TensorFlow 2 中，会话管理发生在幕后。由于新版本使用了急切执行，因此不需要这段冗余代码来计算结果。因此，可以删除对 `tf1.Session()` 的调用。

# `Placeholders`

在之前的示例中，我们计算了两个向量的和。然而，我们在创建图时定义了这些向量的值。如果我们想使用变量代替，我们本可以使用 `tf1.placeholder`：

```py
a = tf1.placeholder(dtype=tf.int32, shape=(None,)) 
b = tf1.placeholder(dtype=tf.int32, shape=(None,))
c = a + b

with tf1.Session() as sess:
  result = sess.run(c, feed_dict={
      a: [1, 2, 3],
      b: [1, 1, 1]
    })
```

在 TensorFlow 1 中，`placeholders` 主要用于提供输入数据。它们的类型和形状必须定义。在我们的示例中，形状是 `(None,)`，因为我们可能希望在任意大小的向量上运行操作。在运行图时，我们必须为 `placeholders` 提供具体的值。这就是我们在 `sess.run` 中使用 `feed_dict` 参数的原因，将变量的内容作为字典传递，`placeholders` 作为键。如果未为所有 `placeholders` 提供值，将会引发异常。

在 TensorFlow 2 之前，`placeholders` 被用来提供输入数据和层的参数。前者可以通过 `tf.keras.Input` 来替代，而后者可以通过 `tf.keras.layers.Layer` 参数来处理。

# 变量管理

在 TensorFlow 1 中，变量是全局创建的。每个变量都有一个唯一的名称，创建变量的最佳实践是使用 `tf1.get_variable()`：

```py
weights = tf1.get_variable(name='W', initializer=[3])
```

在这里，我们创建了一个名为 `W` 的全局变量。删除 Python 中的 `weights` 变量（例如使用 Python 的 `del weights` 命令）不会影响 TensorFlow 内存。事实上，如果我们尝试再次创建相同的变量，我们将会遇到错误：

```py
Variable W already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
```

虽然 `tf1.get_variable()` 允许你重用变量，但它的默认行为是在选择的变量名已经存在时抛出错误，以防止你不小心覆盖变量。为了避免这个错误，我们可以更新调用 `tf1.variable_scope(...)` 并使用 `reuse` 参数：

```py
with tf1.variable_scope("conv1", reuse=True):
    weights = tf1.get_variable(name='W', initializer=[3])
```

`variable_scope` 上下文管理器用于管理变量的创建。除了处理变量重用外，它还通过为变量名称附加前缀来方便地将变量分组。在之前的示例中，变量会被命名为 `conv1/W`。

在这种情况下，将 `reuse` 设置为 `True` 意味着，如果 TensorFlow 遇到名为 `conv1/W` 的变量，它不会像之前那样抛出错误。相反，它会重用现有的变量及其内容。然而，如果你尝试调用之前的代码，而名为 `conv1/W` 的变量不存在，你将遇到以下错误：

```py
Variable conv1/W does not exist
```

实际上，`reuse=True` 只能在重用现有变量时指定。如果你想在变量不存在时创建一个变量，并在它存在时重用，可以传递 `reuse=tf.AUTO_REUSE`。

在 TensorFlow 2 中，行为有所不同。虽然变量作用域依然存在以便于命名和调试，但变量不再是全局的。它们在 Python 层级上进行管理。只要你能够访问 Python 引用（在我们的例子中是 `weights` 变量），就可以修改该变量。要删除变量，你需要删除其引用，例如通过运行以下命令：

```py
del weights
```

以前，变量可以全局访问和修改，并且可能会被其他代码覆盖。全局变量的弃用使得 TensorFlow 代码更加易读且更不容易出错。

# 层与模型

TensorFlow 模型最初是通过 `tf1.layers` 定义的。由于该模块在 TensorFlow 2 中已被弃用，推荐使用 `tf.keras.layers` 作为替代。要使用 TensorFlow 1 训练模型，需要使用优化器和损失函数定义一个*训练操作*。例如，如果 `y` 是全连接层的输出，我们可以使用以下命令定义训练操作：

```py
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=output, logits=y))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

```

每次调用此操作时，一批图像将被送入网络并执行一次反向传播步骤。然后我们运行一个循环来计算多个训练步骤：

```py
num_steps = 10**7

with tf1.Session() as sess:
    sess.run(tf1.global_variables_initializer())

    for i in range(num_steps):
        batch_x, batch_y = next(batch_generator)
        sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
```

在打开会话时，需要调用 `tf1.global_variables_initializer()` 以确保层被正确初始化。如果没有这样做，将抛出异常。在 TensorFlow 2 中，变量的初始化是自动处理的。

# 其他概念

我们详细介绍了在新版本中被弃用的 TensorFlow 1 中最常见的概念。许多较小的模块和范式在 TensorFlow 2 中也进行了重新设计。在迁移项目时，我们建议详细查看两个版本的文档。为了确保迁移顺利，并且 TensorFlow 2 版本按预期工作，我们建议你记录推理指标（如延迟、准确率或平均精度）和训练指标（如收敛前的迭代次数），并比较旧版和新版的值。

由于 TensorFlow 是开源的，并且得到了活跃社区的支持，它不断发展——集成新特性、优化其他功能、改善开发者体验等等。尽管这有时需要额外的工作，但尽早升级到最新版本将为你提供最佳的环境，以开发性能更高的识别应用程序。

# 参考文献

本节列出了本书中提到的科学论文和其他网络资源。

# 第一章：计算机视觉与神经网络

+   Angeli, A., Filliat, D., Doncieux, S., Meyer, J.-A., 2008\. *一种快速且增量的方法，用于基于视觉词袋的回环检测*。*IEEE 机器人学报 1027–1037*。

+   Bradski, G., Kaehler, A., 2000\. OpenCV。*Dr. Dobb’s 软件工具期刊 3*。

+   Cortes, C., Vapnik, V., 1995\. *支持向量网络*。 *机器学习 20，273–297*。

+   Drucker, H., Burges, C.J., Kaufman, L., Smola, A.J., Vapnik, V., 1997\. *支持向量回归机*。*见：神经信息处理系统进展，pp. 155–161*。

+   Krizhevsky, A., Sutskever, I., Hinton, G.E., 2012\. *ImageNet 分类与深度卷积神经网络*。*见：神经信息处理系统进展，pp. 1097–1105*。

+   Lawrence, S., Giles, C.L., Tsoi, A.C., Back, A.D., 1997\. *面部识别：卷积神经网络方法*。*IEEE 神经网络交易 8, 98–113*。

+   LeCun, Y., Boser, B.E., Denker, J.S., Henderson, D., Howard, R.E., Hubbard, W.E., Jackel, L.D., 1990\. *使用反向传播网络进行手写数字识别*。*见：神经信息处理系统进展，pp. 396–404*。

+   LeCun, Y., Cortes, C., Burges, C., 2010\. *MNIST 手写数字数据库。AT&T Labs [在线]*。可在 [`yann.lecun.com/exdb/mnist`](http://yann.lecun.com/exdb/mnist) 查阅 2, 18。

+   Lowe, D.G., 2004\. *从尺度不变关键点提取独特图像特征*。*国际计算机视觉杂志 60, 91–110*。

+   Minsky, M., 1961\. *迈向人工智能的步骤*。*IRE 会议记录 49, 8–30*。

+   Minsky, M., Papert, S.A., 2017\. *感知器：计算几何学入门。MIT 出版社*。

+   Moravec, H., 1984\. *运动、视觉与智能*。

+   Papert, S.A., 1966\. *夏季视觉项目*。

+   Plaut, D.C., 等人，1986\. *反向传播学习实验*。

+   Rosenblatt, F., 1958\. *感知器：大脑中信息存储与组织的概率模型*。*心理学评论 65, 386*。

+   Turk, M., Pentland, A., 1991\. *用于识别的特征脸*。*认知神经科学杂志 3, 71–86*。

+   Wold, S., Esbensen, K., Geladi, P., 1987\. *主成分分析*。*化学计量学与智能实验室系统 2, 37–52*。

# 第二章：TensorFlow 基础与模型训练

+   Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G.S., Davis, A., Dean, 等人。*TensorFlow: 大规模机器学习在异构分布式系统上的应用 19*。

+   *API 文档 [WWW 文档]，无日期。TensorFlow*。网址：[`www.tensorflow.org/api_docs/`](https://www.tensorflow.org/api_docs/)（访问于 2018 年 12 月 14 日）。

+   Chollet, F., 2018\. TensorFlow 是研究界深度学习的首选平台。过去三个月在 arXiv 上有提到深度学习框架，*pic.twitter.com/v6ZEi63hzP. @fchollet*。

+   Goldsborough, P., 2016\. *TensorFlow 介绍。arXiv:1610.01178 [cs]*。

# 第三章：现代神经网络

+   Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., Devin, M., Ghemawat, S., Irving, G., Isard, M., 等人，2016\. *Tensorflow: 大规模机器学习的系统。见：OSDI，pp. 265–283*。

+   *API 文档，网址*：[`www.tensorflow.org/api_docs/`](https://www.tensorflow.org/api_docs/)（访问于 2018 年 12 月 14 日）。

+   Bottou, L., 2010. *使用随机梯度下降进行大规模机器学习. 载于：COMPSTAT'2010 会议论文集*. *Springer，第 177–186 页*。

+   Bottou, L., Curtis, F.E., Nocedal, J., 2018. *大规模机器学习的优化方法. SIAM 综述 60, 223–311*。

+   Dozat, T., 2016. *将 Nesterov 动量融入 Adam*。

+   Duchi, J., Hazan, E., Singer, Y., 2011. *在线学习与随机优化的自适应子梯度方法. 机器学习研究杂志 12, 2121–2159*。

+   Gardner, W.A., 1984. *随机梯度下降算法的学习特性：一项综合研究、分析与批评. 信号处理 6, 113–133*。

+   Girosi, F., Jones, M., Poggio, T., 1995. *正则化理论与神经网络架构*. *神经计算 7, 219–269*。

+   Ioffe, S., Szegedy, C., 2015. *批量归一化：通过减少内部协方差偏移加速深度网络训练.* *arXiv 预印本 arXiv:1502.03167*。

+   Karpathy, A., n.d. *斯坦福大学 CS231n：用于视觉识别的卷积神经网络 [WWW 文档]*. URL: [`cs231n.stanford.edu/`](http://cs231n.stanford.edu/) （访问日期：2018 年 12 月 14 日）。

+   Kingma, D.P., Ba, J., 2014. *Adam：一种随机优化方法. arXiv 预印本 arXiv:1412.6980*。

+   Krizhevsky, A., Sutskever, I., Hinton, G.E., 2012. *使用深度卷积神经网络进行图像分类*. *载于：神经信息处理系统进展，第 1097–1105 页*。

+   Lawrence, S., Giles, C.L., Tsoi, A.C., Back, A.D., 1997. *人脸识别：一种卷积神经网络方法*. *IEEE 神经网络学报 8, 98–113*。

+   Le 和 Borji – 2017 – *卷积神经网络中神经元的感受野、有效感受野和投影场是什么？ pdf, n.d*。

+   Le, H., Borji, A., 2017. *卷积神经网络中神经元的感受野、有效感受野和投影场是什么？ arXiv:1705.07049 [cs]*。

+   LeCun, Y., Cortes, C., Burges, C., 2010. *MNIST 手写数字数据库. AT&T 实验室 [在线]*. 可从 [`yann.lecun.com/exdb/mnist`](http://yann.lecun.com/exdb/mnist) 获取 2。

+   LeCun, Y., 等, 2015. LeNet-5, *卷积神经网络*. URL: [`yann.lecun.com/exdb/lenet`](http://yann.lecun.com/exdb/lenet) 20。

+   Lenail, A., *n.d. NN SVG [WWW 文档]*. URL: [`alexlenail.me/NN-SVG/`](http://alexlenail.me/NN-SVG/) （访问日期：2018 年 12 月 14 日）。

+   Luo, W., Li, Y., Urtasun, R., Zemel, R., n.d. *理解深度卷积神经网络中的有效感受野 9*。

+   Nesterov, Y., 1998. *凸编程导论 第一卷：基础课程. 讲义*。

+   Perkins, E.S., Davson, H., n.d. *人眼 | 定义、结构与功能 [WWW 文档]*. *大英百科全书*. URL: [`www.britannica.com/science/human-eye`](https://www.britannica.com/science/human-eye) （访问日期：2018 年 12 月 14 日）。

+   Perone, C.S., n.d. *卷积神经网络中的有效感受野 | Terra Incognita. Terra Incognita*。

+   Polyak, B.T., 1964\. *加速迭代方法收敛的一些方法. 苏联计算数学与数学物理 4, 1–17*.

+   Raj, D., 2018\. *梯度下降优化算法简短说明*. *Medium*.

+   Simard, P.Y., Steinkraus, D., Platt, J.C., 2003\. *卷积神经网络在视觉文档分析中的最佳实践. 见：Null，第 958 页*.

+   Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R., 2014\. *Dropout：一种防止神经网络过拟合的简单方法. 机器学习研究期刊 15, 1929–1958*.

+   Sutskever, I., Martens, J., Dahl, G., Hinton, G., 2013\. *在深度学习中初始化与动量的重要性. 见：国际机器学习大会，1139–1147 页*.

+   Tieleman, T., Hinton, G., 2012\. *讲座 6.5-rmsprop：* *通过最近的梯度幅度的滑动平均值来划分梯度. COURSERA: 神经网络与机器学习 4, 26–31*.

+   Walia, A.S., 2017\. *神经网络中使用的优化算法类型及优化梯度下降的方法 [WWW 文档]. Towards Data Science*. URL: [`towardsdatascience.com/types-of-optimization-algorithms-used-in-neural-networks-and-ways-to-optimize-gradient-95ae5d39529f`](https://towardsdatascience.com/types-of-optimization-algorithms-used-in-neural-networks-and-ways-to-optimize-gradient-95ae5d39529f)（访问时间：2018 年 12 月 14 日）。

+   Zeiler, M.D., 2012\. *ADADELTA：一种自适应学习率方法. arXiv 预印本 arXiv:1212.5701*.

+   Zhang, T., 2004\. *使用随机梯度下降算法解决大规模线性预测问题*. *见：第二十一届国际机器学习会议论文集，第 116 页*.

# 第四章：影响力的分类工具

+   *API 文档 [WWW 文档], n.d. TensorFlow*. URL: [`www.tensorflow.org/api_docs/`](https://www.tensorflow.org/api_docs/)（访问时间：2018 年 12 月 14 日）。

+   Goodfellow, I., Bengio, Y., Courville, A., 2016\. *深度学习. MIT 出版社*.

+   He, K., Zhang, X., Ren, S., Sun, J., 2015\. *用于图像识别的深度残差学习. arXiv:1512.03385 [cs]*.

+   Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., Adam, H., 2017\. *MobileNets：面向移动视觉应用的高效卷积神经网络. arXiv:1704.04861 [cs]*.

+   Huang, G., Liu, Z., van der Maaten, L., Weinberger, K.Q., 2016\. *密集连接卷积网络. arXiv:1608.06993 [cs]*.

+   Karpathy, A., n.d. *斯坦福大学 CS231n：视觉识别的卷积神经网络 [WWW 文档]*. URL: [`cs231n.stanford.edu/`](http://cs231n.stanford.edu/)（访问时间：2018 年 12 月 14 日）。

+   Karpathy, A. *我从与 ConvNet 在 ImageNet 上竞争中学到的东西 [WWW 文档], 未注明日期*. URL: [`karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/`](http://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/)（访问日期：2019 年 1 月 4 日）。

+   Lin, M., Chen, Q., Yan, S., 2013\. *网络中的网络. arXiv:1312.4400 [cs]*.

+   Pan, S.J., Yang, Q., 2010\. *迁移学习调查. IEEE 知识与数据工程学报 22, 1345–1359*.

+   Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A.C., Fei-Fei, L., 2014\. *ImageNet 大规模视觉识别挑战. arXiv:1409.0575 [cs]*.

+   Sarkar, D. (DJ), 2018\. *迁移学习的全面实用指南：深度学习中的真实世界应用 [WWW 文档]. Towards Data Science*. URL: [`towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a`](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a)（访问日期：2019 年 1 月 15 日）。

+   shu-yusa, 2018\. *使用 TensorFlow Hub 的 Inception-v3 进行迁移学习. Medium*.

+   *Simonyan, K., Zisserman, A., 2014\. 用于大规模图像识别的非常深的卷积网络. arXiv:1409.1556 [cs]*.

+   Srivastava, R.K., Greff, K., Schmidhuber, J., 2015\. *高速公路网络. arXiv:1505.00387 [cs]*.

+   Szegedy, C., Ioffe, S., Vanhoucke, V., Alemi, A., 2016\. *Inception-v4, Inception-ResNet 以及残差连接对学习的影响. arXiv:1602.07261 [cs]*.

+   Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., Rabinovich, A., 2014\. *深入卷积神经网络的研究. arXiv:1409.4842 [cs]*.

+   Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z., 2015\. *重新思考 Inception 架构在计算机视觉中的应用. arXiv:1512.00567 [cs]*.

+   Thrun, S., Pratt, L., 1998\. *学习如何学习*.

+   Zeiler, Matthew D., Fergus, R., 2014\. *卷积网络的可视化与理解*. 见：Fleet, D., Pajdla, T., Schiele, B., Tuytelaars, T. (编辑), *计算机视觉 – ECCV 2014\. Springer 国际出版公司, Cham, 页码 818–833*.

+   Zeiler, Matthew D., Fergus, R., 2014\. *卷积网络的可视化与理解. 见：欧洲计算机视觉会议, 页码 818–833*.

# 第五章：目标检测模型

+   Everingham, M., Eslami, S.M.A., Van Gool, L., Williams, C.K.I., Winn, J., Zisserman, A., 2015\. *Pascal 视觉目标类别挑战：回顾. 国际计算机视觉杂志 111, 98–136*.

+   Girshick, R., 2015\. *Fast R-CNN. arXiv:1504.08083 [cs]*.

+   Girshick, R., Donahue, J., Darrell, T., Malik, J., 2013\. *准确的目标检测和语义分割的丰富特征层次. arXiv:1311.2524 [cs]*.

+   Redmon, J., Divvala, S., Girshick, R., Farhadi, A., 2015\. *You Only Look Once: 统一的实时目标检测。arXiv:1506.02640 [cs]*。

+   Redmon, J., Farhadi, A., 2016\. YOLO9000: *更好、更快、更强。arXiv:1612.08242 [cs]*。

+   Redmon, J., Farhadi, A., 2018\. YOLOv3: *渐进性改进。arXiv:1804.02767 [cs]*。

+   Ren, S., He, K., Girshick, R., Sun, J., 2015\. *Faster R-CNN: 基于区域提议网络的实时目标检测。arXiv:1506.01497 [cs]*。

# 第六章：图像增强与分割

+   Bai, M., Urtasun, R., 2016\. *用于实例分割的深度分水岭变换。arXiv:1611.08303 [cs]*。

+   Beyer, L., 2019\. *Python 包装器用于 Philipp Krähenbühl 的稠密（全连接）条件随机场，带有高斯边缘潜力：lucasb-eyer/pydensecr*`f`。

+   *在 Keras 中构建自编码器 [WWW Document]*, n.d. URL: [`blog.keras.io/building-autoencoders-in-keras.html`](https://blog.keras.io/building-autoencoders-in-keras.html) (accessed January 18, 2019).

+   Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., Schiele, B., 2016\. *用于语义城市场景理解的 Cityscapes 数据集。发表于 2016 IEEE 计算机视觉与模式识别大会（CVPR）*。*在 2016 IEEE 计算机视觉与模式识别大会（CVPR）上展示*，*IEEE, 拉斯维加斯, NV, USA, 第 3213-3223 页*。

+   Dice, L.R., 1945\. *物种间生态关联量度。生态学 26, 297–302*。

+   Drozdzal, M., Vorontsov, E., Chartrand, G., Kadoury, S., Pal, C., 2016\. *跳跃连接在生物医学图像分割中的重要性。arXiv:1608.04117 [cs]*。

+   Dumoulin, V., Visin, F., 2016\. *深度学习卷积算术指南。arXiv:1603.07285 [cs, stat]*。

+   Guan, S., Khan, A., Sikdar, S., Chitnis, P.V., n.d. *用于 2D 稀疏光声断层成像伪影去除的完全稠密 UNet 8*。

+   He, K., Gkioxari, G., Dollár, P., Girshick, R., 2017\. *Mask R-CNN. arXiv:1703.06870 [cs]*.

+   *Kaggle. 2018 数据科学碗 [WWW Document]*, n.d. URL: [`kaggle.com/c/data-science-bowl-2018`](https://kaggle.com/c/data-science-bowl-2018) (accessed February 8, 2019).

+   Krähenbühl, P., Koltun, V., n.d. *带有高斯边缘潜力的完全连接条件随机场高效推理 9*。

+   Lan, T., Li, Y., Murugi, J.K., Ding, Y., Qin, Z., 2018\. *RUN：用于计算机辅助肺结节检测的残差 U-Net，无需候选选择。arXiv:1805.11856 [cs]*。

+   Li, X., Chen, H., Qi, X., Dou, Q., Fu, C.-W., Heng, P.A., 2017\. *H-DenseUNet: 用于肝脏和肿瘤分割的混合稠密连接 UNet，从 CT 体积中提取。arXiv:1709.07330 [cs]*。

+   Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P., 2017\. *Focal Loss for Dense Object Detection. arXiv:1708.02002 [cs]*.

+   Milletari, F., Navab, N., Ahmadi, S.-A., 2016\. *V-Net：用于体积医学图像分割的完全卷积神经网络*。*在：2016 年第四届国际三维视觉会议（3DV）*。*在 2016 年第四届国际三维视觉会议（3DV）上展示，IEEE，斯坦福，美国加利福尼亚州，页码 565–571*。

+   Noh, H., Hong, S., Han, B., 2015\. *用于语义分割的去卷积网络学习*。在：2015 *IEEE 国际计算机视觉会议（ICCV）*。*2015 年 ICCV 会议上展示，IEEE，智利圣地亚哥，页码 1520–1528*。

+   Odena, A., Dumoulin, V., Olah, C., 2016\. *去卷积与棋盘伪影。Distill 1, e3*。

+   Ronneberger, O., Fischer, P., Brox, T., 2015\. *U-Net：用于生物医学图像分割的卷积网络。arXiv:1505.04597 [cs]*。

+   Shelhamer, E., Long, J., Darrell, T., 2017\. *完全卷积网络用于语义分割。IEEE 模式分析与机器智能学报 39, 640–651*。

+   Sørensen, T., 1948\. *基于物种相似性在植物社会学中建立相等幅度群体的方法及其在丹麦公共草地植被分析中的应用。Biol. Skr. 5, 1–34*。

+   *无监督特征学习与深度学习教程 [WWW 文档]*，无日期。URL：[`ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/`](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/)（访问日期：2019 年 1 月 17 日）。

+   Zeiler, M.D., Fergus, R., 2013\. *可视化与理解卷积网络。arXiv:1311.2901 [cs]*。

+   Zhang, Z., Liu, Q., Wang, Y., 2018\. *基于深度残差 U-Net 的道路提取。IEEE 地球科学与遥感快报 15, 749–753*。

# 第七章：在复杂和稀缺数据集上的训练

+   Bousmalis, K., Silberman, N., Dohan, D., Erhan, D., Krishnan, D., 2017a. *基于生成对抗网络的无监督像素级领域适应*。*在：2017 年 IEEE 计算机视觉与模式识别会议（CVPR）上展示。2017 年 IEEE 计算机视觉与模式识别会议（CVPR），IEEE，夏威夷檀香山，页码 95–104*。

+   Bousmalis, K., Silberman, N., Dohan, D., Erhan, D., Krishnan, D., 2017b. *基于生成对抗网络的无监督像素级领域适应。 在：IEEE 计算机视觉与模式识别会议论文集，页码 3722–3731*。

+   Brodeur, S., Perez, E., Anand, A., Golemo, F., Celotti, L., Strub, F., Rouat, J., Larochelle, H., Courville, A., 2017\. *HoME：家庭多模态环境。arXiv:1711.11017 [cs, eess]*。

+   Chang, A.X., Funkhouser, T., Guibas, L., Hanrahan, P., Huang, Q., Li, Z., Savarese, S., Savva, M., Song, S., Su, H., Xiao, J., Yi, L., Yu, F., 2015\. ShapeNet: *一个信息丰富的 3D 模型库（编号 arXiv:1512.03012 [cs.GR]）。斯坦福大学 – 普林斯顿大学 – 芝加哥丰田技术研究院*。

+   Chen, Y., Li, W., Sakaridis, C., Dai, D., Van Gool, L., 2018\. *面向野外目标检测的领域自适应 Faster R-CNN. 见：2018 年 IEEE/CVF 计算机视觉与模式识别大会*。*在 2018 年 IEEE/CVF 计算机视觉与模式识别大会（CVPR）上发表，IEEE，美国犹他州盐湖城，第 3339–3348 页*。

+   Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., Schiele, B., 2016\. *Cityscapes 数据集：语义城市场景理解. 见：IEEE 计算机视觉与模式识别会议论文集，第 3213–3223 页*。

+   Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M., Lempitsky, V., 2017\. *神经网络的领域对抗训练. 见：Csurka, G.（编），《计算机视觉应用中的领域适应》，Springer 国际出版社，Cham，第 189–209 页*。

+   Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y., 2014\. *生成对抗网络. 见：《神经信息处理系统进展》，第 2672–2680 页*。

+   Gschwandtner, M., Kwitt, R., Uhl, A., Pree, W., 2011\. *BlenSor：Blender 传感器仿真工具箱. 见：国际视觉计算研讨会，第 199–208 页*。

+   Hernandez-Juarez, D., Schneider, L., Espinosa, A., Vázquez, D., López, A.M., Franke, U., Pollefeys, M., Moure, J.C., 2017\. *倾斜的 Stixels：表示旧金山最陡峭的街道. arXiv:1707.05397 [cs]*。

+   Hoffman, J., Tzeng, E., Park, T., Zhu, J.-Y., Isola, P., Saenko, K., Efros, A.A., Darrell, T., 2017\. *CyCADA：循环一致性对抗领域适应. arXiv:1711.03213 [cs]*。

+   Isola, P., Zhu, J.-Y., Zhou, T., Efros, A.A., 2017\. *基于条件对抗网络的图像到图像翻译. 见：IEEE 计算机视觉与模式识别会议论文集，第 1125–1134 页*。

+   Kingma, D.P., Welling, M., 2013\. *自编码变分贝叶斯. arXiv 预印本 arXiv:1312.6114*。

+   Long, M., Cao, Y., Wang, J., Jordan, M.I., 无日期。 *使用深度适应网络学习可迁移特征 9*。

+   Planche, B., Wu, Z., Ma, K., Sun, S., Kluckner, S., Lehmann, O., Chen, T., Hutter, A., Zakharov, S., Kosch, H., 等人，2017\. *Depthsynth：来自 CAD 模型的实时逼真合成数据生成，用于 2.5D 识别. 见：2017 年国际三维视觉会议（3DV），第 1–10 页*。

+   Planche, B., Zakharov, S., Wu, Z., Hutter, A., Kosch, H., Ilic, S., 2018\. *超越外观——将真实图像映射到几何领域以进行无监督 CAD 识别. arXiv 预印本 arXiv:1810.04158*。

+   *协议缓冲区 [WWW 文档]，无日期。Google 开发者*. URL：[`developers.google.com/protocol-buffers/`](https://developers.google.com/protocol-buffers/)（访问日期：2019 年 2 月 23 日）。

+   Radford, A., Metz, L., Chintala, S., 2015\. *无监督表示学习与深度卷积生成对抗网络. arXiv:1511.06434 [cs]*。

+   Richter, S.R., Vineet, V., Roth, S., Koltun, V., 2016\. *为数据而玩：来自电脑游戏的真实数据。见：欧洲计算机视觉会议，页 102–118*。

+   Ros, G., Sellart, L., Materzynska, J., Vazquez, D., Lopez, A.M., 2016\. *SYNTHIA 数据集：用于城市场景语义分割的大规模合成图像集合。见：2016 IEEE 计算机视觉与模式识别会议（CVPR）。在 2016 IEEE 计算机视觉与模式识别会议（CVPR）上展示，IEEE，拉斯维加斯，美国，页 3234–3243*。

+   Rozantsev, A., Lepetit, V., Fua, P., 2015\. *为训练物体检测器渲染合成图像。计算机视觉与图像理解 137, 24–37*。

+   Tremblay, J., Prakash, A., Acuna, D., Brophy, M., Jampani, V., Anil, C., To, T., Cameracci, E., Boochoon, S., Birchfield, S., 2018\. *使用合成数据训练深度网络：通过领域随机化弥合现实差距。见：2018 IEEE/CVF 计算机视觉与模式识别研讨会（CVPRW）。在 2018 IEEE/CVF CVPRW 上展示，IEEE，盐湖城，美国，页 1082–10828*。

+   Tzeng, E., Hoffman, J., Saenko, K., Darrell, T., 2017\. *对抗性辨别领域适应。见：2017 IEEE 计算机视觉与模式识别会议（CVPR）。在 2017 IEEE CVPR 上展示，IEEE，檀香山，美国，页 2962–2971*。

+   Zhu, J.-Y., Park, T., Isola, P., Efros, A.A., 2017\. *使用循环一致的对抗网络进行非配对图像到图像的转换。见：IEEE 国际计算机视觉会议论文集，页 2223–2232*。

# 第八章：视频与递归神经网络

+   Britz, D., 2015\. *递归神经网络教程，第三部分 – 时间反向传播与梯度消失。WildML*。

+   Brown, C., 2019\. *用于学习神经网络及相关资料的仓库：go2carter/nn-learn*。

+   *Chung, J., Gulcehre, C., Cho, K., Bengio, Y., 2014\. 门控递归神经网络在序列建模中的实证评估。arXiv:1412.3555 [cs]*。

+   Hochreiter, S., Schmidhuber, J., 1997\. *长短期记忆。神经计算 9, 1735–1780*。

+   Lipton, Z.C., Berkowitz, J., Elkan, C., 2015\. *递归神经网络在序列学习中的关键回顾。arXiv:1506.00019 [cs]*。

+   Soomro, K., Zamir, A.R., Shah, M., 2012\. *UCF101：来自野外视频的 101 个人类动作类别数据集。arXiv:1212.0402 [cs]*。

# 第九章：优化模型并部署到移动设备

+   Goodfellow, I.J., Erhan, D., Carrier, P.L., Courville, A., Mirza, M., Hamner, B., Cukierski, W., Tang, Y., Thaler, D., Lee, D.-H., Zhou, Y., Ramaiah, C., Feng, F., Li, R., Wang, X., Athanasakis, D., Shawe-Taylor, J., Milakov, M., Park, J., Ionescu, R., Popescu, M., Grozea, C., Bergstra, J., Xie, J., Romaszko, L., Xu, B., Chuang, Z., Bengio, Y., 2013\. *表示学习中的挑战：三项机器学习竞赛报告。arXiv:1307.0414 [cs, stat]*。

+   Hinton, G., Vinyals, O., Dean, J.，2015 年。*从神经网络中提取知识。arXiv:1503.02531 [cs, stat]*。

+   Hoff, T.，未注明日期。*苹果照片背后的技术以及深度学习与隐私的未来——高可扩展性*。

+   *腾讯，未注明日期。腾讯/PocketFlow：一个用于开发更小、更快的 AI 应用的自动模型压缩（AutoMC）框架*。
