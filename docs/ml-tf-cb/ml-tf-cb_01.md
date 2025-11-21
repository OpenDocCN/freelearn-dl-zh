

# 第一章：开始使用 TensorFlow 2.x

Google 的 TensorFlow 引擎有一种独特的解决问题方式，使我们能够非常高效地解决机器学习问题。如今，机器学习已应用于几乎所有的生活和工作领域，著名的应用包括计算机视觉、语音识别、语言翻译、医疗保健等。我们将在本书的后续页面中覆盖理解 TensorFlow 操作的基本步骤，并最终讲解如何构建生产代码。此时，本章中呈现的基础内容至关重要，它们将为你提供核心理解，帮助你更好地理解本书中其他部分的食谱。

在本章中，我们将从一些基本的食谱开始，帮助你理解 TensorFlow 2.x 的工作原理。你还将学习如何访问本书中示例所使用的数据，以及如何获取额外的资源。到本章结束时，你应该掌握以下知识：

+   理解 TensorFlow 2.x 的工作原理

+   声明和使用变量与张量

+   与矩阵合作

+   声明操作

+   实现激活函数

+   与数据源合作

+   寻找额外资源

不再多说，让我们从第一个食谱开始，它以简易的方式展示了 TensorFlow 如何处理数据和计算。

# TensorFlow 的工作原理

TensorFlow 最初是由 Google Brain 团队的研究人员和工程师作为一个内部项目开始的，最初名为 **DistBelief**，并于 2015 年 11 月发布为一个开源框架，名称为 TensorFlow（张量是标量、向量、矩阵及更高维度矩阵的广义表示）。你可以在这里阅读关于该项目的原始论文：[`download.tensorflow.org/paper/whitepaper2015.pdf`](http://download.tensorflow.org/paper/whitepaper2015.pdf)。在 2017 年发布 1.0 版本后，去年，Google 发布了 TensorFlow 2.0，它在继续发展和改进 TensorFlow 的同时，使其变得更加用户友好和易于使用。

TensorFlow 是一个面向生产的框架，能够处理不同的计算架构（CPU、GPU，现在还有 TPU），适用于需要高性能和易于分布式的各种计算。它在深度学习领域表现出色，可以创建从浅层网络（由少数层组成的神经网络）到复杂的深度网络，用于图像识别和自然语言处理。

在本书中，我们将呈现一系列食谱，帮助你以更高效的方式使用 TensorFlow 进行深度学习项目，减少复杂性，帮助你实现更广泛的应用并取得更好的结果。

一开始，TensorFlow 中的计算可能看起来不必要地复杂。但这背后是有原因的：由于 TensorFlow 处理计算的方式，当你习惯了 TensorFlow 风格时，开发更复杂的算法会变得相对容易。本方案将引导我们通过 TensorFlow 算法的伪代码。

## 准备开始

目前，TensorFlow 已在以下 64 位系统上测试并获得支持：Ubuntu 16.04 或更高版本、macOS 10.12.6（Sierra）或更高版本（不过不支持 GPU）、Raspbian 9.0 或更高版本，以及 Windows 7 或更高版本。本书中的代码已在 Ubuntu 系统上开发并测试，但它在其他任何系统上也应该能正常运行。本书的代码可在 GitHub 上找到，地址为 [`github.com/PacktPublishing/Machine-Learning-Using-TensorFlow-Cookbook`](https://github.com/PacktPublishing/Machine-Learning-Using-TensorFlow-Cookbook)，它作为本书所有代码和一些数据的代码库。

在本书中，我们将只关注 TensorFlow 的 Python 库封装，尽管 TensorFlow 的大多数核心代码是用 C++ 编写的。TensorFlow 与 Python 很好地兼容，支持 3.7 至 3.8 版本。此书将使用 Python 3.7（你可以在 [`www.python.org`](https://www.python.org) 获取该解释器）和 TensorFlow 2.2.0（你可以在 [`www.tensorflow.org/install`](https://www.tensorflow.org/install) 查找安装它所需的所有说明）。

虽然 TensorFlow 可以在 CPU 上运行，但大多数算法如果在 GPU 上处理，运行会更快，并且支持在具有 Nvidia 计算能力 3.5 或更高版本的显卡上运行（特别是在运行计算密集型的复杂网络时更为推荐）。

你在书中找到的所有方案都与 TensorFlow 2.2.0 兼容。如有必要，我们将指出与以前的 2.1 和 2.0 版本在语法和执行上的区别。

在工作站上运行基于 TensorFlow 的脚本时，常用的 GPU 有 Nvidia Titan RTX 和 Nvidia Quadro RTX，而在数据中心，我们通常会找到至少配备 24 GB 内存的 Nvidia Tesla 架构（例如，Google Cloud Platform 提供了 Nvidia Tesla K80、P4、T4、P100 和 V100 型号）。要在 GPU 上正常运行，你还需要下载并安装 Nvidia CUDA 工具包，版本为 5.x+（[`developer.nvidia.com/cuda-downloads`](https://developer.nvidia.com/cuda-downloads)）。

本章中的一些方案将依赖于安装当前版本的 SciPy、NumPy 和 Scikit-learn Python 包。这些附带包也包含在 Anaconda 包中（https://www.anaconda.com/products/individual#Downloads）。

## 如何进行…

在这里，我们将介绍 TensorFlow 算法的一般流程。大多数方案将遵循这个大纲：

1.  **导入或生成数据集**：我们所有的机器学习算法都依赖于数据集。在本书中，我们将生成数据或使用外部数据源。有时，依赖生成的数据更好，因为我们可以控制如何变化并验证预期结果。大多数情况下，我们将访问给定食谱的公共数据集。有关如何访问这些数据集的详细信息，请参见本章末尾的 *附加资源* 章节：

    ```py
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import numpy as np
    data = tfds.load("iris", split="train") 
    ```

1.  **转换和规范化数据**：通常，输入数据集的形式并不是我们实现目标所需要的精确形式。TensorFlow 期望我们将数据转换为接受的形状和数据类型。实际上，数据通常不符合我们算法所期望的正确维度或类型，我们必须在使用之前正确地转换它。大多数算法还期望规范化数据（这意味着变量的均值为零，标准差为一），我们也将在这里讨论如何实现这一点。TensorFlow 提供了内置函数，可以加载数据、将数据拆分为批次，并允许您使用简单的 NumPy 函数转换变量和规范化每个批次，包括以下内容：

    ```py
    for batch in data.batch(batch_size, drop_remainder=True):
        labels = tf.one_hot(batch['label'], 3)
        X = batch['features']
        X = (X - np.mean(X)) / np.std(X) 
    ```

1.  **将数据集划分为训练集、测试集和验证集**：我们通常希望在不同的数据集上测试我们的算法，这些数据集是我们训练过的。许多算法还需要进行超参数调整，因此我们预留了一个验证集，用于确定最佳的超参数组合。

1.  **设置算法参数（超参数）**：我们的算法通常会有一组参数，这些参数在整个过程中保持不变。例如，这可能是迭代次数、学习率或我们选择的其他固定参数。通常建议将这些参数一起初始化为全局变量，以便读者或用户能够轻松找到它们，如下所示：

    ```py
    epochs = 1000 
    batch_size = 32
    input_size = 4
    output_size = 3
    learning_rate = 0.001 
    ```

1.  **初始化变量**：TensorFlow 依赖于知道它可以修改什么以及不能修改什么。在优化过程中，TensorFlow 将修改/调整变量（模型的权重/偏置），以最小化损失函数。为了实现这一点，我们通过输入变量输入数据。我们需要初始化变量和占位符的大小和类型，以便 TensorFlow 知道该期待什么。TensorFlow 还需要知道期望的数据类型。在本书的大部分内容中，我们将使用 `float32`。TensorFlow 还提供了 `float64` 和 `float16` 数据类型。请注意，使用更多字节来获得更高精度会导致算法变慢，而使用更少字节则会导致结果算法的精度降低。请参考以下代码，了解如何在 TensorFlow 中设置一个权重数组和一个偏置向量的简单示例：

    ```py
    weights = tf.Variable(tf.random.normal(shape=(input_size, 
                                                  output_size), 
                                            dtype=tf.float32))
    biases  = tf.Variable(tf.random.normal(shape=(output_size,), 
                                           dtype=tf.float32)) 
    ```

1.  **定义模型结构**：在获得数据并初始化变量之后，我们必须定义模型。这是通过构建计算图来完成的。这个示例中的模型将是一个逻辑回归模型（logit `E`(`Y`) = b`X` + a）：

    ```py
    logits = tf.add(tf.matmul(X, weights), biases) 
    ```

1.  **声明损失函数**：在定义模型之后，我们必须能够评估输出。这就是我们声明损失函数的地方。损失函数非常重要，因为它告诉我们预测值与实际值之间的偏差。不同类型的损失函数将在 *第二章*，*TensorFlow 实践方式* 中的 *实现反向传播* 这一章节中详细探讨。在此，我们以交叉熵为例，使用 logits 计算 softmax 交叉熵与标签之间的差异：

    ```py
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels, logits)) 
    ```

1.  **初始化并训练模型**：现在我们已经准备好了一切，需要创建图的实例，输入数据，并让 TensorFlow 调整变量，以更好地预测我们的训练数据。以下是初始化计算图的一种方法，通过多次迭代，使用 SDG 优化器收敛模型结构中的权重：

    ```py
    optimizer = tf.optimizers.SGD(learning_rate)
    with tf.GradientTape() as tape:
       logits = tf.add(tf.matmul(X, weights), biases)
       loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels, logits))
    gradients = tape.gradient(loss, [weights, biases])
    optimizer.apply_gradients(zip(gradients, [weights, biases])) 
    ```

1.  **评估模型**：一旦我们建立并训练了模型，我们应该通过某些指定的标准评估模型，看看它在新数据上的表现如何。我们会在训练集和测试集上进行评估，这些评估将帮助我们判断模型是否存在过拟合或欠拟合的问题。我们将在后续的内容中讨论这个问题。在这个简单的例子中，我们评估最终的损失，并将拟合值与真实的训练值进行比较：

    ```py
    print(f"final loss is: {loss.numpy():.3f}")
    preds = tf.math.argmax(tf.add(tf.matmul(X, weights), biases), axis=1)
    ground_truth = tf.math.argmax(labels, axis=1)
    for y_true, y_pred in zip(ground_truth.numpy(), preds.numpy()):
        print(f"real label: {y_true} fitted: {y_pred}") 
    ```

1.  **调整超参数**：大多数时候，我们会希望回到之前的步骤，调整一些超参数，并根据我们的测试结果检查模型的性能。然后，我们使用不同的超参数重复之前的步骤，并在验证集上评估模型。

1.  **部署/预测新结果**：了解如何对新数据和未见过的数据进行预测也是一个关键要求。一旦我们训练好模型，就可以通过 TensorFlow 轻松实现这一点。

## 它是如何工作的……

在 TensorFlow 中，我们必须先设置数据、输入变量和模型结构，然后才能告诉程序训练并调整其权重，以提高预测效果。TensorFlow 通过计算图来完成这项工作。计算图是没有递归的有向图，允许并行计算。

为此，我们需要创建一个损失函数，以便 TensorFlow 最小化它。TensorFlow 通过修改计算图中的变量来实现这一点。TensorFlow 能够修改变量，因为它跟踪模型中的计算，并自动计算变量的梯度（如何改变每个变量），以最小化损失。因此，我们可以看到，进行更改并尝试不同数据源是多么容易。

## 另见

+   有关 TensorFlow 的进一步介绍以及更多资源，请参考 TensorFlow 官方页面上的官方文档和教程：[`www.tensorflow.org/`](https://www.tensorflow.org/)

+   在官方页面中，一个更具百科全书性质的入门位置是官方 Python API 文档，[`www.tensorflow.org/api_docs/python/`](https://www.tensorflow.org/api_docs/python/)，在那里您可以找到所有可能的命令列表。

+   还有教程可供学习：[`www.tensorflow.org/tutorials/`](https://www.tensorflow.org/tutorials/)

+   除此之外，还可以在这里找到一个非官方的 TensorFlow 教程、项目、演示和代码库集合：[`github.com/dragen1860/TensorFlow-2.x-Tutorials`](https://github.com/dragen1860/TensorFlow-2.x-Tutorials )

# 声明变量和张量

张量是 TensorFlow 用于在计算图上进行操作的主要数据结构。即使在 TensorFlow 2.x 中，这一方面被隐藏了，但数据流图仍然在幕后运行。这意味着构建神经网络的逻辑在 TensorFlow 1.x 和 TensorFlow 2.x 之间并没有发生太大变化。最引人注目的变化是，您不再需要处理占位符，后者是 TensorFlow 1.x 图中数据的输入门。

现在，您只需将张量声明为变量，然后继续构建图。

*张量* 是一个数学术语，指的是广义的向量或矩阵。如果向量是一维的，矩阵是二维的，那么张量就是 `n` 维的（其中 `n` 可以是 1、2 或更大）。

我们可以将这些张量声明为变量，并将它们用于计算。为了做到这一点，我们首先需要学习如何创建张量。

## 准备工作

当我们创建一个张量并将其声明为变量时，TensorFlow 会在我们的计算图中创建多个图结构。还需要指出的是，仅仅创建一个张量并不会向计算图中添加任何内容。TensorFlow 仅在执行操作以初始化变量后才会这样做。有关更多信息，请参阅下节关于变量和占位符的内容。

## 如何做到这一点…

在这里，我们将介绍在 TensorFlow 中创建张量的四种主要方式。

在本食谱或其他食谱中，我们不会进行不必要的详细说明。我们倾向于仅说明不同 API 调用中的必需参数，除非您认为覆盖某些可选参数对食谱有帮助；当这种情况发生时，我们会说明其背后的理由。

1.  固定大小的张量：

    +   在以下代码中，我们创建了一个全为 0 的张量：

    ```py
    row_dim, col_dim = 3, 3
    zero_tsr = tf.zeros(shape=[row_dim, col_dim], dtype=tf.float32) 
    ```

    +   在以下代码中，我们创建了一个全为 1 的张量：

    ```py
    ones_tsr = tf.ones([row_dim, col_dim]) 
    ```

    +   在以下代码中，我们创建了一个常量填充的张量：

    ```py
    filled_tsr = tf.fill([row_dim, col_dim], 42) 
    ```

    +   在以下代码中，我们从一个现有常量创建了一个张量：

    ```py
    constant_tsr = tf.constant([1,2,3]) 
    ```

    请注意，`tf.constant()` 函数可以用来将一个值广播到数组中，通过写 `tf.constant(42, [row_dim, col_dim])` 来模仿 `tf.fill()` 的行为。

1.  **相似形状的张量**：我们也可以根据其他张量的形状初始化变量，如下所示：

    ```py
    zeros_similar = tf.zeros_like(constant_tsr) 
    ones_similar = tf.ones_like(constant_tsr) 
    ```

    请注意，由于这些张量依赖于先前的张量，我们必须按顺序初始化它们。尝试以随机顺序初始化张量会导致错误。

1.  **序列张量**：在 TensorFlow 中，所有的参数都被文档化为张量。即使需要标量，API 也会将其作为零维标量提及。因此，TensorFlow 允许我们指定包含定义区间的张量也就不足为奇了。以下函数的行为与 NumPy 的`linspace()`输出和`range()`输出非常相似（参考： [`docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html)）。请看以下函数：

    ```py
    linear_tsr = tf.linspace(start=0.0, stop=1.0, num=3) 
    ```

    请注意，起始值和终止值参数应为浮点值，而`num`应为整数。

    结果张量的序列为[0.0, 0.5, 1.0]（`print(linear_tsr`命令将提供必要的输出）。请注意，此函数包括指定的终止值。以下是`tf.range`函数的对比：

    ```py
    integer_seq_tsr = tf.range(start=6, limit=15, delta=3) 
    ```

    结果是序列[6, 9, 12]。请注意，此函数不包括限制值，并且可以处理起始值和限制值的整数和浮点数。

1.  **随机张量**：以下生成的随机数来自均匀分布：

    ```py
    randunif_tsr = tf.random.uniform([row_dim, col_dim], 
                                     minval=0, maxval=1) 
    ```

请注意，这种随机均匀分布是从包含`minval`但不包含`maxval`的区间中抽取的（`minval <= x < maxval`）。因此，在这种情况下，输出范围是[0, 1)。如果你需要仅抽取整数而不是浮点数，只需在调用函数时添加`dtype=tf.int32`参数。

若要获得从正态分布中随机抽取的张量，可以运行以下代码：

```py
randnorm_tsr = tf.random.normal([row_dim, col_dim], 
                                 mean=0.0, stddev=1.0) 
```

还有一些情况，我们希望生成在某些范围内保证的正态分布随机值。`truncated_normal()`函数总是从指定均值的两个标准差内挑选正态值：

```py
runcnorm_tsr = tf.random.truncated_normal([row_dim, col_dim], 
                                          mean=0.0, stddev=1.0) 
```

我们可能还对随机化数组的条目感兴趣。为此，两个函数可以帮助我们：`random.shuffle()`和`image.random_crop()`。以下代码执行此操作：

```py
shuffled_output = tf.random.shuffle(input_tensor) 
cropped_output = tf.image.random_crop(input_tensor, crop_size) 
```

在本书后续的内容中，我们将关注对大小为（高度，宽度，3）的图像进行随机裁剪，其中包含三种颜色光谱。为了在`cropped_output`中固定某一维度，你必须为该维度指定最大值：

```py
height, width = (64, 64)
my_image = tf.random.uniform([height, width, 3], minval=0,
         maxval=255, dtype=tf.int32)
cropped_image = tf.image.random_crop(my_image, 
       [height//2, width//2, 3]) 
```

这段代码将生成随机噪声图像，这些图像会被裁剪，既能减小高度和宽度的一半，但深度维度不会受影响，因为你已将其最大值固定为参数。

## 它是如何工作的…

一旦我们决定了如何创建张量，我们还可以通过将张量包装在`Variable()`函数中来创建相应的变量，如下所示：

```py
my_var = tf.Variable(tf.zeros([row_dim, col_dim])) 
```

在接下来的示例中会有更多内容介绍。

## 还有更多…

我们不局限于内置函数：我们可以将任何 NumPy 数组转换为 Python 列表，或使用 `convert_to_tensor()` 函数将常量转换为张量。注意，这个函数也接受张量作为输入，以便我们希望在函数内对计算进行通用化时使用。

# 使用急切执行

在开发深度且复杂的神经网络时，你需要不断地尝试不同的架构和数据。这在 TensorFlow 1.0 中是困难的，因为你总是需要从头到尾运行代码，以检查是否成功。TensorFlow 2.x 默认在急切执行模式下工作，这意味着你可以随着项目的进展，逐步开发并检查代码。这是个好消息；现在我们只需要理解如何在急切执行模式下进行实验，这样我们就能充分利用 TensorFlow 2.x 的这一特性。本教程将为你提供入门的基础知识。

## 准备开始

TensorFlow 1.x 的表现最优，因为它在编译静态计算图之后执行计算。所有计算都被分配并连接成一个图，你编译网络时，该图帮助 TensorFlow 执行计算，利用可用资源（多核 CPU 或多个 GPU）以最佳方式，并在资源之间以最及时高效的方式分配操作。这也意味着，无论如何，一旦你定义并编译了图，就不能在运行时对其进行更改，而必须从头开始实例化，这会带来额外的工作量。

在 TensorFlow 2.x 中，你仍然可以定义你的网络，编译并以最佳方式运行它，但 TensorFlow 开发团队现在默认采用了一种更为实验性的方式，允许立即评估操作，从而使调试更容易，尝试网络变种也更加方便。这就是所谓的急切执行。现在，操作返回的是具体的值，而不是指向稍后构建的计算图部分的指针。更重要的是，你现在可以在模型执行时使用主机语言的所有功能，这使得编写更复杂、更精细的深度学习解决方案变得更容易。

## 如何做……

你基本上不需要做任何事情；在 TensorFlow 2.x 中，**急切执行（eager execution）**是默认的操作方式。当你导入 TensorFlow 并开始使用它的功能时，你会在急切执行模式下工作，因为你可以在执行时进行检查：

```py
tf.executing_eagerly()
True 
```

这就是你需要做的全部。

## 它是如何工作的……

只需运行 TensorFlow 操作，结果将立即返回：

```py
x = [[2.]]
m = tf.matmul(x, x)
print("the result is {}".format(m))
the result is [[4.]] 
```

就是这么简单！

## 还有更多…

由于 TensorFlow 现在默认启用了即时执行模式，你不会惊讶地发现 `tf.Session` 已经从 TensorFlow API 中移除。你不再需要在运行计算之前构建计算图；现在你只需构建你的网络，并在过程中进行测试。这为常见的软件最佳实践打开了道路，比如文档化代码、在编写代码时使用面向对象编程，以及将代码组织成可重用的自包含模块。

# 与矩阵的操作

理解 TensorFlow 如何与矩阵协作在开发数据流通过计算图时非常重要。在这个方案中，我们将涵盖矩阵的创建以及可以使用 TensorFlow 执行的基本操作。

值得强调的是，矩阵在机器学习（以及数学中的一般应用）中的重要性：机器学习算法在计算上是通过矩阵运算来表示的。了解如何进行矩阵运算是使用 TensorFlow 时的一个加分项，尽管你可能不需要经常使用它；其高级模块 Keras 可以在后台处理大多数矩阵代数内容（更多关于 Keras 的内容请参见*第三章*，*Keras*）。

本书不涵盖矩阵属性和矩阵代数（线性代数）的数学背景，因此不熟悉的读者强烈建议学习足够的矩阵知识，以便能够熟练掌握矩阵代数。在*另见*部分，你可以找到一些资源，帮助你复习微积分技巧或从零开始学习，以便从 TensorFlow 中获得更多收益。

## 准备工作

许多算法依赖于矩阵运算。TensorFlow 为我们提供了易于使用的操作来执行这些矩阵计算。你只需要导入 TensorFlow，并按照本节内容进行操作；如果你不是矩阵代数专家，请首先查看本方案的*另见*部分，寻找帮助你充分理解下面方案的资源。

## 如何操作…

我们按如下步骤进行：

1.  **创建矩阵**：我们可以从 NumPy 数组或嵌套列表创建二维矩阵，正如本章开头的*声明和使用变量与张量*方案中所描述的那样。我们可以使用张量创建函数，并为如 `zeros()`、`ones()` 和 `truncated_normal()` 等函数指定二维形状。TensorFlow 还允许我们使用 `diag()` 函数从一维数组或列表创建对角矩阵，示例如下：

    ```py
    identity_matrix = tf.linalg.diag([1.0, 1.0, 1.0]) 
    A = tf.random.truncated_normal([2, 3]) 
    B = tf.fill([2,3], 5.0) 
    C = tf.random.uniform([3,2]) 
    D = tf.convert_to_tensor(np.array([[1., 2., 3.],
                                       [-3., -7., -1.],
                                       [0., 5., -2.]]), 
                             dtype=tf.float32) 
    print(identity_matrix)

    [[ 1\.  0\.  0.] 
     [ 0\.  1\.  0.] 
     [ 0\.  0\.  1.]] 
    print(A) 
    [[ 0.96751703  0.11397751 -0.3438891 ] 
     [-0.10132604 -0.8432678   0.29810596]] 
    print(B) 
    [[ 5\.  5\.  5.] 
     [ 5\.  5\.  5.]] 
    print(C)
    [[ 0.33184157  0.08907614] 
     [ 0.53189191  0.67605299] 
     [ 0.95889051 0.67061249]] 
    ```

    请注意，C 张量是随机创建的，它可能与你的会话中显示的内容有所不同。

    ```py
    print(D) 
    [[ 1\.  2\.  3.] 
     [-3\. -7\. -1.] 
     [ 0\.  5\. -2.]] 
    ```

1.  **加法、减法和乘法**：要对相同维度的矩阵进行加法、减法或乘法，TensorFlow 使用以下函数：

    ```py
    print(A+B) 
    [[ 4.61596632  5.39771316  4.4325695 ] 
     [ 3.26702736  5.14477345  4.98265553]] 
    print(B-B)
    [[ 0\.  0\.  0.] 
     [ 0\.  0\.  0.]] 
    print(tf.matmul(B, identity_matrix)) 
    [[ 5\.  5\.  5.] 
     [ 5\.  5\.  5.]] 
    ```

    需要注意的是，`matmul()`函数有一些参数，用于指定是否在乘法前转置参数（布尔参数`transpose_a`和`transpose_b`），或者每个矩阵是否为稀疏矩阵（`a_is_sparse`和`b_is_sparse`）。

    如果你需要对两个形状和类型相同的矩阵进行逐元素乘法（这非常重要，否则会报错），只需使用`tf.multiply`函数：

    ```py
    print(tf.multiply(D, identity_matrix))
    [[ 1\.  0\.  0.] 
     [-0\. -7\. -0.] 
     [ 0\.  0\. -2.]] 
    ```

    请注意，矩阵除法没有明确定义。虽然许多人将矩阵除法定义为乘以逆矩阵，但它在本质上不同于实数除法。

1.  **转置**：转置矩阵（翻转列和行），如下所示：

    ```py
    print(tf.transpose(C)) 
    [[0.33184157 0.53189191 0.95889051]
     [0.08907614 0.67605299 0.67061249]] 
    ```

    再次提到，重新初始化会给我们不同于之前的值。

1.  **行列式**：要计算行列式，请使用以下代码：

    ```py
    print(tf.linalg.det(D))
    -38.0 
    ```

1.  **逆矩阵**：要找到一个方阵的逆矩阵，请参阅以下内容：

    ```py
    print(tf.linalg.inv(D))
    [[-0.5        -0.5        -0.5       ] 
     [ 0.15789474  0.05263158  0.21052632] 
     [ 0.39473684  0.13157895  0.02631579]] 
    ```

    逆矩阵方法仅在矩阵是对称正定时，基于 Cholesky 分解。如果矩阵不是对称正定的，则基于 LU 分解。

1.  **分解**：对于 Cholesky 分解，请使用以下代码：

    ```py
    print(tf.linalg.cholesky(identity_matrix))
    [[ 1\.  0\.  1.] 
     [ 0\.  1\.  0.] 
     [ 0\.  0\.  1.]] 
    ```

1.  **特征值和特征向量**：对于特征值和特征向量，请使用以下代码：

    ```py
    print(tf.linalg.eigh(D))
    [[-10.65907521  -0.22750691   2.88658212] 
     [  0.21749542   0.63250104  -0.74339638] 
     [  0.84526515   0.2587998    0.46749277] 
     [ -0.4880805    0.73004459   0.47834331]] 
    ```

请注意，`tf.linalg.eigh()`函数输出两个张量：第一个张量包含**特征值**，第二个张量包含**特征向量**。在数学中，这种操作称为矩阵的**特征分解**。

## 它的工作原理……

TensorFlow 为我们提供了所有开始进行数值计算的工具，并将这些计算添加到我们的神经网络中。

## 另见

如果你需要快速提升微积分技能，并深入了解更多关于 TensorFlow 操作的内容，我们建议以下资源：

+   这本免费的书籍《机器学习数学》（Mathematics for Machine Learning），可以在这里找到：[`mml-book.github.io/`](https://mml-book.github.io/)。如果你想在机器学习领域中成功操作，这本书包含了你需要知道的一切。

+   对于一个更加易于获取的资源，可以观看 Khan Academy 的关于向量和矩阵的课程（[`www.khanacademy.org/math/precalculus`](https://www.khanacademy.org/math/precalculus)），以便学习神经网络中最基本的数据元素。

# 声明操作

除了矩阵操作，TensorFlow 还有许多其他操作我们至少应该了解。这个教程将为你提供一个简要且必要的概览，帮助你掌握真正需要知道的内容。

## 准备好

除了标准的算术操作外，TensorFlow 还为我们提供了更多需要了解的操作。在继续之前，我们应该认识到这些操作并学习如何使用它们。再次提醒，我们只需要导入 TensorFlow：

```py
import tensorflow as tf 
```

现在我们准备运行接下来的代码。

## 如何操作……

TensorFlow 提供了张量的标准操作，即`add()`、`subtract()`、`multiply()`和`division()`，它们都位于`math`模块中。请注意，本节中的所有操作，除非另有说明，否则都将逐元素计算输入：

1.  TensorFlow 还提供了`division()`及相关函数的变种。

1.  值得注意的是，`division()`返回与输入相同类型的结果。这意味着如果输入是整数，它实际上返回除法的地板值（类似 Python 2）。要返回 Python 3 版本的除法，即在除法前将整数转换为浮点数并始终返回浮点数，TensorFlow 提供了`truediv()`函数，具体如下：

    ```py
    print(tf.math.divide(3, 4))
    0.75 
    print(tf.math.truediv(3, 4)) 
    tf.Tensor(0.75, shape=(), dtype=float64) 
    ```

1.  如果我们有浮点数并且需要整数除法，可以使用`floordiv()`函数。请注意，这仍然会返回浮点数，但它会向下舍入到最接近的整数。该函数如下：

    ```py
    print(tf.math.floordiv(3.0,4.0)) 
    tf.Tensor(0.0, shape=(), dtype=float32) 
    ```

1.  另一个重要的函数是`mod()`。该函数返回除法后的余数，具体如下：

    ```py
    print(tf.math.mod(22.0, 5.0))
    tf.Tensor(2.0, shape=(), dtype=float32) 
    ```

1.  两个张量的叉积通过`cross()`函数实现。请记住，叉积仅对两个三维向量定义，因此它只接受两个三维张量。以下代码展示了这一用法：

    ```py
    print(tf.linalg.cross([1., 0., 0.], [0., 1., 0.]))
    tf.Tensor([0\. 0\. 1.], shape=(3,), dtype=float32) 
    ```

1.  下面是常用数学函数的简明列表。所有这些函数均逐元素操作：

    | 函数 | 操作 |
    | --- | --- |
    | `tf.math.abs()` | 输入张量的绝对值 |
    | `tf.math.ceil()` | 输入张量的向上取整函数 |
    | `tf.math.cos()` | 输入张量的余弦函数 |
    | `tf.math.exp()` | 输入张量的底数 `e` 指数函数 |
    | `tf.math.floor()` | 输入张量的向下取整函数 |
    | `tf.linalg.inv()` | 输入张量的乘法逆（1/x） |
    | `tf.math.log()` | 输入张量的自然对数 |
    | `tf.math.maximum()` | 两个张量的逐元素最大值 |
    | `tf.math.minimum()` | 两个张量的逐元素最小值 |
    | `tf.math.negative()` | 输入张量的负值 |
    | `tf.math.pow()` | 第一个张量按逐元素方式升至第二个张量 |
    | `tf.math.round()` | 对输入张量进行四舍五入 |
    | `tf.math.rsqrt()` | 一个张量的平方根倒数 |
    | `tf.math.sign()` | 根据张量的符号返回 -1、0 或 1 |
    | `tf.math.sin()` | 输入张量的正弦函数 |
    | `tf.math.sqrt()` | 输入张量的平方根 |
    | `tf.math.square()` | 输入张量的平方 |

1.  **特殊数学函数**：有一些在机器学习中经常使用的特殊数学函数值得一提，TensorFlow 为它们提供了内建函数。再次强调，除非另有说明，否则这些函数都是逐元素操作：

| `tf.math.digamma()` | Psi 函数，即`lgamma()`函数的导数 |
| --- | --- |
| `tf.math.erf()` | 一个张量的高斯误差函数（逐元素） |
| `tf.math.erfc()` | 一个张量的互补误差函数 |
| `tf.math.igamma()` | 下正则化不完全伽马函数 |
| `tf.math.igammac()` | 上不完全伽马函数的正则化形式 |
| `tf.math.lbeta()` | beta 函数绝对值的自然对数 |
| `tf.math.lgamma()` | 伽马函数绝对值的自然对数 |
| `tf.math.squared_difference()` | 计算两个张量之间差值的平方 |

## 它是如何工作的……

了解哪些函数对我们可用是很重要的，这样我们才能将它们添加到我们的计算图中。我们将主要关注前面提到的函数。我们也可以通过组合这些函数生成许多不同的自定义函数，如下所示：

```py
# Tangent function (tan(pi/4)=1) 
def pi_tan(x):
    return tf.tan(3.1416/x)
print(pi_tan(4))
tf.Tensor(1.0000036, shape=(), dtype=float32) 
```

组成深度神经网络的复杂层仅由前面的函数组成，因此，凭借这个教程，你已经掌握了创建任何你想要的内容所需的所有基础知识。

## 还有更多……

如果我们希望向图中添加其他未列出的操作，我们必须从前面的函数中创建自己的操作。下面是一个示例，这是之前未使用过的操作，我们可以将其添加到图中。我们可以使用以下代码添加一个自定义的多项式函数，*3 * x² - x + 10*：

```py
def custom_polynomial(value): 
    return tf.math.subtract(3 * tf.math.square(value), value) + 10
print(custom_polynomial(11))
tf.Tensor(362, shape=(), dtype=int32) 
```

现在，你可以创建无限制的自定义函数，不过我始终建议你首先查阅 TensorFlow 文档。通常，你不需要重新发明轮子；你会发现你需要的功能已经被编码实现了。

# 实现激活函数

激活函数是神经网络逼近非线性输出并适应非线性特征的关键。它们在神经网络中引入非线性操作。如果我们小心选择激活函数并合理放置它们，它们是非常强大的操作，可以指示 TensorFlow 进行拟合和优化。

## 准备就绪

当我们开始使用神经网络时，我们会经常使用激活函数，因为激活函数是任何神经网络的重要组成部分。激活函数的目标就是调整权重和偏置。在 TensorFlow 中，激活函数是对张量进行的非线性操作。它们的作用类似于之前的数学操作。激活函数有很多用途，但主要的概念是它们在图中引入非线性，同时对输出进行归一化。

## 如何实现……

激活函数位于 TensorFlow 的 **神经网络** (**nn**) 库中。除了使用内置的激活函数外，我们还可以使用 TensorFlow 操作设计自己的激活函数。我们可以导入预定义的激活函数（通过 `tensorflow import nn`），或者在函数调用中明确写出 `nn`。在这里，我们选择对每个函数调用显式声明：

1.  修正线性单元（ReLU）是最常见和最基本的方式，用于在神经网络中引入非线性。这个函数就叫做 `max(0,x)`。它是连续的，但不光滑。它的形式如下：

    ```py
    print(tf.nn.relu([-3., 3., 10.]))
    tf.Tensor([ 0\.  3\. 10.], shape=(3,), dtype=float32) 
    ```

1.  有时我们希望限制前面 ReLU 激活函数的线性增涨部分。我们可以通过将 `max(0,x)` 函数嵌套在 `min()` 函数中来实现。TensorFlow 实现的版本被称为 ReLU6 函数，定义为 `min(max(0,x),6)`。这是一个硬 sigmoid 函数的版本，计算速度更快，并且不容易遇到梯度消失（趋近于零）或梯度爆炸的问题。这在我们后续讨论卷积神经网络和递归神经网络时会非常有用。它的形式如下：

    ```py
    print(tf.nn.relu6([-3., 3., 10.]))
    tf.Tensor([ 0\.  3\. 6.], shape=(3,), dtype=float32) 
    ```

1.  Sigmoid 函数是最常见的连续且平滑的激活函数。它也被称为 logistic 函数，形式为 *1 / (1 + exp(-x))*。由于在训练过程中容易导致反向传播的梯度消失，sigmoid 函数并不常用。它的形式如下：

    ```py
    print(tf.nn.sigmoid([-1., 0., 1.]))
    tf.Tensor([0.26894143 0.5 0.7310586 ], shape=(3,), dtype=float32) 
    ```

    我们应该意识到一些激活函数，如 sigmoid，并不是零中心的。这将要求我们在使用这些函数前对数据进行零均值化处理，特别是在大多数计算图算法中。

1.  另一个平滑的激活函数是双曲正切函数。双曲正切函数与 sigmoid 函数非常相似，只不过它的范围不是 0 到 1，而是 -1 到 1。这个函数的形式是双曲正弦与双曲余弦的比值。另一种写法如下：

    ```py
    ((exp(x) – exp(-x))/(exp(x) + exp(-x)) 
    ```

    这个激活函数如下：

    ```py
    print(tf.nn.tanh([-1., 0., 1.]))
    tf.Tensor([-0.7615942  0\. 0.7615942], shape=(3,), dtype=float32) 
    ```

1.  `softsign` 函数也被用作激活函数。这个函数的形式是 *x/(|x| + 1)*。`softsign` 函数应该是符号函数的一个连续（但不光滑）近似。见以下代码：

    ```py
    print(tf.nn.softsign([-1., 0., -1.]))
    tf.Tensor([-0.5  0\.  -0.5], shape=(3,), dtype=float32) 
    ```

1.  另一个函数是 `softplus` 函数，它是 ReLU 函数的平滑版本。这个函数的形式是 *log(exp(x) + 1)*。它的形式如下：

    ```py
    print(tf.nn.softplus([-1., 0., -1.]))
    tf.Tensor([0.31326166 0.6931472  0.31326166], shape=(3,), dtype=float32) 
    ```

    `softplus` 函数随着输入的增大趋于无穷大，而 `softsign` 函数则趋向 1。不过，随着输入变小，`softplus` 函数接近零，而 `softsign` 函数则趋向 -1。

1.  **指数线性单元** (**ELU**) 与 softplus 函数非常相似，只不过它的下渐近线是 -1，而不是 0。其形式为 *（exp(x) + 1)*，当 *x < 0* 时；否则为 `x`。它的形式如下：

    ```py
    print(tf.nn.elu([-1., 0., -1.])) 
    tf.Tensor([-0.63212055  0\. -0.63212055], shape=(3,), dtype=float32) 
    ```

1.  现在，通过这个公式，你应该能理解基本的关键激活函数。我们列出的现有激活函数并不全面，你可能会发现对于某些问题，你需要尝试其中一些不太常见的函数。除了这个公式中的激活函数，你还可以在 Keras 激活函数页面上找到更多激活函数：[`www.tensorflow.org/api_docs/python/tf/keras/activations`](https://www.tensorflow.org/api_docs/python/tf/keras/activations)

## 它的工作原理…

这些激活函数是我们未来可以在神经网络或其他计算图中引入非线性的方法。需要注意的是，我们在网络中的哪个位置使用了激活函数。如果激活函数的值域在 0 和 1 之间（如 sigmoid），那么计算图只能输出 0 到 1 之间的值。如果激活函数位于节点之间并被隐藏，那么我们需要注意这个范围对张量的影响，特别是在通过张量时。如果我们的张量被缩放为零均值，我们将希望使用一个能够尽可能保持零附近方差的激活函数。

这意味着我们希望选择一个激活函数，比如**双曲正切**（**tanh**）或**softsign**。如果张量都被缩放为正数，那么我们理想中会选择一个能够保持正域方差的激活函数。

## 还有更多…

我们甚至可以轻松创建自定义的激活函数，如 Swish，公式为 `x`*sigmoid(`x`)*（参见 *Swish: a Self-Gated Activation Function*, Ramachandran 等，2017，[`arxiv.org/abs/1710.05941`](https://arxiv.org/abs/1710.05941)），它可以作为 ReLU 激活函数在图像和表格数据问题中的一个更高效的替代品：

```py
def swish(x):
    return x * tf.nn.sigmoid(x)
print(swish([-1., 0., 1.]))
tf.Tensor([-0.26894143  0\.  0.7310586 ], shape=(3,), dtype=float32) 
```

在尝试过 TensorFlow 提供的激活函数后，你的下一步自然是复制那些你在深度学习论文中找到的激活函数，或者你自己创建的激活函数。

# 处理数据源

本书的大部分内容都将依赖于使用数据集来训练机器学习算法。本节提供了如何通过 TensorFlow 和 Python 访问这些数据集的说明。

一些数据源依赖于外部网站的维护，以便你能够访问数据。如果这些网站更改或删除了数据，那么本节中的部分代码可能需要更新。你可以在本书的 GitHub 页面上找到更新后的代码：

[`github.com/PacktPublishing/Machine-Learning-Using-TensorFlow-Cookbook`](https://github.com/PacktPublishing/Machine-Learning-Using-TensorFlow-Cookbook)

## 准备就绪

在本书中，我们将使用的大多数数据集可以通过 TensorFlow 数据集（TensorFlow Datasets）访问，而一些其他的数据集则需要额外的努力，可能需要使用 Python 脚本来下载，或者通过互联网手动下载。

**TensorFlow 数据集**（**TFDS**）是一个现成可用的数据集集合（完整列表可以在此处找到：[`www.tensorflow.org/datasets/catalog/overview`](https://www.tensorflow.org/datasets/catalog/overview)）。它自动处理数据的下载和准备，并且作为 `tf.data` 的封装器，构建高效且快速的数据管道。

为了安装 TFDS，只需在控制台中运行以下安装命令：

```py
pip install tensorflow-datasets 
```

现在，我们可以继续探索本书中你将使用的核心数据集（并非所有这些数据集都会包含在内，只有最常见的几个数据集会被介绍，其他一些非常特定的数据集将在本书的不同章节中介绍）。

## 如何操作…

1.  **鸢尾花数据集**：这个数据集可以说是机器学习中经典的结构化数据集，可能也是所有统计学示例中的经典数据集。它是一个测量三种不同类型鸢尾花的萼片长度、萼片宽度、花瓣长度和花瓣宽度的数据集：*Iris setosa*、*Iris virginica* 和 *Iris versicolor*。总共有 150 个测量值，这意味着每个物种有 50 个测量值。要在 Python 中加载该数据集，我们将使用 TFDS 函数，代码如下：

    ```py
    import tensorflow_datasets as tfds
    iris = tfds.load('iris', split='train') 
    ```

    当你第一次导入数据集时，下载数据集时会显示一个进度条，指示你所在的位置。如果你不想看到进度条，可以通过输入以下代码来禁用它：

    `tfds.disable_progress_bar()`

1.  **出生体重数据**：该数据最初来自 1986 年 Baystate 医疗中心（马萨诸塞州斯普林菲尔德）。该数据集包含了出生体重和母亲的其他人口统计学及医学测量数据，以及家庭病史的记录。数据集有 189 条记录，包含 11 个变量。以下代码展示了如何将该数据作为`tf.data.dataset`来访问：

    ```py
    import tensorflow_datasets as tfds
    birthdata_url = 'https://raw.githubusercontent.com/PacktPublishing/TensorFlow-2-Machine-Learning-Cookbook-Third-Edition/master/birthweight.dat' 
    path = tf.keras.utils.get_file(birthdata_url.split("/")[-1], birthdata_url)
    def map_line(x):
        return tf.strings.to_number(tf.strings.split(x))
    birth_file = (tf.data
                  .TextLineDataset(path)
                  .skip(1)     # Skip first header line
                  .map(map_line)
                 ) 
    ```

1.  **波士顿房价数据集**：卡内基梅隆大学在其`StatLib`库中维护了一系列数据集。该数据可以通过加州大学欧文分校的机器学习仓库轻松访问（[`archive.ics.uci.edu/ml/index.php`](https://archive.ics.uci.edu/ml/index.php)）。该数据集包含 506 条房价观察记录，以及各种人口统计数据和房屋属性（14 个变量）。以下代码展示了如何在 TensorFlow 中访问该数据：

    ```py
    import tensorflow_datasets as tfds
    housing_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
    path = tf.keras.utils.get_file(housing_url.split("/")[-1], housing_url)
    def map_line(x):
        return tf.strings.to_number(tf.strings.split(x))
    housing = (tf.data
               .TextLineDataset(path)
               .map(map_line)
              ) 
    ```

1.  **MNIST 手写数字数据集**：**美国国家标准与技术研究院**（**NIST**）的手写数据集的子集即为**MNIST**数据集。MNIST 手写数字数据集托管在 Yann LeCun 的网站上（[`yann.lecun.com/exdb/mnist/`](http://yann.lecun.com/exdb/mnist/)）。该数据库包含 70,000 张单个数字（0-9）的图像，其中大约 60,000 张用于训练集，10,000 张用于测试集。这个数据集在图像识别中使用非常频繁，以至于 TensorFlow 提供了内置函数来访问该数据。在机器学习中，提供验证数据以防止过拟合（目标泄露）也是很重要的。因此，TensorFlow 将训练集中的 5,000 张图像分配为验证集。以下代码展示了如何在 TensorFlow 中访问此数据：

    ```py
    import tensorflow_datasets as tfds
    mnist = tfds.load('mnist', split=None)
    mnist_train = mnist['train']
    mnist_test = mnist['test'] 
    ```

1.  **垃圾邮件-正常邮件文本数据**。UCI 的机器学习数据集库也包含了一个垃圾邮件-正常邮件文本数据集。我们可以访问这个`.zip`文件并获取垃圾邮件-正常邮件文本数据，方法如下：

    ```py
    import tensorflow_datasets as tfds
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    path = tf.keras.utils.get_file(zip_url.split("/")[-1], zip_url, extract=True)
    path = path.replace("smsspamcollection.zip", "SMSSpamCollection")
    def split_text(x):
        return tf.strings.split(x, sep='\t')
    text_data = (tf.data
                 .TextLineDataset(path)
                 .map(split_text)
                ) 
    ```

1.  **电影评论数据**：康奈尔大学的 Bo Pang 发布了一个电影评论数据集，将评论分类为好或坏。你可以在康奈尔大学网站上找到该数据：[`www.cs.cornell.edu/people/pabo/movie-review-data/`](http://www.cs.cornell.edu/people/pabo/movie-review-data/)。要下载、解压并转换这些数据，我们可以运行以下代码：

    ```py
    import tensorflow_datasets as tfds
    movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    path = tf.keras.utils.get_file(movie_data_url.split("/")[-1], movie_data_url, extract=True)
    path = path.replace('.tar.gz', '')
    with open(path+filename, 'r', encoding='utf-8', errors='ignore') as movie_file:
        for response, filename in enumerate(['\\rt-polarity.neg', '\\rt-polarity.pos']):
            with open(path+filename, 'r') as movie_file:
                for line in movie_file:
                    review_file.write(str(response) + '\t' + line.encode('utf-8').decode())

    def split_text(x):
        return tf.strings.split(x, sep='\t')
    movies = (tf.data
              .TextLineDataset('movie_reviews.txt')
              .map(split_text)
             ) 
    ```

1.  **CIFAR-10 图像数据**：加拿大高级研究院发布了一个包含 8000 万标记彩色图像的图像集（每张图像的尺寸为 32 x 32 像素）。该数据集包含 10 个不同的目标类别（如飞机、汽车、鸟类等）。CIFAR-10 是一个子集，包含 60,000 张图像，其中训练集有 50,000 张图像，测试集有 10,000 张图像。由于我们将在多种方式下使用该数据集，并且它是我们较大的数据集之一，我们不会每次都运行脚本来获取它。要获取该数据集，只需执行以下代码来下载 CIFAR-10 数据集（这可能需要较长时间）：

    ```py
    import tensorflow_datasets as tfds
    ds, info = tfds.load('cifar10', shuffle_files=True, with_info=True)
    print(info)
    cifar_train = ds['train']
    cifar_test = ds['test'] 
    ```

1.  **莎士比亚作品文本数据**：Project Gutenberg 是一个发布免费书籍电子版的项目。他们已经将莎士比亚的所有作品汇编在一起。以下代码展示了如何通过 TensorFlow 访问这个文本文件：

    ```py
    import tensorflow_datasets as tfds
    shakespeare_url = 'https://raw.githubusercontent.com/PacktPublishing/TensorFlow-2-Machine-Learning-Cookbook-Third-Edition/master/shakespeare.txt'
    path = tf.keras.utils.get_file(shakespeare_url.split("/")[-1], shakespeare_url)
    def split_text(x):
        return tf.strings.split(x, sep='\n')
    shakespeare_text = (tf.data
                        .TextLineDataset(path)
                        .map(split_text)
                       ) 
    ```

1.  **英语-德语句子翻译数据**：Tatoeba 项目（[`tatoeba.org`](http://tatoeba.org)）收集了多种语言的句子翻译。他们的数据已根据创意共享许可证发布。从这些数据中，ManyThings.org（[`www.manythings.org`](http://www.manythings.org)）编译了可供下载的文本文件，包含逐句翻译。在这里，我们将使用英语-德语翻译文件，但你可以根据需要更改 URL 来使用其他语言：

    ```py
    import os
    import pandas as pd
    from zipfile import ZipFile
    from urllib.request import urlopen, Request
    import tensorflow_datasets as tfds
    sentence_url = 'https://www.manythings.org/anki/deu-eng.zip'
    r = Request(sentence_url, headers={'User-Agent': 'Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11'})
    b2 = [z for z in sentence_url.split('/') if '.zip' in z][0] #gets just the '.zip' part of the url
    with open(b2, "wb") as target:
        target.write(urlopen(r).read()) #saves to file to disk
    with ZipFile(b2) as z:
        deu = [line.split('\t')[:2] for line in z.open('deu.txt').read().decode().split('\n')]
    os.remove(b2) #removes the zip file
    # saving to disk prepared en-de sentence file
    with open("deu.txt", "wb") as deu_file:
        for line in deu:
            data = ",".join(line)+'\n'
            deu_file.write(data.encode('utf-8'))

    def split_text(x):
        return tf.strings.split(x, sep=',')
    text_data = (tf.data
                 .TextLineDataset("deu.txt")
                 .map(split_text)
                ) 
    ```

使用完这个数据集后，我们已完成对本书中您在使用配方时最常遇到的数据集的回顾。在每个配方开始时，我们会提醒您如何下载相关数据集，并解释它为何与该配方相关。

## 工作原理……

当涉及到在某个配方中使用这些数据集时，我们将参考本节内容，并假定数据已经按照我们刚才描述的方式加载。如果需要进一步的数据转换或预处理，相关代码将在配方中提供。

通常，当我们使用来自 TensorFlow 数据集的数据时，方法通常如下所示：

```py
import tensorflow_datasets as tfds
dataset_name = "..."
data = tfds.load(dataset_name, split=None)
train = data['train']
test = data['test'] 
```

无论如何，根据数据的位置，可能需要下载、解压并转换它。

## 另见

这是我们在本书中使用的部分数据资源的附加参考资料：

+   Hosmer, D.W., Lemeshow, S., 和 Sturdivant, R. X.（2013 年） *《应用逻辑回归：第 3 版》*

+   Lichman, M. (2013). *UCI 机器学习库*：[`archive.ics.uci.edu/ml`](http://archive.ics.uci.edu/ml)。加利福尼亚州欧文市：加利福尼亚大学信息与计算机科学学院

+   Bo Pang, Lillian Lee 和 Shivakumar Vaithyanathan，*好评？使用机器学习技术进行情感分类*，EMNLP 2002 会议论文：[`www.cs.cornell.edu/people/pabo/movie-review-data/`](http://www.cs.cornell.edu/people/pabo/movie-review-data/)

+   Krizhevsky. (2009). *从小图像中学习多层特征*： [`www.cs.toronto.edu/~kriz/cifar.html`](http://www.cs.toronto.edu/~kriz/cifar.html)

+   *Project Gutenberg. 访问于* 2016 年 4 月：[`www.gutenberg.org/`](http://www.gutenberg.org/ )

# 其他资源

在这一部分，你将找到更多的链接、文档资源和教程，这些在学习和使用 TensorFlow 时会提供很大帮助。

## 准备工作

在学习如何使用 TensorFlow 时，知道在哪里寻求帮助或提示是很有帮助的。本节列出了启动 TensorFlow 和解决问题的一些资源。

## 如何做到……

以下是 TensorFlow 资源的列表：

+   本书的代码可以在 Packt 仓库在线访问：[`github.com/PacktPublishing/Machine-Learning-Using-TensorFlow-Cookbook`](https://github.com/PacktPublishing/Machine-Learning-Using-TensorFlow-Cookbook)

+   TensorFlow 官方 Python API 文档位于 [`www.tensorflow.org/api_docs/python`](https://www.tensorflow.org/api_docs/python)。在这里，你可以找到所有 TensorFlow 函数、对象和方法的文档及示例。

+   TensorFlow 的官方教程非常全面和详细，位于 [`www.tensorflow.org/tutorials/index.html`](https://www.tensorflow.org/tutorials/index.html)。它们从图像识别模型开始，接着讲解 Word2Vec、RNN 模型以及序列到序列模型。它们还提供了生成分形图形和求解 PDE 系统的额外教程。请注意，他们会不断地向这个合集添加更多教程和示例。

+   TensorFlow 的官方 GitHub 仓库可以通过 [`github.com/tensorflow/tensorflow`](https://github.com/tensorflow/tensorflow) 访问。在这里，你可以查看开源代码，甚至如果你愿意，可以分叉或克隆当前版本的代码。你也可以通过访问 `issues` 目录来查看当前已提交的问题。

+   TensorFlow 提供的一个由官方维护的公共 Docker 容器，始终保持最新版本，位于 Dockerhub：[`hub.docker.com/r/tensorflow/tensorflow/`](https://hub.docker.com/r/tensorflow/tensorflow/)。

+   Stack Overflow 是一个很好的社区帮助源。在这里有一个 TensorFlow 标签。随着 TensorFlow 越来越受欢迎，这个标签的讨论也在增长。要查看该标签的活动，可以访问 [`stackoverflow.com/questions/tagged/Tensorflow`](http://stackoverflow.com/questions/tagged/Tensorflow)。

+   虽然 TensorFlow 非常灵活，能够用于许多用途，但它最常见的用途是深度学习。为了理解深度学习的基础、底层数学如何运作，并培养更多的深度学习直觉，Google 创建了一个在线课程，课程在 Udacity 上提供。要注册并参加这个视频讲座课程，请访问[`www.udacity.com/course/deep-learning--ud730`](https://www.udacity.com/course/deep-learning--ud730)。

+   TensorFlow 还创建了一个网站，你可以在其中通过调整参数和数据集来直观地探索训练神经网络。访问[`playground.tensorflow.org/`](http://playground.tensorflow.org/)来探索不同设置如何影响神经网络的训练。

+   Andrew Ng 讲授了一门名为《神经网络与深度学习》的在线课程：[`www.coursera.org/learn/neural-networks-deep-learning`](https://www.coursera.org/learn/neural-networks-deep-learning)

+   斯坦福大学提供了一个在线大纲和详细的课程笔记，内容涉及*卷积神经网络与视觉识别*：[`cs231n.stanford.edu/`](http://cs231n.stanford.edu/)
