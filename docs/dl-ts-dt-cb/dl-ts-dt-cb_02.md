

# 第二章：开始使用 PyTorch

在本章中，我们将探索 **PyTorch**，一个领先的 Python 深度学习库。

我们将介绍一些有助于理解如何使用 PyTorch 构建神经网络的操作。除了张量操作，我们还将探讨如何训练不同类型的神经网络。具体来说，我们将重点关注前馈神经网络、循环神经网络、**长短期记忆**（**LSTM**）和 1D 卷积神经网络。

在后续章节中，我们还将介绍其他类型的神经网络，例如 Transformer。这里，我们将使用合成数据进行演示，这将帮助我们展示每种模型背后的实现和理论。

完成本章后，您将对 PyTorch 有深入的理解，并掌握进行更高级深度学习项目的工具。

在本章中，我们将介绍以下几种方法：

+   安装 PyTorch

+   PyTorch 中的基本操作

+   PyTorch 中的高级操作

+   使用 PyTorch 构建一个简单的神经网络

+   训练前馈神经网络

+   训练循环神经网络

+   训练 LSTM 神经网络

+   训练卷积神经网络

# 技术要求

在开始之前，您需要确保您的系统满足以下技术要求：

+   **Python 3.9**：您可以从[`www.python.org/downloads/`](https://www.python.org/downloads/) 下载 Python。

+   pip（23.3.1）或 Anaconda：这些是 Python 的常用包管理器。pip 默认与 Python 一起安装。Anaconda 可以从[`www.anaconda.com/products/distribution`](https://www.anaconda.com/products/distribution)下载。

+   torch（2.2.0）：本章中我们将使用的主要深度学习库。

+   **CUDA（可选）**：如果您的计算机上有支持 CUDA 的 GPU，您可以安装支持 CUDA 的 PyTorch 版本。这将使您能够在 GPU 上进行计算，并且可以显著加快深度学习实验的速度。

值得注意的是，本章中介绍的代码是平台无关的，并且应当可以在满足前述要求的任何系统上运行。

本章的代码可以在以下 GitHub 地址找到：[`github.com/PacktPublishing/Deep-Learning-for-Time-Series-Data-Cookbook`](https://github.com/PacktPublishing/Deep-Learning-for-Time-Series-Data-Cookbook)。

# 安装 PyTorch

要开始使用 PyTorch，首先需要安装它。根据写作时的信息，PyTorch 支持 Linux、macOS 和 Windows 平台。在这里，我们将引导您完成这些操作系统上的安装过程。

## 准备工作

`PyTorch` 通常通过 `pip` 或 Anaconda 安装。我们建议在安装库之前创建一个新的 Python 环境，特别是当您需要在系统上进行多个 Python 项目时。这是为了避免不同项目可能需要的 Python 库版本之间发生冲突。

## 如何实现…

让我们看看如何安装`PyTorch`。我们将描述如何使用`pip`或`Anaconda`来完成此操作。我们还将提供有关如何使用 CUDA 环境的一些信息。

如果你使用`pip`，Python 的包管理器，你可以在终端中运行以下命令来安装 PyTorch：

```py
pip install torch
```

使用 Anaconda Python 发行版，你可以使用以下命令安装 PyTorch：

```py
conda install pytorch torchvision -c pytorch
```

如果你的机器上有支持 CUDA 的 GPU，你可以安装支持 CUDA 的`PyTorch`版本，以便在 GPU 上进行计算。这可以显著加速你的深度学习实验。PyTorch 官网提供了一个工具，根据你的需求生成相应的安装命令。访问 PyTorch 官网，在**Quick Start Locally**部分选择你的偏好（如操作系统、包管理器、Python 版本和 CUDA 版本），然后将生成的命令复制到终端中。

## 它是如何工作的…

安装`PyTorch`后，你可以通过打开 Python 解释器并运行以下代码来验证一切是否正常工作：

```py
import torch
print(torch.__version__)
```

这应该输出你安装的`PyTorch`版本。现在，你已经准备好开始使用`PyTorch`进行深度学习了！

在接下来的章节中，我们将熟悉`PyTorch`的基础知识，并构建我们的第一个神经网络。

# PyTorch 中的基本操作

在我们开始使用`PyTorch`构建神经网络之前，理解如何使用这个库操作数据是至关重要的。在`PyTorch`中，数据的基本单元是张量，它是矩阵的一种推广，支持任意维度（也称为多维数组）。

## 准备开始

张量可以是一个数字（0 维张量），一个向量（1 维张量），一个矩阵（2 维张量），或者任何多维数据（3 维张量、4 维张量等等）。`PyTorch`提供了多种函数来创建和操作张量。

## 如何做…

让我们从导入`PyTorch`开始：

```py
import torch
```

我们可以使用各种技术在`PyTorch`中创建张量。让我们从使用列表创建张量开始：

```py
t1 = torch.tensor([1, 2, 3])
print(t1)
t2 = torch.tensor([[1, 2], [3, 4]])
print(t2)
```

`PyTorch`可以与`NumPy`无缝集成，允许从`NumPy`数组轻松创建张量：

```py
import numpy as np
np_array = np.array([5, 6, 7])
t3 = torch.from_numpy(np_array)
print(t3)
```

`PyTorch`还提供了生成特定值（如零或一）张量的函数：

```py
t4 = torch.zeros((3, 3))
print(t4)
t5 = torch.ones((3, 3))
print(t5)
t6 = torch.eye(3)
print(t6)
```

这些是`NumPy`中常用的方法，`PyTorch`中也可以使用这些方法。

## 它是如何工作的…

现在我们知道如何创建张量，让我们来看看一些基本操作。我们可以对张量执行所有标准的算术操作：

```py
result = t1 + t3
print(result)
result = t3 - t1
print(result)
result = t1 * t3
print(result)
result = t3 / t1
print(result)
```

你可以使用`.reshape()`方法来重塑张量：

```py
t7 = torch.arange(9) # Creates a 1D tensor [0, 1, 2, ..., 8]
t8 = t7.reshape((3, 3)) # Reshapes the tensor to a 3x3 matrix
print(t8)
```

这是`PyTorch`中张量操作的简要介绍。随着你深入学习，你会发现`PyTorch`提供了多种操作来处理张量，给予你实现复杂深度学习模型和算法所需的灵活性和控制力。

# PyTorch 中的高级操作

探索了基本的张量操作后，现在让我们深入了解 `PyTorch` 中的更高级操作，特别是构成深度学习中大多数数值计算基础的线性代数操作。

## 准备工作

线性代数是数学的一个子集。它涉及向量、向量空间及这些空间之间的线性变换，如旋转、缩放和剪切。在深度学习的背景下，我们处理的是高维向量（张量），对这些向量的操作在模型的内部工作中起着至关重要的作用。

## 如何做……

让我们从回顾上一节中创建的张量开始：

```py
print(t1)
print(t2)
```

两个向量的点积是一个标量，衡量向量的方向和大小。在 `PyTorch` 中，我们可以使用 `torch.dot()` 函数计算两个 `1D` 张量的点积：

```py
dot_product = torch.dot(t1, t3)
print(dot_product)
```

与逐元素相乘不同，矩阵乘法，也叫做点积，是将两个矩阵相乘以产生一个新的矩阵的操作。`PyTorch` 提供了 `torch.mm()` 函数来执行矩阵乘法：

```py
matrix_product = torch.mm(t2, t5)
print(matrix_product)
```

矩阵的转置是一个新矩阵，它的行是原始矩阵的列，而列是原始矩阵的行。你可以使用 `.T` 属性来计算张量的转置：

```py
t_transposed = t2.T
print(t_transposed)
```

你可以执行其他操作，例如计算矩阵的行列式和求矩阵的逆。让我们看几个这样的操作：

```py
det = torch.det(t2)
print(det)
inverse = torch.inverse(t2)
print(inverse)
```

注意，这两个操作仅在 `2D` 张量（矩阵）上定义。

## 它是如何工作的……

`PyTorch` 是一个高度优化的库，特别适用于执行基本和高级操作，尤其是深度学习中至关重要的线性代数操作。

这些操作使得 `PyTorch` 成为构建和训练神经网络以及在更一般的背景下执行高阶计算的强大工具。在下一节中，我们将使用这些构建块开始构建深度学习模型。

# 使用 PyTorch 构建一个简单的神经网络

本节将从头开始构建一个简单的两层神经网络，仅使用基本的张量操作来解决时间序列预测问题。我们旨在演示如何手动实现前向传播、反向传播和优化步骤，而不依赖于 `PyTorch` 的预定义层和优化例程。

## 准备工作

我们使用合成数据进行这个演示。假设我们有一个简单的时间序列数据，共 `100` 个样本，每个样本有 `10` 个时间步。我们的任务是根据前面的时间步预测下一个时间步：

```py
X = torch.randn(100, 10)
y = torch.randn(100, 1)
```

现在，让我们创建一个神经网络。

## 如何做……

让我们从定义模型参数及其初始值开始。在这里，我们创建了一个简单的两层网络，因此我们有两组权重和偏置：

我们使用 `requires_grad_()` 函数告诉 `PyTorch`，我们希望在反向传播时计算这些张量的梯度。

接下来，我们定义我们的模型。对于这个简单的网络，我们将在隐藏层使用 sigmoid 激活函数：

```py
input_size = 10
hidden_size = 5
output_size = 1
W1 = torch.randn(hidden_size, input_size).requires_grad_()
b1 = torch.zeros(hidden_size, requires_grad=True)
W2 = torch.randn(output_size, hidden_size).requires_grad_()
b2 = torch.zeros(output_size, requires_grad=True)
def simple_neural_net(x, W1, b1, W2, b2):
    z1 = torch.mm(x, W1.t()) + b1
    a1 = torch.sigmoid(z1)
    z2 = torch.mm(a1, W2.t()) + b2
    return z2
```

现在，我们已经准备好训练模型了。让我们定义学习率和训练的轮次（epochs）：

```py
lr = 0.01
epochs = 100
loss_fn = torch.nn.MSELoss()
for epoch in range(epochs):
    y_pred = simple_neural_net(X, W1, b1, W2, b2)
    loss = loss_fn(y_pred.squeeze(), y)
    loss.backward()
    with torch.no_grad():
        W1 -= lr * W1.grad
        b1 -= lr * b1.grad
        W2 -= lr * W2.grad
        b2 -= lr * b2.grad
    W1.grad.zero_()
    b1.grad.zero_()
    W2.grad.zero_()
    b2.grad.zero_()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch} \t Loss: {loss.item()}')
```

这段基本代码演示了神经网络的基本部分：前向传播，我们计算预测值；反向传播，我们计算梯度；以及更新步骤，我们调整权重以最小化损失。

## 还有更多内容…

本章重点探讨神经网络训练过程的复杂性。在未来的章节中，我们将展示如何训练深度神经网络，而无需担心这些细节。

# 训练前馈神经网络

本教程将带你逐步完成使用 PyTorch 构建前馈神经网络的过程。

## 准备工作

前馈神经网络，也被称为**多层感知器**（**MLPs**），是最简单的人工神经网络之一。数据从输入层流向输出层，经过隐藏层，不包含任何循环。在这种类型的神经网络中，一层的所有隐藏单元都与下一层的单元相连。

## 如何实现…

让我们使用 `PyTorch` 创建一个简单的前馈神经网络。首先，我们需要导入必要的 `PyTorch` 模块：

```py
import torch
import torch.nn as nn
```

现在，我们可以定义一个带有单一隐藏层的简单前馈神经网络：

```py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
net = Net()
print(net)
```

在上述代码中，`nn.Module` 是 `PyTorch` 中所有神经网络模块的基类，我们的网络是它的一个子类。

该类中的 `forward()` 方法表示网络的前向传播过程。这是网络在将输入转换为输出时执行的计算。以下是逐步的解释：

+   `forward()` 方法接收一个输入张量 `x`。这个张量表示输入数据，它的形状应该与网络的层兼容。在这里，作为第一个线性层（`self.fc1`）期望有 `10` 个输入特征，`x` 的最后一个维度应该是 `10`。

+   输入张量首先通过线性变换处理，由 `self.fc1` 表示。这个对象是 `PyTorch` 的 `nn.Linear` 类的一个实例，它执行线性变换，涉及用权重矩阵乘以输入数据并加上偏置向量。正如在 `__init__()` 方法中定义的那样，这一层将 10 维空间转化为 5 维空间，使用的是线性变换。这个降维过程通常被视为神经网络“学习”或“提取”输入数据中的特征。

+   第一层的输出随后通过 `torch.relu()` 进行处理。这是一个简单的非线性函数，它将张量中的负值替换为零。这使得神经网络能够建模输入和输出之间更复杂的关系。

+   `ReLU()`函数的输出接着通过另一个线性变换`self.fc2`。和之前一样，这个对象是`PyTorch`的`nn.Linear`类的一个实例。这个层将张量的维度从`5`（前一层的输出大小）缩减到`1`（所需的输出大小）。

最后，第二个线性层的输出由`forward()`方法返回。这个输出可以用于多种目的，例如计算用于训练网络的损失，或者作为推理任务中的最终输出（即网络用于预测时）。

## 它是如何工作的…

要训练网络，我们需要一个数据集，一个损失函数，以及一个优化器。

我们使用与前一个示例相同的合成数据集：

```py
X = torch.randn(100, 10)
Y = torch.randn(100, 1)
```

我们可以使用**均方误差**（**MSE**）损失来进行我们的任务，这是回归问题中常用的损失函数。PyTorch 提供了这个损失函数的内置实现：

```py
loss_fn = nn.MSELoss()
```

我们将使用**随机梯度下降**（**SGD**）作为我们的优化器。SGD 是一种迭代方法，用于优化目标函数：

```py
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
```

现在我们可以训练我们的网络。我们将训练`100`个周期：

```py
for epoch in range(100):
    output = net(X)
    loss = loss_fn(output, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
```

在每个周期中，我们执行前向传播，计算损失，进行反向传播以计算梯度，然后更新权重。

你现在已经使用`PyTorch`训练了一个简单的前馈神经网络。在接下来的章节中，我们将深入探讨更复杂的网络架构及其在时间序列分析中的应用。

# 训练递归神经网络

**递归神经网络**（**RNNs**）是一类神经网络，特别适用于涉及序列数据的任务，如时间序列预测和自然语言处理。

## 准备工作

RNN 通过具有隐藏层，能够将序列中的信息从一个步骤传递到下一个步骤，从而利用序列信息。

## 如何操作…

类似于前馈神经网络，我们首先定义了`RNN`类。为了简化，假设我们定义了一个单层的`RNN`：

```py
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)  # get RNN output
        out = self.fc(out[:, -1, :])
        return out
rnn = RNN(10, 20, 1)
print(rnn)
```

这里，`input_size`是每个时间步的输入特征数量，`hidden_size`是隐藏层中的神经元数量，`output_size`是输出特征的数量。在`forward()`方法中，我们将输入`x`和初始隐藏状态`h0`传递给递归层。RNN 返回输出和最终的隐藏状态，我们暂时忽略隐藏状态。然后，我们取序列的最后一个输出（`out[:, -1, :]`），并通过一个全连接层得到最终输出。隐藏状态充当网络的记忆，编码输入的时间上下文，直到当前时间步，这也是这种类型的神经网络在序列数据中非常有用的原因。

让我们注意一下在代码中使用的一些细节：

+   `x.device`：指的是张量`x`所在的设备。在`PyTorch`中，张量可以位于 CPU 或 GPU 上，而`.device`是一个属性，表示张量当前所在的设备。当你在 GPU 上进行计算时，所有输入到计算的张量必须位于相同的设备上。在代码行`h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)`中，我们确保初始隐藏状态张量`h0`与`x`输入张量位于同一设备上。

+   `x.size(0)`：指的是张量`x`的第`0`维的大小。在`PyTorch`中，`size()`返回张量的形状，而`size(0)`给出第一维的大小。在这个 RNN 的上下文中，`x`预计是一个形状为（`batch_size`、`sequence_length`、`num_features`）的 3D 张量，因此`x.size(0)`会返回批次大小。

## 它是如何工作的…

RNN 的训练过程与前馈网络类似。我们将使用之前示例中的相同合成数据集、损失函数（MSE）和优化器（SGD）。不过，让我们将输入数据修改为 3D 格式，以满足 RNN 的要求（`batch_size`、`sequence_length`、`num_features`）。RNN 输入张量的三个维度代表以下方面：

+   `batch_size`：表示每个批次数据中的序列数。在时间序列中，你可以把一个样本看作是一个子序列（例如，过去五天的销售数据）。因此，一个批次包含多个这样的样本或子序列，允许模型同时处理和学习多个序列。

+   `sequence_length`：本质上是你用来观察数据的窗口大小。它指定了每个输入子序列包含的时间步数。例如，如果你是基于过去的数据预测今天的温度，`sequence_length`就决定了模型每次查看的数据向后回溯了多少天。

+   `num_features`：该维度表示数据序列中每个时间步的特征（变量）数量。在时间序列的上下文中，单变量序列（例如某一地点的每日温度）在每个时间步只有一个特征。相比之下，多变量序列（例如同一地点的每日温度、湿度和风速）在每个时间步有多个特征。

让我们创建一个合成数据集作为示例：

```py
X = torch.randn(100, 5, 10)
Y = torch.randn(100, 1)
```

现在，我们可以开始训练我们的网络。我们将进行`100`轮训练：

```py
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01)
for epoch in range(100):
    output = rnn(X)
    loss = loss_fn(output, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

现在，我们已经训练了一个 RNN。这是将这些模型应用于现实世界时间序列数据的一个重要步骤，我们将在下一章讨论这一点。

# 训练 LSTM 神经网络

RNN 存在一个根本问题，即“梯度消失”问题，由于神经网络中反向传播的性质，较早输入对整体误差的影响在序列长度增加时急剧减小。在存在长期依赖的序列处理任务中尤其严重（即未来的输出依赖于很早之前的输入）。

## 准备工作

LSTM 网络被引入以克服这个问题。与 RNN 相比，它们为每个单元使用了更复杂的内部结构。具体来说，LSTM 能够根据一个叫做细胞的内部结构来决定丢弃或保存哪些信息。这个细胞通过门控（输入门、遗忘门和输出门）来控制信息的流入和流出。这有助于保持和操作“长期”信息，从而缓解梯度消失问题。

## 如何实现…

我们首先定义 `LSTM` 类。为了简化起见，我们将定义一个单层的 `LSTM` 网络。请注意，`PyTorch` 的 LSTM 期望输入是 3D 的，格式为 `batch_size`，`seq_length` 和 `num_features`：

```py
class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
     super(LSTM, self).__init__()
     self.hidden_size = hidden_size
     self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
     self.fc = nn.Linear(hidden_size, output_size)
  def forward(self, x):
     h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
     c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
     out, _ = self.lstm(x, (h0, c0))  # get LSTM output
     out = self.fc(out[:, -1, :])
     return out
lstm = LSTM(10, 20, 1) # 10 features, 20 hidden units, 1 output
print(lstm)
```

`forward()` 方法与我们之前介绍的 RNN 方法非常相似。主要的区别在于，在 RNN 的情况下，我们初始化了一个单一的隐藏状态 `h0`，并将其与输入 `x` 一起传递给 `RNN` 层。而在 LSTM 中，你需要初始化隐藏状态 `h0` 和细胞状态 `c0`，这是因为 LSTM 单元的内部结构。然后，这些状态作为元组与输入 `x` 一起传递给 `LSTM` 层。

## 它是如何工作的……

LSTM 网络的训练过程与前馈网络和 RNN 的训练过程相似。我们将使用之前示例中的相同合成数据集、损失函数（MSE）和优化器（SGD）：

```py
X = torch.randn(100, 5, 10)
Y = torch.randn(100, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(lstm.parameters(), lr=0.01)
for epoch in range(100):
      output = lstm(X)
      loss = loss_fn(output, Y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      print(f'Epoch {epoch+1}, Loss: {loss.item()}')
```

# 训练卷积神经网络

**卷积神经网络**（**CNNs**）是一类特别适用于网格状输入数据（如图像、音频谱图，甚至某些类型的时间序列数据）的神经网络。

## 准备工作

CNN 的核心思想是使用卷积滤波器（也称为内核）对输入数据进行卷积操作，这些滤波器滑过输入数据并产生输出特征图。

## 如何实现…

为了简化起见，我们定义一个单层的 `1D` 卷积神经网络，这特别适用于时间序列和序列数据。在 `PyTorch` 中，我们可以使用 `nn.Conv1d` 层来实现：

```py
class ConvNet(nn.Module):
    def __init__(self,
        input_size,
        hidden_size,
        output_size,
        kernel_size,
        seq_length):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size)
        self.fc = nn.Linear(hidden_size*(seq_length-kernel_size+1),
            output_size)
    def forward(self, x):
        x = x.transpose(1, 2)
        out = torch.relu(self.conv1(x))
        out = out.view(out.size(0), -1)  # flatten the tensor
        out = self.fc(out)
        return out
convnet = ConvNet(5, 20, 1, 3, 10)
print(convnet)
```

在 `forward` 方法中，我们将输入通过卷积层，然后是 `ReLU()` 激活函数，最后通过一个全连接层。`Conv1d` 层期望输入的形状为（`batch_size`，`num_channels`，和 `sequence_length`）。其中，`num_channels` 指输入通道的数量（相当于时间序列数据中的特征数量），`sequence_length` 则指每个样本的时间步数。

`Linear`层将接受来自`Conv1d`层的输出，并将其缩减到所需的输出大小。`Linear`层的输入计算为`hidden_size*(seq_length-kernel_size+1)`，其中`hidden_size`是`Conv1d`层的输出通道数，`seq_length-kernel_size+1`是卷积操作后的输出序列长度。

## 它是如何工作的……

`1D` CNN 的训练过程与前面的网络类型类似。我们将使用相同的损失函数（MSE）和优化器（SGD），但我们将修改输入数据的大小为（`batch_size`，`sequence_length`，`num_channels`）。请记住，通道数等于特征的数量：

```py
X = torch.randn(100, 10, 5)
Y = torch.randn(100, 1)
```

现在，我们可以训练我们的网络。我们将进行`100`个训练周期：

```py
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(convnet.parameters(), lr=0.01)
for epoch in range(100):
    output = convnet(X)
    loss = loss_fn(output, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
```

在前面的代码中，我们对每个训练周期进行迭代。每个训练周期结束后，我们将模型的误差打印到控制台，以便监控训练过程。
