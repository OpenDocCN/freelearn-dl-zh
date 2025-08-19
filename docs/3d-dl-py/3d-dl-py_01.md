# 前言

从事 3D 计算机视觉开发的人员可以通过这本实践指南，将他们的知识应用于 3D 深度学习。书中提供了一个动手实现的方式，并配有相关方法论，让你迅速上手并有效工作。

本书通过详细的步骤讲解基础概念、实践示例和自我评估问题，帮助你从探索最前沿的 3D 深度学习开始。

你将学习如何使用 PyTorch3D 处理基本的 3D 网格和点云数据，例如加载和保存 PLY 和 OBJ 文件、使用透视相机模型或正交相机模型将 3D 点投影到相机坐标系，并将点云和网格渲染成图像等。你还将学习如何实现某些最前沿的 3D 深度学习算法，如微分渲染、NeRF、SynSin 和 Mesh R-CNN，因为使用 PyTorch3D 库进行这些深度学习模型的编程将变得更加容易。

到本书结束时，你将能够实现自己的 3D 深度学习模型。

# 本书适合谁阅读

本书适合初学者和中级机器学习从业者、数据科学家、机器学习工程师和深度学习工程师，他们希望掌握使用 3D 数据的计算机视觉技术。

# 本书的内容

*第一章*，*介绍 3D 数据处理*，将介绍 3D 数据的基础知识，例如 3D 数据是如何存储的，以及网格和点云、世界坐标系和相机坐标系的基本概念。它还会向我们展示什么是 NDC（一个常用的坐标系），如何在不同坐标系之间进行转换，透视相机和正交相机，以及应使用哪些相机模型。

*第二章*，*介绍 3D 计算机视觉与几何学*，将向我们展示计算机图形学中的基本概念，如渲染和着色。我们将学习一些在后续章节中需要的基础概念，包括 3D 几何变换、PyTorch 张量和优化。

*第三章*，*将可变形网格模型拟合到原始点云*，将展示一个使用可变形 3D 模型拟合噪声 3D 观测值的动手项目，利用我们在前几章中学到的所有知识。我们将探讨常用的代价函数，为什么这些代价函数很重要，以及它们通常在何时使用。最后，我们将探讨一个具体的例子，说明哪些代价函数被选用于哪些任务，以及如何设置优化循环以获得我们想要的结果。

*第四章*，*通过可微分渲染学习物体姿态检测与跟踪*，将讲解可微分渲染的基本概念。它将帮助你理解基本概念，并了解何时可以应用这些技术来解决自己的问题。

*第五章*，*理解可微分体积渲染*，将通过一个实际项目介绍如何使用可微分渲染从单张图片和已知的 3D 网格模型估计相机位置。我们将学习如何实际使用 PyTorch3D 来设置相机、渲染和着色器。我们还将亲手体验如何使用不同的代价函数来获得优化结果。

*第六章*，*探索神经辐射场（NeRF）*，将提供一个实际项目，使用可微分渲染从多张图片和纹理模型估计 3D 网格模型。

*第七章*，*探索可控神经特征场*，将介绍一个视图合成的重要算法，即 NeRF。我们将了解它的原理、如何使用以及它的价值所在。

*第八章*，*3D 人体建模*，将探讨使用 SMPL 算法进行 3D 人体拟合。

*第九章*，*使用 SynSin 进行端到端视图合成*，将介绍 SynSin，这是一个最先进的深度学习图像合成模型。

*第十章*，*Mesh R-CNN*，将介绍 Mesh R-CNN，这是另一种最先进的方法，用于从单张输入图像预测 3D 体素模型。

# 为了最大限度地从本书中受益

| **本书中涵盖的软件/硬件** | **操作系统要求** |
| --- | --- |
| Python 3.6+ | Windows、macOS 或 Linux |

**如果你使用的是本书的电子版，我们建议你自己输入代码或访问** **本书的 GitHub 代码库（下一节中会提供链接）。这样可以帮助你避免因复制粘贴代码而可能出现的错误。**

请参考以下论文：

*第六章**:* [`arxiv.org/abs/2003.08934`](https://arxiv.org/abs/2003.08934)*,* [`github.com/yenchenlin/nerf-pytorch`](https://github.com/yenchenlin/nerf-pytorch)

*第七章**:* [`m-niemeyer.github.io/project-pages/giraffe/index.xhtml`](https://m-niemeyer.github.io/project-pages/giraffe/index.xhtml)*,*[`arxiv.org/abs/2011.12100`](https://arxiv.org/abs/2011.12100)

*第八章**:* [`smpl.is.tue.mpg.de/`](https://smpl.is.tue.mpg.de/)*,* [`smplify.is.tue.mpg.de/, https://smpl-x.is.tue.mpg.de/`](https://smplify.is.tue.mpg.de/)

*第九章**:* [`arxiv.org/pdf/1912.08804.pdf`](https://arxiv.org/pdf/1912.08804.pdf)

*第十章**:* [`arxiv.org/abs/1703.06870`](https://arxiv.org/abs/1703.06870), [`arxiv.org/abs/1906.02739`](https://arxiv.org/abs/1906.02739)

# 下载示例代码文件

你可以从 GitHub 上下载本书的示例代码文件，网址为[`github.com/PacktPublishing/3D-Deep-Learning-with-Python`](https://github.com/PacktPublishing/3D-Deep-Learning-with-Python)。如果代码有更新，将在 GitHub 仓库中更新。

我们还提供来自丰富图书和视频目录中其他代码包，网址为[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)。请查看！

# 下载彩色图像

我们还提供了一个 PDF 文件，其中包含本书使用的截图和图表的彩色图像。您可以从这里下载：[`packt.link/WJr0Q`](https://packt.link/WJr0Q)。

# 使用的约定

本书中使用了许多文本约定。

`文本中的代码`: 指示文本中的代码字词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 句柄。例如：“接下来，我们需要更新`./``options/options.py`文件”

代码块如下所示：

```py
elif opt.dataset == 'kitti':
```

```py
   opt.min_z = 1.0
```

```py
   opt.max_z = 50.0
```

```py
   opt.train_data_path = (
```

```py
       './DATA/dataset_kitti/'
```

```py
   )
```

```py
   from data.kitti import KITTIDataLoader
```

```py
   return KITTIDataLoader
```

当我们希望引起您对代码块特定部分的注意时，相关行或项目将以**粗体**显示：

```py
wget https://dl.fbaipublicfiles.com/synsin/checkpoints/realestate/synsin.pth
```

任何命令行输入或输出均如下所示：

```py
bash ./download_models.sh
```

**粗体**: 表示新术语、重要词汇或屏幕上显示的词语。例如：“细化模块（**g**）从神经点云渲染器获取输入，然后输出最终重建的图像。”

提示或重要说明

如此显示。

# 联系我们

我们时刻欢迎读者的反馈。

**常规反馈**: 如果您对本书的任何方面有疑问，请发送电子邮件至[customercare@packtpub.com](http://customercare@packtpub.com)，并在邮件主题中提到书名。

**勘误**: 尽管我们已经非常注意确保内容的准确性，但错误是不可避免的。如果您在本书中发现错误，请向我们报告。请访问[www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)并填写表格。

**盗版**: 如果您在互联网上发现我们作品的任何非法副本，请提供地址或网站名称，我们将不胜感激。请联系我们，网址为[copyrigh@packt.com](http://copyright@packt.com)，并附上链接。

**如果您有兴趣成为作者**: 如果您在某个专题上有专长，并且有意撰写或为书籍做贡献，请访问[authors.packtpub.com](http://authors.packtpub.com)。

# 分享您的想法

阅读完《*Python 与 3D 深度学习*》后，我们非常希望听到您的想法！[请点击这里直接进入亚马逊评论页面](https://packt.link/r/1-803-24782-7)并分享您的反馈。

您的评论对我们和技术社区至关重要，将帮助我们确保提供优质的内容。

# 下载本书的免费 PDF 版本

感谢您购买本书！

您喜欢随时随地阅读，但无法将纸质书籍带到处吗？

您的电子书购买是否与您选择的设备不兼容？

不用担心，现在每本 Packt 图书都附带无 DRM 保护的 PDF 版本，您可以免费获得。

在任何地方、任何设备上阅读。搜索、复制并将代码从您最喜欢的技术书籍直接粘贴到您的应用程序中。

优惠不仅仅如此，您还可以独享折扣、新闻通讯以及每天直接发送到您邮箱的精彩免费内容

按照这些简单步骤获取相关福利：

1.  扫描二维码或访问下面的链接

![](img/B18217_QR_Free_PDF.jpg)

[`packt.link/free-ebook/9781803247823`](https://packt.link/free-ebook/9781803247823)

1.  提交您的购买凭证

1.  就这样！我们将直接把您的免费 PDF 和其他福利发送到您的电子邮箱。

# 第一部分：3D 数据处理基础

本书的第一部分将定义数据和图像处理的最基本概念，因为这些概念对我们后续的讨论至关重要。此部分内容使本书内容自成一体，读者无需阅读其他书籍即可开始学习 PyTorch3D。

本部分包括以下章节：

+   *第一章*，*介绍 3D 数据处理*

+   *第二章*，*介绍 3D 计算机视觉与几何学*
