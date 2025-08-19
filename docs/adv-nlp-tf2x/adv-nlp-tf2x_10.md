

# 第十章：代码安装和设置说明

本章提供了为书中代码设置环境的说明。这些说明：

+   已在 macOS 10.15 和 Ubuntu 18.04.3 LTS 上进行测试。你可能需要将这些说明转换为 Windows 版本。

+   仅覆盖 TensorFlow 的 CPU 版本。有关最新的 GPU 安装说明，请参考 [`www.tensorflow.org/install/gpu`](https://www.tensorflow.org/install/gpu)。请注意，强烈推荐使用 GPU，它将把复杂模型的训练时间从几天缩短到几个小时。

安装使用了 Anaconda 和 `pip`。假设你的机器上已经设置并准备好使用 Anaconda。注意，我们使用了一些新的和不常见的包，这些包可能无法通过 `conda` 安装。在这种情况下，我们将使用 `pip`。

**说明**：

+   在 macOS 上：`conda` 49.2，`pip` 20.3.1

+   在 Ubuntu 上：`conda` 4.6.11，`pip` 20.0.2

# GitHub 位置

本书的代码位于以下公共 GitHub 仓库：

[`github.com/PacktPublishing/Advanced-Natural-Language-Processing-with-TensorFlow-2`](https://github.com/PacktPublishing/Advanced-Natural-Language-Processing-with-TensorFlow-2)

请克隆此仓库以访问书中的所有代码。请注意，每一章的经典论文都包含在 GitHub 仓库中的相应章节目录里。

现在，设置 `conda` 环境的常规步骤如下所述：

+   **步骤 1**：创建一个新的 `conda` 环境，使用 Python 3.7.5：

    ```py
    $ conda create -n tf24nlp python==3.7.5 
    ```

    环境名为 `tf24nlp`，但你可以自由使用自己命名，并确保在接下来的步骤中使用该名称。我喜欢在我的环境名称前加上正在使用的 TensorFlow 版本，并且如果该环境包含 GPU 版本的库，我会在名称后加上“g”。正如你可能推测的那样，我们将使用 TensorFlow 2.4。

+   **步骤 2**：激活环境并安装以下软件包：

    ```py
    $ conda activate tf24nlp
    (tf24nlp) $  conda install pandas==1.0.1 numpy==1.18.1 
    ```

    这会在我们新创建的环境中安装 NumPy 和 pandas 库。

+   **步骤 3**：安装 TensorFlow 2.4。为此，我们需要使用 `pip`。截至写作时，TensorFlow 的 `conda` 发行版仍为 2.0。TensorFlow 发展非常迅速。通常，`conda` 发行版会稍微滞后于最新版本：

    ```py
    (tf24nlp) $ pip install tensorflow==2.4 
    ```

    请注意，这些说明适用于 TensorFlow 的 CPU 版本。如需 GPU 安装说明，请参考 [`www.tensorflow.org/install/gpu`](https://www.tensorflow.org/install/gpu)。

+   **步骤 4**：安装 Jupyter Notebook —— 可以自由安装最新版本：

    ```py
    (tf24nlp) $ conda install Jupyter 
    ```

    剩余的安装说明涉及特定章节中使用的库。如果你在 Jupyter Notebook 中安装时遇到问题，可以通过命令行安装。

每个章节的具体安装说明如下所示。

# 第一章安装说明

本章不需要特定的安装说明，因为本章的代码将在 Google Colab 上运行，网址是[colab.research.google.com](http://colab.research.google.com)。

# 第二章安装说明

需要安装`tfds`包：

```py
(tf24nlp) $ pip install tensorflow_datasets==3.2.1 
```

在接下来的大多数章节中，我们将使用`tfds`。

# 第三章安装说明

1.  通过以下命令安装`matplotlib`：

    ```py
    (tf24nlp) $ conda install matplotlib==3.1.3 
    ```

    新版本也可能可行。

1.  安装用于维特比解码的 TensorFlow Addons 包：

    ```py
    (tf24nlp) $ pip install tensorflow_addons==0.11.2 
    ```

    请注意，此包无法通过`conda`获取。

# 第四章安装说明

本章需要安装`sklearn`：

```py
(tf24nlp) $ conda install scikit-learn==0.23.1 
```

还需要安装 Hugging Face 的 Transformers 库：

```py
(tf24nlp) $ pip install transformers==3.0.2 
```

# 第五章安装说明

无需其他安装。

# 第六章安装说明

需要安装一个用于计算 ROUGE 分数的库：

```py
(tf24nlp) $ pip install rouge_score 
```

# 第七章安装说明

我们需要 Pillow 库来处理图像。该库是 Python Imaging Library 的友好版。可以通过以下命令安装：

```py
(tf24nlp) conda install pillow==7.2.0 
```

TQDM 是一个很好的工具，可以在执行长时间循环时显示进度条：

```py
(tf24nlp) $ conda install tqdm==4.47.0 
```

# 第八章安装说明

需要安装 Snorkel。写本文时，安装的 Snorkel 版本为 0.9.5。请注意，这个版本的 Snorkel 使用了旧版本的 pandas 和 TensorBoard。为了本书中的代码，您应该能够安全地忽略关于版本不匹配的警告。但是，如果您的环境中继续出现冲突，建议创建一个专门的 Snorkel `conda`环境。

在该环境中运行标注函数，并将输出存储为单独的 CSV 文件。TensorFlow 训练可以通过切换回`tf24nlp`环境并加载标注数据来运行：

```py
(tf24nlp) $ pip install snorkel==0.9.5 
```

我们还将使用 BeautifulSoup 来解析 HTML 标签：

```py
(tf24nlp) $ conda install beautifulsoup4==4.9 
```

本章有一个可选部分涉及绘制词云。此部分需要安装以下包：

```py
(tf24nlp) $ pip install wordcloud==1.8 
```

请注意，本章还使用了 NLTK，这是我们在第一章中安装的。

# 第九章安装说明

无。

| **分享您的经验**感谢您抽出时间阅读本书。如果您喜欢本书，请帮助其他人找到它。在[`www.amazon.com/dp/1800200935`](https://www.amazon.com/dp/1800200935)上留下评论。 |
| --- |
