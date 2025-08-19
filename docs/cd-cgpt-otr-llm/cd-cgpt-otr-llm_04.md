

# 第四章：解密生成代码的可读性

本章深入探讨了如何使 LLM 生成的代码更加清晰，易于访问和理解。我们将探索增强 LLM 生成的代码可解释性的技巧，确保 LLM 生成代码背后的逻辑和决策过程清晰可见。通过帮助更好地理解 LLM 生成的代码，本章旨在促进软件开发中的透明度和信任。

本章有不良和良好的代码及提示示例。

本章将帮助你了解提高 LLM 生成代码可读性的策略，培养一种协作和知情的方式来利用这些强大的语言模型进行编码实践。

本章将教你如何使用 LLM/chatbots 生成更易读的代码，并使用 LLM 生成文档和注释来帮助理解代码，简化复杂的代码块，便于人类理解。

本章涵盖以下主题：

+   生成更易读的代码

+   总结代码以便理解

+   生成文档

# 技术要求

本章你将需要以下内容：

+   访问 LLM/chatbot，如 GPT-4、Gemini 或 Claude 3——每个都需要登录。对于 GPT-4，你需要一个 OpenAI 账户，对于 Gemini，你需要一个 Google 账户。

+   一个 Python IDE，如 Spyder、IDLE、PyCharm、Eclipse 或 Visual Studio。在线解释器在这里不够用。

+   一个 HTML 解释器，如[`htmledit.squarefree.com/`](https://htmledit.squarefree.com/) 、[`onecompiler.com/html`](https://onecompiler.com/html) 或[`www.w3schools.com/tryit/`](https://www.w3schools.com/tryit/) ，或者网站编辑器：Wix、GoDaddy、One.com 等。

+   获取本书中的代码： [`github.com/PacktPublishing/Coding-with-ChatGPT-and-other-LLMs/tree/main/Chapter4`](https://github.com/PacktPublishing/Coding-with-ChatGPT-and-other-LLMs/tree/main/Chapter4)

现在，让我们来看看如何使用 Claude、Gemini 和 ChatGPT 来从一开始就写出非常易读的代码。

# 生成更易读的代码

在本节中，你将学习如何生成别人可以轻松阅读并立即开始使用的代码，而不需要花费几个小时去思考它的内部原理。

如果我们写代码，我们希望它被他人使用，这意味着代码应该容易被别人理解，特别是当我们希望别人对其进行扩展时。如果我们使代码易于阅读，他们更有可能愿意去理解和修改。

在子章节*用 Python 3.10 编写的数据压缩代码*中，我们将请求 LLM 生成一些数据压缩的代码，但首先，这里是关于数据压缩的简介，帮助我们更好地了解正在做的事情。数据压缩对于使用设备和互联网非常重要。

## 数据压缩方法介绍

这里是关于数据压缩的一些背景信息，帮助你更容易理解接下来的内容。

### 无损数据压缩

无损数据压缩意味着信号和噪声都被压缩，没有任何损失，所有内容都可以重新创建 [*Wiki_LZMA2*]。

无损压缩的一个例子是这样的数据集：[1,1,1,1,1,1,1….1,1,1,1,1,1,0,0,0,0,0,0…]。

这个数据集可能包含任意数量的 1 和 0，但可以通过每个值的计数和名称来表示。如果有 213 个 1 和 789 个 0，那么无损压缩方法**游程编码**（**RLE**）会将其表示为[(213, 1), (789,0)]。现在，数据只包含 4 个数字，而不是 213+789 = 1,002 个数字，这样的压缩比为 1002/4 = 250.5 [*WikiDataComp*]。这种方法在不丢失任何信息的情况下，成功地实现了良好的压缩比。这可以是实际数据，例如，“雨刷是否开启？”的问题答案。

当你有一个长时间保持不变的信号时，使用无损压缩。如果你用它处理非常精确或始终变化的数据，它反而会占用更多的磁盘空间！

LZMA 是另一种无损压缩方法，我们稍后将使用它。

### 有损压缩

第二种压缩方法叫做有损压缩，其中算法或方法尝试保持信号并去除一定程度的噪声。对于有损压缩，有不同的压缩级别，可以在保真度和文件大小之间进行权衡：更多的压缩意味着较低的保真度——更多的信号和噪声被丢弃或抑制。

有损压缩的例子包括**离散傅里叶变换**（**DFTs**）、**离散小波变换**（**DWTs**）或最常见的**离散余弦变换**（**DCTs**） [*Wiki_Wavelets*]。

DCTs 被广泛应用于许多标准的图像压缩方法（**JPEG**）、视频压缩（**MPEG**）、音频压缩（杜比数字、MP3、AAC）、数字电视（高清电视、标准清晰电视）以及数字广播（AAC+、DAB） [Wiki_Cosine]。

DCTs 将数据从随时间变化的形式转换为由不同频率的余弦波组成的形式。这听起来非常简单，但你可以用这种方式表示大多数事物。我简化了一下。你通常保留低频波形，抑制高频波形，假定它们是噪声。然后信号会变回原来的基于时间的形式：随时间变化。

如果你想要更高的保真度——更好的信号——你将减少对高频波的抑制，但文件大小几乎不会减少。如果你确实需要较小的文件大小，你将需要抑制更多的波形，但这会导致信号、图像或声音变差。另一种看压缩比的方法是比特率，单位是比特/秒。更高的比特率可以提供更高质量的数据流（例如视频或声音），如果它被存储在文件中，意味着更大的文件大小。

或者，你可以选择量化它们，而不是将值置零。在这里，值不再是精确的，而是变得不那么精确——例如，3.48724 可以适合到[3.4 到 3.5]这个范围，甚至更不精确的[3 到 4]这个范围。值变成了一个直方图——离散的而不是连续的。这可以节省空间。

为了获得理想的压缩数据，需要在良好的压缩和良好的信号、图像或视频质量之间找到平衡。

### 学习压缩或神经压缩

第三种压缩方式是学习压缩，在这种方法中，ML 方法试图学习信号是什么，并保留它，同时牺牲噪声。

前两种压缩方式，无损压缩和有损压缩，都是基于规则的，但在学习压缩中，神经网络被用来学习数据中的模式并提取特定的统计属性。这些学习方法更具适应性 [Gemini]。当我用无损和有损方法压缩表格或数据文件时，我必须尝试不同的方法并为每种数据类型选择最佳的方法 [*Taylor_2022*]。这是一种很好的方法，但比学习压缩的适应性差。

学习压缩模型更具适应性，模型可以不断学习，因此随着新数据的加入而改进 [*Gemini*]。

一种学习压缩方法是**自编码器**（**AE**）。它是一个回文或镜像形状的神经网络，接受数据，将其压缩到一个小空间——**潜在空间**，然后再重建输入数据。输出层看起来与输入层相同，且大小相同。这些是最大的层，其他层在中间逐渐变小，类似两个三角形，或者像一个蝴蝶结。这很有用，因为潜在空间是数据的压缩版本——没有多余的空间！第一边或三角形是编码器，第二边是解码器。编码器首先压缩数据，然后解码器重建输入数据。

在学习压缩中，良好的压缩与良好的质量之间的平衡是损失函数 [*Ioijilai*]。

#### 学习压缩的优势

+   **适应性**：与更传统的压缩方法不同，学习压缩模型可以适应复杂和多样的数据类型，比如科学数据，因为变量之间存在关系。

+   **效率**：学习压缩模型有时能比旧方法压缩得更好：

    +   学习型模型可以学习语言的统计特性和音频数据中的潜在结构，因此可以更高效地压缩数据。

    +   **卷积神经网络**（**CNNs**）特别擅长检测图像和视频中的模式和冗余，因此比 JPEG 和 MPEG 压缩更有效。在更高分辨率的数据中，这种效果更加明显。

    +   总体来说，ML 模型可能更高的复杂度能够学习更复杂的结构，从而实现更好的压缩。

+   **生成数据**：解码器可以创建更多的数据示例，就像输入数据一样。

#### 学习压缩的缺点

+   **复杂性**：它可能更有效，但也更难理解。

+   **计算成本**：这些 ML 模型可能非常昂贵，需要大量的训练和运行成本。这对于需要实时处理的应用程序（如自动驾驶车辆）是一个更大的劣势。您可能需要将简化和优化的训练模型部署到边缘或使用专门设计的硬件。

+   **有损**：学习压缩可以是有损的。这意味着需要在质量和压缩之间达成平衡。

我的背景包括创建数据压缩软件。与华威大学的其他人一起工作，我为捷豹路虎和安森美半导体制造商创建了数据压缩软件方法论。如果您想了解更多信息，我在压缩方面有一两篇论文，以及其他 ML 研究，请访问此处：[`scholar.google.com/citations?user=9lCeU94AAAAJ&hl`](https://scholar.google.com/citations?user=9lCeU94AAAAJ&hl)。

压缩后，您可以存储或传输数据。但是要取回数据，通常需要解压缩。

现在，让我们获取一些自行进行压缩的代码。

## 用 Python 3.10 编写的数据压缩代码

至少有三种类型的数据压缩：无损压缩、有损压缩和学习压缩。

让我们尝试以下代码（参见“克劳德 3 提供以下输出”）；这是使用一种常用的压缩算法**Lempel-Ziv-Markov 链算法**或**LZMA2**来压缩数据并使其在文件中变小。对于这段代码，我们将使用无损数据压缩，感谢 LZMA。

LZMA2 比 LZMA 更强大；它是一个用于未压缩和 LZMA 数据的容器格式，以及部分无法压缩的数据的压缩[ *Wiki_LZMA2，Taylor_2022* ]。

无论如何，在这里的代码应该使用 LZMA2 压缩数据。

代码由克劳德 3 LLM 提供。

### 请求克劳德 3 写一些糟糕的代码

让我们从如何获得功能正常但难以阅读的代码开始。这是为了将其与您可能已经拥有但可能更易于解释的代码进行比较。希望 LLMs 永远不会给您这种类型的代码，除非您明确要求编写质量较差的代码。

我们将使用以下提示：

```py
Please give me some Python 3.10 code that compresses tabular data with LZMA2 that is very hard to understand because it's badly written.
```

克劳德 3 提供以下输出：

```py
Here's an intentionally obfuscated and poorly written Python 3.10 code that compresses tabular data using the LZMA2 algorithm. This code is designed to be hard to understand and maintain:
python
import lzma
from typing import Any
def j(b):
    c=[]
    d=lzma.LZMACompressor(lzma.FORMAT_ALONE)
    for a in b:e=''.join([str(f)+'\t'for f in a]);e=e[:-1].encode();c.extend(list(d.compress(e)))
    c.extend(list(d.flush()))
    return bytes(c)
def l(m):
    o=[];p=lzma.LZMADecompressor();q=b''
    for s in [m]:  # Change: Iterate over a list containing the compressed data
        try:
            t=p.decompress(s)
            q+=t
        except lzma.LZMAError:
            u=p.flush()
            o.append(q.decode().split('\t'))
            q=u
    if q:
        o.append(q.decode().split('\t'))
    return o
def k(v):
    w=[];x=[]
    for y in v:x.extend([z.split('\t')for z in y])
    for a in zip(*x):w.append(list(a))
    return w
if __name__=='__main__':
    A=k([['Name','Age','City'],['Alice','25','New York'],['Bob','30','Los Angeles'],['Charlie','35','Chicago']])
    B=j(A)
    with open('compressed.lzma','wb')as C:C.write(B)
    with open('compressed.lzma','rb')as D:E=D.read()
    F=l(E)
    G=k(F)
    print(G)
```

您可以在我们的 GitHub 仓库中的**LZMAcompress_bad.py**文件中找到此代码，位置在这里：

[`github.com/PacktPublishing/Coding-with-ChatGPT-and-Other-LLMs/tree/main/Chapter4//LZMAcompress_bad.py`](https://github.com/PacktPublishing/Coding-with-ChatGPT-and-Other-LLMs/tree/main/Chapter4//LZMAcompress_bad.py)

这段代码经过了几轮调试，但你现在看到的适用于给定的小型玩具数据集。Claude 3 非常有帮助。它犯了一个错误，没能修复一个 bug，但整个过程非常简单。你只需重新提到代码（如果聊天机器人已经看过它）或把它粘贴到聊天框里（如果它没有见过），然后陈述问题，并给出错误消息所指示的那一行代码。接着，聊天机器人应该给出更好的版本。继续这样做，直到代码对你有效，然后进行测试。

这段代码几乎没有注释。变量和函数的命名非常不清晰。有时一行代码做了很多事情，尤其是在一个**for**语句的行中！条件语句中不应该包含计算或字符串操作！它在最后打印**G**，不管那是什么。变量有时是单个小写字母，有时是单个大写字母——没有任何规范。

我不太喜欢年龄是字符串的形式——它们能不能是**int**类型，或者至少是**float**类型？

它使用了**try/except**，这是好的，并且使用了不同的函数，这些函数都不大，因此从操作角度来看，代码不一定很差。

我不确定**l()**是否总会返回某些东西。

这段代码真的很不清晰。我得花一段时间才能搞明白发生了什么。

它使用了**for**语句，这在像 Python 这样的语言中不太受欢迎，因为与向量化的方式相比，它们较慢。

好吧，这就是我快速浏览所发现的内容。

### 这是 Claude 3 对其自己的坏代码示例的评价：

这段故意混淆且写得很差的代码使用 LZMA2 算法在 Python 3.10 中压缩表格数据，展示了几个使其难以阅读和理解的特点。以下是代码的一些主要问题：

+   **差的变量命名**：代码使用了像**a, b, c, d, e, f, m, n, o, p, q, r, s, t, u, v, w, x, y,** 和 **z** 这样的单字母变量名，这些变量名并没有提供有关变量用途或内容的任何有意义的信息。这使得理解每个变量代表什么，或者它是用来做什么的变得非常困难。

+   **缺乏注释**：代码完全没有注释或文档字符串，通常这些可以提供代码的目的、功能和逻辑的解释。没有注释的代码，理解它试图实现什么成了一个猜谜游戏。

+   **复杂的逻辑**：压缩和解压表格数据的逻辑分散在多个函数中（**j, l,** 和 **k**），且不必要地复杂。代码使用了嵌套的列表推导、不必要的类型转换以及晦涩的操作，导致很难跟踪执行的流程。

+   **违反编码规范**：该代码违反了多个 Python 编码规范和最佳实践，例如使用有意义的变量名、正确的缩进和格式化、以及关注点的分离。这使得熟悉 Python 规范的开发者更难理解该代码。

+   **缺乏错误处理**：该代码没有包含任何错误处理机制或异常处理，这可能导致在发生错误或遇到边缘情况时出现意外行为或崩溃。

+   **功能混合**：该代码混合了与压缩、解压缩和数据操作相关的功能，使得它更难以理解和维护。理想情况下，这些功能应分离成不同的函数或模块。

+   **晦涩的类型注解**：虽然代码使用了类型注解，但由于变量命名不规范，注解本身显得晦涩且无用——例如，类型注解**Any**被用于所有变量，这削弱了类型注解的意义。

+   **不必要的复杂性**：该代码通过使用晦涩的技术和简洁的表达式引入了不必要的复杂性，牺牲了可读性以追求简洁。这使得开发者，尤其是那些不熟悉此类技术的开发者，更难理解和维护代码。

+   **缺乏模块化**：该代码没有以模块化的方式设计，使得它难以在更大的应用程序或库中集成或重用。

+   **扩展性有限**：由于代码的晦涩性质以及缺乏适当的文档或注释，扩展或修改代码变得困难，且容易引发意外后果或引入新错误。

总体而言，该代码展示了许多不良编码实践和故意的晦涩处理，导致它非常难以阅读、理解和维护。虽然这一练习对理解处理糟糕编写代码的挑战非常有价值，但它也提醒我们在实际软件开发中，编写清晰、易读和文档完善的代码的重要性。

以上是 Claude 3 输出的结尾。

接下来，我们必须找到一些既能正常运行又易于阅读的代码，并遵循良好的实践。再次，我选择了当时最好的编码 LLM——Claude 3。

## 让我们来看一些写得很好的代码

现在我们将请 Claude 3 提供一些好的代码，类似于之前的例子：使用 Python 进行数据压缩：

```py
Please give me some Python 3.10 code that compresses tabular data with LZMA2 that is easy to understand because it's well-written.
```

Claude 3 给出了以下输出：

```py
Here's a well-written Python 3.10 code that compresses tabular data using the LZMA2 algorithm. This code follows best practices and is designed to be easy to read and understand:
python
import lzma
from typing import List
def compress_tabular_data(tabular_data: List[List[str]]) -> bytes:
    """
    Compresses tabular data using the LZMA2 algorithm.
    Args:
        tabular_data: A list of lists representing tabular data, where each inner list is a row.
    Returns:
        A bytes object containing the compressed tabular data.
    """
    compressor = lzma.LZMACompressor(lzma.FORMAT_ALONE)
    compressed_data = []
    for row in tabular_data:
        row_bytes = '\t'.join(row).encode() + b'\n'
        compressed_data.append(compressor.compress(row_bytes))
    compressed_data.append(compressor.flush())
    return b''.join(compressed_data)
def decompress_tabular_data(compressed_data: bytes) -> List[List[str]]:
    """
    Decompresses tabular data compressed using the LZMA2 algorithm.
    Args:
        compressed_data: A bytes object containing the compressed tabular data.
    Returns:
        A list of lists representing the decompressed tabular data, where each inner list is a row.
    """
    decompressor = lzma.LZMADecompressor()
    decompressed_data = decompressor.decompress(compressed_data)
    rows = [row.split('\t') for row in decompressed_data.decode().split('\n') if row]
    return rows
if __name__ == '__main__':
    tabular_data = [
        ['Name', 'Age', 'City'],
        ['Alice', '25', 'New York'],
        ['Bob', '30', 'Los Angeles'],
        ['Charlie', '35', 'Chicago']
    ]
    # Compress tabular data
    compressed_data = compress_tabular_data(tabular_data)
    # Save compressed data to a file
    with open('compressed_data.lzma', 'wb') as file:
        file.write(compressed_data)
    # Load compressed data from the file
    with open('compressed_data.lzma', 'rb') as file:
        compressed_data = file.read()
    # Decompress tabular data
    decompressed_data = decompress_tabular_data(compressed_data)
    print(decompressed_data)
```

该代码可以成功执行。你可以在 GitHub 仓库的**LZMAcompress_good.py**文件中找到这段代码，链接如下：[`github.com/PacktPublishing/Coding-with-ChatGPT-and-other-LLMs/tree/main/Chapter4//LZMAcompress_good.py`](https://github.com/PacktPublishing/Coding-with-ChatGPT-and-other-LLMs/tree/main/Chapter4//LZMAcompress_good.py)。

### Claude 3 对其自己写的好代码的评价

该代码易于阅读和理解，原因如下：

+   **清晰的函数名称**：函数名**compress_tabular_data**和**decompress_tabular_data**清楚地描述了每个函数的目的。

+   **类型提示**：代码使用类型提示来指定函数参数和返回值的预期类型，使得理解输入和输出格式更加容易。

+   **文档字符串**：每个函数都有一个文档字符串，简要描述了函数的目的、参数和返回值。

+   **有意义的变量名**：代码中使用的变量名具有描述性，能够自解释，使得理解每个变量的含义变得更容易。

+   **关注点分离**：代码将压缩和解压缩逻辑分成了两个独立的函数，使得维护和扩展变得更加容易。

+   **正确的缩进和格式化**：代码的缩进和格式化正确，使得代码可读且视觉上令人愉悦。

+   **错误处理**：虽然在这个示例中没有明确显示，但代码可以扩展以包括错误处理机制，以处理在压缩或解压缩过程中可能出现的异常或错误。

+   **模块化设计**：代码以模块化的方式设计，便于集成到更大的应用程序或库中。

+   **使用标准库**：代码使用了 Python 标准库中的标准**lzma**模块，该模块有良好的文档支持并被广泛使用。

+   **示例用法**：代码在**if __name__ == '__main__'**块中包含了一个示例用法，使得理解如何使用这些函数变得更加容易。

总体来说，这段代码遵循了编写干净、可读且易于维护的 Python 代码的最佳实践。对于熟悉 Python 和 LZMA2 压缩算法的开发者来说，应该很容易理解。

这就是 Claude 3 输出的结束以及我们讨论其更具可读性版本的 Python LZMA 压缩代码的优点。

现在，让我们来看看一些具体的因素，这些因素使得代码容易或难以阅读和理解。

# 什么因素使得代码容易或难以阅读？

让我们看看一些可能导致可读性差的实践，然后再看看如何使可读性尽可能好。

## 为什么阅读代码会很难？

阅读代码比写代码要困难得多，至少有三个原因：

+   写代码时，你只需要关注你正在编写的功能的变量、算法、数据和结构。而阅读代码时，你必须记住所有关于更广泛功能和潜在功能的信息，这是你从阅读代码中猜测出来的。

+   思考比理解别人思考的过程要容易——要简单得多。

+   那不是你的代码——你的“孩子”，所以它可能不如你自己写的好，至少这是很多程序员的想法。这是情感方面的因素。或者，你可能会认为这段代码是由一个更聪明、更有经验的人编写的，所以你无法理解它。

要理解别人的代码，你需要在脑中有一个代码的工作模型，并了解项目的目标 [ *GameDev_Dieterich* ]。

不要害怕阅读他人的代码并理解它。你可能会很好地利用它，而不必重新编写代码。

这让我想起了我曾经需要阅读并使用来自捷克同事的代码的时光。

除了试图理解 C 代码（而我通常用 C#、R 或 MATLAB 编程）并学习一些物理知识外，其中一些变量是捷克语的，而我不懂捷克语。所以，我还得学一点捷克语。幸运的是，我喜欢学习语言，也已经对物理有一定了解。最终，我理解了代码，我们的合作也非常顺利，包括互访各自的国家，英国和捷克。我们做出了一些很棒的产品。

当然，如果你可以与原始编码者沟通，你可以向他们询问代码问题。要有礼貌——那是他们的心血结晶，他们可能花了很多时间和精力来创建它。

## 可读性代码的注意事项——如何编写可读的代码

根据程序员的意见，以下是使代码难以或容易阅读的因素。

### 不要做这些事情

+   一些不一致的地方，如在一个文件或一组文件中使用多种不同的制表符/缩进样式，或者先检查空指针后又检查非空指针，会让试图理解代码并确保其正常工作的读者感到困惑。命名规范也要保持一致。

+   使用不具描述性的变量和函数名，比如使用名人名字或电影标题，即使脚本与电影毫无关系！

    +   使用非常相似的函数名会让定位 bug 变得极其困难，例如**functionl()**和**functionL()**。使未来的程序员（包括你自己）能够轻松区分函数。

+   **switch**语句的过度使用：当编辑他人的代码时，如果你的情况已经涵盖，不要仅仅添加一个**switch**语句，因为那可能导致穿透情况随机分布在标准情况之间，使代码难以阅读和理解 [ *O’Reilly_Algo_Flow,* *Quora_Chen, StackOverflow_switch* ]。

+   使用过多嵌套语句/层级的**if**语句：如果一个**if**语句出现在屏幕的中间，可能说明嵌套层次太多。

### 做这些事情

+   代码应该包含不同的函数，并且函数应该简洁：

    +   函数名应反映它们的功能，且函数应该非常简洁：每个函数只负责一个任务，而不是多个任务。拥有这样的简单函数能使它们更易理解。

+   过度使用**if**语句的替代方案是使用**switch**语句。**switch**语句通常比**if/else**语句要快，特别是当条件较多时 [ *O’Reilly_Algo_Flow,* *Quora_Chen, StackOverflow_switch* ]。

### 一些有效使用大语言模型（LLMs）进行代码生成的策略

#### 提供清晰具体的提示语：

+   *精确*：清晰地表达所需的功能或任务。

+   *提供背景*：提供与项目或代码库相关的背景信息。

+   *指定语言*：指明你希望代码使用的编程语言及其版本，如 Python 3.10。

#### 迭代与优化：

+   *评估输出*：仔细审查生成的代码，检查其正确性和可读性。

+   *提供反馈*：如果代码有错误或需要改进，向 LLM 提供具体反馈——例如，“这段代码可以工作，但请使其更易读，尤其是在**Xzy**函数中。”

+   *迭代*：继续优化提示语并迭代，直到得到理想的结果。

#### 利用 LLM 的能力：

+   *代码补全*：使用 LLM 建议代码片段或完成未完成的代码行。

+   *函数生成*：要求 LLM 根据给定的描述生成完整的函数。

+   *代码解释*：让 LLM 解释现有代码，以便更好地理解其功能。

#### 考虑上下文信息：

+   *项目结构*：提供关于项目结构、依赖关系和约定的信息。

+   *代码风格*：指定首选的编码风格或指南。

+   *最佳实践*：指示 LLM 遵循最佳实践，避免常见的陷阱。

#### 将人类专业知识与 LLM 能力结合：

+   *人工监督*：始终让人类开发者审查和验证生成的代码。

+   *领域知识*：利用人类专业知识，确保代码符合特定领域的要求。

这是一个示例提示语：

```py
"Write a Python function to calculate the factorial of a given non-negative integer."
```

这是 LLM 的回复：

Python：

```py
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)
```

使用代码时要小心。

注意事项

虽然这段代码是正确的，但人类开发者可能会建议改进，比如添加类型提示或对较大的 n 值使用更高效的迭代方法。

请记住

LLM 是代码生成的有价值工具，但它们应与人类专业知识结合使用，以确保代码的最高质量和可维护性。通过遵循这些指南并迭代优化提示语，你可以有效地利用 LLM 来提高编程生产力。

学习阅读代码确实需要练习，所以要不断练习，随着时间的推移，你会发现它变得更加容易和高效。协作是我们用来创造惊人成果的最强大工具之一！

本节讨论了在使用 LLM 编写或生成代码时的坏习惯与最佳实践：不要使用太多**if**语句，不要让代码过于复杂或没有注释，也不要随便添加东西；要和 LLM 来回互动，提供所需的语言版本和上下文，如项目结构，自行评估代码，编写简单的函数，使用具有描述性的命名，良好地文档化代码，遵循编码标准和规范，处理错误等等。我们看了一些难以理解的代码和一些更容易阅读和理解的代码，以及如何从 Packt GitHub 仓库获取代码。下一节将讨论如何总结你不理解的代码。

# 总结代码以便理解

本节讲述了如何总结你所给出的或找到的、你无法完全理解的代码，或者你不太清楚它的运行机制。或者，也许你只是觉得不想花时间脑力思考，反而希望通过自动化这一过程来节省精力和时间。

在这里，我请 Alphabet 的 Gemini 给我一些我不理解的代码：

```py
Could you write a longish, complicated script of Python 3.10 code and don't tell me what it does, very few comments and no text after the code.
```

这是来自 Gemini（+ Claude 3）的代码：

```py
import requests
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
def fetch_data(url, params):
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise an exception for non-200 status codes
    return response.json()
def calculate_indicators(data):
    # Convert 'Close' column to float
    data['Close'] = data['Close'].astype(float)
    data["SMA"] = data["Close"].rolling(window=20).mean()
    data["RSI"] = calculate_rsi(data["Close"])
    return data
def calculate_rsi(closes, window=14):
    delta = closes.diff()
    delta = delta.dropna()
    gains = delta[delta > 0]
    losses = -delta[delta < 0]
    avg_gain = gains.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1/window, min_periods=window).mean().abs()
    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    return rsi
def plot_data(data, title):
    data[["Close", "SMA"]].plot(figsize=(12, 6), style=["b-", "g-"])
    data["RSI"].plot(ax=plt.twinx(), style="r-")
    plt.title(title)
    plt.show()
if __name__ == "__main__":
    base_url = "https://api.binance.com/api/v3/klines"
    symbol = "BTCUSDT"
    interval = "1d"  # Daily data
    today = datetime.utcnow()
    one_year_ago = today - timedelta(days=365)
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(one_year_ago.timestamp() * 1000),  # Convert to Unix timestamp in milliseconds
        "endTime": int(today.timestamp() * 1000),  # Convert to Unix timestamp in milliseconds
    }
    data = fetch_data(base_url, params)
    data = pd.DataFrame(data)
    data.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time", "Quote Asset Volume", "Number of Trades", "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"]
    data["Open Time"] = pd.to_datetime(data["Open Time"], unit="ms")
    data.set_index("Open Time", inplace=True)
    data = calculate_indicators(data.copy())
    plot_data(data, f"{symbol} Price & Indicators (Past Year)")
```

重要提示

我请 Claude 3 Sonnet 来调试这段代码，因为它无法运行——这是最终版本（Gemini + Claude），但解释是在调试之前完成的。

在代码文件中，这个文件被命名为**crypto_price_and_indicators.py**，这是 Claude 所命名的，但与我曾给它起的名字非常接近。

我其实不太想运行我完全不理解的代码——从理论上讲，这可能是危险的，尽管我大多数时候信任 Alphabet。所以，让我们看看 ChatGPT 能告诉我们什么。

这是传递给 ChatGPT 的提示：

```py
Explain code.
Hi, please explain this code to me, I have no idea what it does but it's in Python 3.<code>
```

这是 ChatGPT 的回应：

```py
This Python code fetches historical cryptocurrency price data from the Binance API, calculates some technical indicators (Simple Moving Average and Relative Strength Index), and then plots the data along with these indicators using pandas and matplotlib libraries.
Here’s a breakdown of what each part of the code does:
1\. Importing libraries: The code imports necessary libraries such as requests for making HTTP requests, datetime for handling dates and times, pandas for data manipulation and analysis, and matplotlib.pyplot for plotting graphs.
2\. The fetch_data function: This function sends an HTTP GET request to the specified URL (the Binance API in this case) with the provided parameters. It then checks if the response status code is not 200 (indicating a successful request) and raises an exception if not. Finally, it returns the response data in JSON format.
3\. The calculate_indicators function: This function calculates two technical indicators: rolling() function from pandas to calculate the SMA, and a custom function, calculate_rsi(), to compute the RSI.
4\. The calculate_rsi function: This function calculates the Relative Strength Index (RSI) using the closing prices of cryptocurrency. RSI is a momentum oscillator that measures the speed and change of price movements. It computes the average gains and losses over a specified window (the default is 14 days) and then uses these averages to calculate the RSI formula.
5\. The plot_data function: This function plots the closing price and SMA on the primary y axis and the RSI on the secondary y axis. It uses Matplotlib to create the plot and sets the title as provided.
6\. Main code block: The main block sets up parameters such as the base URL of the Binance API, the cryptocurrency symbol (BTCUSDT for Bitcoin priced in USDT), the data interval (1d for daily data), and start and end times for fetching historical data (one year ago from today), and then fetches the data using the fetch_data function.
7\. After fetching the data, it converts it to a pandas DataFrame, sets the column names, converts the timestamps to datetime format, and sets the timestamp column as the index. Then, it calculates the indicators using the calculate_indicators function and plots the data using the plot_data function.
8\. Finally, it executes the main block only if the script is run directly (not imported as a module). It fetches the data, processes it, and plots the results for the past year.
```

ChatGPT 为我们分解了函数，并给出了整体概述，包括主函数的作用，甚至解释了为什么要导入那些代码库。ChatGPT 还解释了什么是 RSI。这使得读者对脚本的功能和每个元素的作用充满信心。对我来说，这看起来相当不错。如果你不确定解释是否做得很好，可以将代码和解释复制到 Claude 或 Gemini 中，或者其他聊天机器人中，检查解释是否在各个部分都正确。

我自己检查代码后，认为这是 ChatGPT 给出的准确解释。

这段代码成功运行，并生成了比特币（BTC）价格与美元稳定币 Tether（USDT）和 SMA 指标的图表。请参见*图 4.1*。

![图 4.1：比特币历史价格（波动的蓝色线）与 USDT 和 SMA 指标（平滑的绿色线）对比。比特币价格在左侧垂直轴上[30000 到 70000]，SMA 在右侧垂直轴上[-0.04 到 0.04]。](img/B21009_04_1.jpg)

图 4.1：比特币历史价格（波动的蓝色线）与 USDT 和 SMA 指标（平滑的绿色线）对比。比特币价格在左侧垂直轴上[30000 到 70000]，SMA 在右侧垂直轴上[-0.04 到 0.04]。

现在你可以让代码向你解释了，可能只有再复杂一点的解释步骤，尤其是当你想发布你的代码时，你很可能希望完全文档化它，这样其他程序员可以通过阅读手册，真正顺利地使用它。

下一节是讲解如何为你的代码编写文档，让其他人了解它的用途、如何排除故障以及如何最好地使用你的代码。

# 生成文档

让我们来看看解释前述代码与该代码完整文档之间的区别。通过使用前述代码来生成文档，我们来做一个直接的对比。

我请 Claude（因为我认为它目前是表现最好的用于代码的 LLM 聊天机器人）为我提供前述代码的完整文档。

## crypto_price_and_indicators.py 的文档

Claude 3 Sonnet 的回答如下：

```py
"Here's the complete and detailed documentation for the provided Python code:
Cryptocurrency Price and Indicators Fetcher
This Python script allows you to fetch historical cryptocurrency price data from the Binance API and calculate technical indicators such as the Simple Moving Average (SMA) and the Relative Strength Index (RSI). The script also provides a visualization of the price data and indicators using the Matplotlib library.
Prerequisites
Before running this script, you'll need to have the following Python libraries installed:
requests: Used to make HTTP requests to the Binance API.
pandas: Used for data manipulation and analysis.
matplotlib: Used for data visualization.
You can install these libraries using pip:
pip install requests pandas matplotlib
Usage
Save the provided Python code to a file (e.g., crypto_price_and_indicators.py).
Open a terminal or command prompt and navigate to the directory where the file is saved.
Run the script using the following command:
python crypto_price_and_indicators.py
The script will fetch the daily price data for Bitcoin (BTC) against the US Dollar (USDT) for the past year from the Binance API. It will then calculate the SMA and RSI indicators and display a plot with the price data and indicators.
Functions
The script contains the following functions:
fetch_data(url, params)
This function sends an HTTP GET request to the specified URL with the provided parameters and returns the API response as a JSON object.
Parameters:
url (str): The URL of the API endpoint.
params (dict): A dictionary containing the query parameters for the API request.
Returns:
The API response as a JSON object.
calculate_indicators(data)
This function calculates the SMA and RSI indicators for the provided price data.
Parameters:
data (pandas.DataFrame): A DataFrame containing the price data.
Returns:
The input DataFrame with two additional columns: SMA (Simple Moving Average) and RSI (Relative Strength Index).
calculate_rsi(closes, window=14)
This function calculates the Relative Strength Index (RSI) for the provided price data.
Parameters:
closes (pandas.Series): A Series containing the closing prices.
window (int, optional): The window size for the RSI calculation. Default is 14.
Returns:
A pandas.Series containing the RSI values.
plot_data(data, title)
This function creates a plot of the price data and indicators using Matplotlib.
Parameters:
data (pandas.DataFrame): A DataFrame containing the price data and indicators.
title (str): The title of the plot.
Returns:
None (displays the plot).
Main Execution
The main execution block of the script performs the following steps:
Defines the base URL of the Binance API and the cryptocurrency symbol (BTC/USDT) and interval (1 day).
Calculates the start and end times for fetching data from the past year.
Constructs the API request parameters.
Fetches the price data from the Binance API using the fetch_data function.
Converts the fetched data into a pandas DataFrame and performs some data preprocessing.
Calculates the SMA and RSI indicators using the calculate_indicators function.
Plots the price data and indicators using the plot_data function.
Note: The script is designed to fetch and analyze historical data for Bitcoin (BTC) against the US Dollar (USDT) for the past year. If you want to analyze a different cryptocurrency pair or a different time period, you'll need to modify the symbol and interval variables, as well as the startTime and endTime parameters in the params dictionary."
```

该文档结构非常清晰，详细介绍了库的要求；它列出了每个函数：参数、每个函数返回的内容，以及当然，函数所做的总体总结（在开头部分）。它甚至给出了如何改进功能的建议。

说实话，说这项 LLM 技术是“革命性的”绝对不算夸张！这项技术让编程和文档编写变得异常简单和快速！

这比手工写一切轻松多了！我们真幸运！是的，这本书中的例子虽然简短，但 LLMs 已经做了大量的繁重工作，而且它会变得越来越好。

重要提示

尽管这项技术令人震撼，但这是 2024 年初的技术，而更强大、更具适应性和智能的技术正在以加速的速度涌现。所以，这项技术很快就会过时。

*保持最新！不断寻找新的* *编程工具。*

这就是 *第四章* – 概要即将呈现。

# 概要

在本章中，我们探讨了如何使用 LLMs 生成更易读的代码，总结和解释现有代码，并为现有代码生成文档。坦白说，简直让人震惊！使用 LLMs 编程太简单了！

在 *第五章* 中，我们将讨论 LLM 生成的代码中的偏见和伦理问题。Gemini 在 2024 年 2 月引发了一些问题，公众对此并不买账。这始终是一个学习过程，希望人类和人工智能在生成改变世界的 AI 方面能越来越好。

# 参考文献

+   *GameDev_Dieterich* ： “大多数程序员看不懂代码”，Robert Dieterich，[`www.gamedeveloper.com/programming/most-programmers-can-t-read-code#close-modal`](https://www.gamedeveloper.com/programming/most-programmers-can-t-read-code#close-modal)

+   *Ioijilai* ： “神经/学习图像压缩：概述”，Ioijilai，`medium.com/@loijilai_43024/learned-image-compression-an-overview-625f3ab709f2`

+   *O’Reilly_Algo_Flow*：“第四章 算法与流程控制”，Nicholas C. Zakas, [`www.oreilly.com/library/view/high-performance-javascript/9781449382308/ch04.html#:~:text=As%20it%20turns%20out%2C%20the,than%20it%20is%20for%20switch%20`](https://www.oreilly.com/library/view/high-performance-javascript/9781449382308/ch04.html#:~:text=As%20it%20turns%20out%2C%20the,than%20it%20is%20for%20switch%20)

+   *Quora_Chen*：“是什么让一些代码难以阅读，而另一些代码易于阅读？”，Howard Chen, [`www.quora.com/What-makes-some-code-hard-to-read-and-some-code-easy-to-read`](https://www.quora.com/What-makes-some-code-hard-to-read-and-some-code-easy-to-read)

+   *StackOverflow_switch*：“何时使用 If-else if-else 而非 switch 语句，反之亦然【重复问题】：关于可读性：”，Sonu Oommen, [`stackoverflow.com/questions/427760/when-to-use-if-else-if-else-over-switch-statements-and-vice-versa`](https://stackoverflow.com/questions/427760/when-to-use-if-else-if-else-over-switch-statements-and-vice-versa)

+   *Taylor_2022*：Phillip Taylor, Nathan Griffiths, Vince Hall, Zhou Xu, Alex Mouzakitis，“监督学习与压缩的特征选择，”[`www.tandfonline.com/doi/full/10.1080/08839514.2022.2034293`](https://www.tandfonline.com/doi/full/10.1080/08839514.2022.2034293)

+   *Wiki_Cosine*：“离散余弦变换”，Wikipedia, [`en.wikipedia.org/wiki/Discrete_cosine_transform`](https://en.wikipedia.org/wiki/Discrete_cosine_transform)

+   *WikiDataComp*：“数据压缩比”，[`en.wikipedia.org/wiki/Data_compression_ratio#:~:text=Definition,-Data%20compression%20ratio&text=Thus%2C%20a%20representation%20that%20compresses,implicit%20ratio%2C%205%2F1`](https://en.wikipedia.org/wiki/Data_compression_ratio#:~:text=Definition,-Data%20compression%20ratio&text=Thus%2C%20a%20representation%20that%20compresses,implicit%20ratio%2C%205%2F1)

+   *Wiki_LZMA2*：“Lempel–Ziv–Markov 链算法”，Wikipedia, [`en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Markov_chain_algorithm`](https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Markov_chain_algorithm)

+   *Wiki_Wavelets*：“小波变换”，[`en.wikipedia.org/wiki/Wavelet_transform#Wavelet_compression`](https://en.wikipedia.org/wiki/Wavelet_transform#Wavelet_compression)
