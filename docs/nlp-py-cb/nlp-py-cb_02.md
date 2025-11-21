# 第二章：原始文本、来源和规范化

本章将涵盖以下主题：

+   字符串操作的重要性

+   深入了解字符串操作

+   在 Python 中读取 PDF 文件

+   在 Python 中读取 Word 文档

+   处理 PDF、DOCX 和纯文本文件，并从中创建用户定义的语料库

+   从 RSS 源读取内容

+   使用 BeautifulSoup 解析 HTML

# 介绍

在前一章中，我们探讨了 NLTK 内建语料库。这些语料库的组织和规范化非常好，适合使用，但在你处理行业问题时，情况并不总是如此。更不用说规范化和组织了，甚至我们可能根本得不到需要的统一格式的数据。本章的目标是介绍一些 Python 库，帮助你从二进制格式中提取数据：PDF 和 Word DOCX 文件。我们还将看一些可以从 Web 源（如 RSS）提取数据的库，以及一个帮助你解析 HTML 并提取文档原始文本的库。我们还将学习如何从异构来源提取原始文本、对其进行规范化，并从中创建用户定义的语料库。

在本章中，你将学习七个不同的示例。正如章节名称所示，我们将学习如何从 PDF 文件、Word 文档和 Web 中获取数据。PDF 和 Word 文档是二进制文件，而在 Web 上，你将以 HTML 格式获得数据。因此，我们还将对这些数据进行规范化和原始文本转换任务。

# 字符串操作的重要性

作为一名 NLP 专家，你将处理大量文本内容。当你处理文本时，必须了解字符串操作。我们将从几个简短而精炼的示例开始，帮助你理解 Python 中的`str`类及其操作。

# 准备开始…

对于这个例子，你只需要 Python 解释器和一个文本编辑器，其他的都不需要。我们将查看`join`、`split`、加法、乘法操作符和索引。

# 如何做…

1.  创建一个名为`StringOps1.py`的新 Python 文件。

1.  定义两个对象：

```py
namesList = ['Tuffy','Ali','Nysha','Tim' ]
sentence = 'My dog sleeps on sofa'
```

第一个对象`nameList`是一个包含一些名称的`str`对象的列表，第二个对象`sentence`是一个`str`类型的句子。

1.  首先，我们将了解`join`功能及其作用：

```py
names = ';'.join(namesList)
print(type(names), ':', names)
```

`join()`函数可以在任何`string`对象上调用。它接受一个`str`对象的列表作为参数，并将所有的`str`对象连接成一个单一的`str`对象，调用字符串对象的内容作为连接分隔符。它返回这个对象。运行这两行代码后，你的输出应该如下：

```py
<class 'str'> : Tuffy;Ali;Nysha;Tim
```

1.  接下来，我们将查看`split`方法：

```py
wordList = sentence.split(' ')
print((type(wordList)), ':', wordList)
```

对字符串调用的`split`函数会将其内容分割成多个`str`对象，创建一个包含这些对象的列表，并返回该列表。该函数接受一个`str`参数，作为分割的标准。运行代码后，你将看到以下输出：

```py
<class 'list'> : ['My', 'dog', 'sleeps', 'on', 'sofa']
```

1.  算术运算符 `+` 和 `*` 也可以与字符串一起使用。添加以下代码行并查看输出：

```py
additionExample = 'ganehsa' + 'ganesha' + 'ganesha'
multiplicationExample = 'ganesha' * 2
print('Text Additions :', additionExample)
print('Text Multiplication :', multiplicationExample)
```

这次我们将首先看到输出，然后讨论它是如何工作的：

```py
Text Additions: ganehsaganeshaganesha
Text Multiplication: ganeshaganesha
```

`+` 运算符被称为连接操作。它会生成一个新的字符串，将多个字符串连接成一个单一的 `str` 对象。使用 `*` 运算符，我们也可以对字符串进行乘法操作，如前面的输出所示。此外，请注意，这些操作不会添加任何额外的内容，例如在字符串之间插入空格。

1.  让我们来看看字符串中字符的索引。添加以下代码行：

```py
str = 'Python NLTK'
print(str[1])
print(str[-3])
```

首先，我们声明一个新的 `string` 对象。然后我们访问字符串中的第二个字符（`y`），这表明它是直接的。接下来是有点棘手的部分；Python 允许你在访问任何列表对象时使用负索引；`-1` 表示最后一个元素，`-2` 表示倒数第二个，依此类推。例如，在前面的 `str` 对象中，索引 `7` 和 `-4` 是相同的字符，`N`：

```py
Output: <class 'str'> : Tuffy;Ali;Nysha;Tim
<class 'list'> : ['My', 'dog', 'sleeps', 'on', 'sofa']
Text Additions : ganehsaganeshaganesha
Text Multiplication : ganeshaganesha
y
L
```

# 它是如何工作的…

我们使用 `split()` 和 `join()` 函数分别从字符串中创建字符串列表，并从字符串列表中创建字符串。然后我们看到了如何使用一些算术运算符与字符串进行操作。请注意，我们不能对字符串使用 "`-`"（取反）和 "`/`"（除法）运算符。最后，我们看到了如何访问字符串中的单个字符，特别是我们可以在访问字符串时使用负索引。

这个步骤相当简单直接，目的是介绍一些 Python 允许的常见和不常见的字符串操作。接下来，我们将从中断处继续，进行更多字符串操作。

# 深入了解字符串操作

在上一个步骤的基础上，我们将看到子字符串、字符串替换以及如何访问字符串的所有字符。

让我们开始吧。

# 如何做…

1.  创建一个名为`StringOps2.py`的新 Python 文件，并定义以下字符串对象 `str`：

```py
str = 'NLTK Dolly Python'
```

1.  让我们访问 `str` 对象中以第四个字符结尾的子字符串。

```py
print('Substring ends at:',str[:4])
```

正如我们知道索引从零开始，这将返回包含从零到三的字符的子字符串。当你运行时，输出将是：

```py
Substring ends at: NLTK
```

1.  现在我们将访问从某个位置开始直到结尾的 `str` 对象的子字符串：

```py
print('Substring starts from:',str[11:] )
```

这告诉解释器返回从索引 `11` 到字符串结尾的子字符串。当你运行时，以下输出将显示：

```py
Substring starts from: Python
```

1.  让我们从 `str` 对象中访问 `Dolly` 子字符串。添加以下一行：

```py
print('Substring :',str[5:10])
```

上面的语法返回从索引 `5` 到 `10` 的字符，不包括第 10 个字符。输出为：

```py
Substring : Dolly
```

1.  现在，是时候展示一个花式技巧了。我们已经看到负索引如何在字符串操作中工作。让我们尝试以下操作，看看它是如何工作的：

```py
print('Substring fancy:', str[-12:-7])
Run and check the output, it will be –
Substring fancy: Dolly
```

完全类似于上一步！继续做回溯计算：`-1` 为最后一个字符，`-2` 为倒数第二个，以此类推。这样，你就会得到索引值。

1.  让我们检查一下带有`if`的`in`操作符：

```py
if 'NLTK' in str:
  print('found NLTK')
```

运行前面的代码并检查输出；它将是：

```py
found NLTK
```

尽管看起来很复杂，`in`操作符实际上只是检查左侧的字符串是否为右侧字符串的子串。

1.  我们将在一个`str`对象上使用简单的`replace`函数：

```py
replaced = str.replace('Dolly', 'Dorothy')
print('Replaced String:', replaced)
```

`replace`函数只需要两个参数。第一个是需要被替换的子字符串，第二个是将替换它的新子字符串。它返回一个新的`string`对象，并且不会修改它所调用的对象。运行并查看以下输出：

```py
Replaced String: NLTK Dorothy Python
```

1.  最后但同样重要的是，我们将遍历`replaced`对象并访问每个字符：

```py
print('Accessing each character:')
for s in replaced:
  print(s)
```

这将把替换对象中的每个字符打印到新的一行。让我们看看最终的输出：

```py
Output: Substring ends at: NLTK
Substring starts from: Python
Substring : Dolly
Substring fancy: Dolly
found NLTK
Replaced String: NLTK Dorothy Python
Accessing each character:
N
L
T
K
D
o
r
o
t
h
y
P
y
t
h
o
n
```

# 它是如何工作的……

一个`string`对象实际上就是一组字符。正如我们在第一步中看到的，我们可以使用`for`语法来访问字符串中的每个字符，正如我们访问列表一样。在方括号中的字符`:`表示我们想要获取列表的一部分；`:`后跟数字表示我们想要从零开始并结束于该索引减去 1 的子列表。类似地，数字后面跟`:`表示我们想要从给定的数字开始，直到列表的末尾。

这结束了我们简短的探索字符串操作的旅程。接下来，我们将进入文件、在线资源、HTML 等内容。

# 在 Python 中读取 PDF 文件

我们从一个简单的教程开始，教你如何从 Python 访问 PDF 文件。为此，你需要安装`PyPDF2`库。

# 准备工作

我们假设你已经安装了`pip`。然后，要在 Python 2 和 3 中使用`pip`安装`PyPDF2`库，你只需从命令行运行以下命令：

```py
pip install pypdf2
```

如果你成功安装了库，那么我们可以继续进行。除此之外，我还请求你从这个链接下载一些测试文档，我们将在本章中使用它们：[`www.dropbox.com/sh/bk18dizhsu1p534/AABEuJw4TArUbzJf4Aa8gp5Wa?dl=0`](https://www.dropbox.com/sh/bk18dizhsu1p534/AABEuJw4TArUbzJf4Aa8gp5Wa?dl=0)。

# 如何做……

1.  创建一个名为`pdf.py`的文件，并向其中添加以下导入语句：

```py
from PyPDF2 import PdfFileReader
```

它从`PyPDF2`库中导入`PdfFileReader`类。

1.  将此 Python 函数添加到应读取文件并返回 PDF 文件中所有文本的文件中：

```py
def getTextPDF(pdfFileName, password = '')
```

这个函数接受两个参数，一个是你想要读取的 PDF 文件的路径，另一个是 PDF 文件的密码（如果有的话）。如你所见，`password`参数是可选的。

1.  现在让我们定义这个函数。在函数下方添加以下几行：

```py
pdf_file = open(pdfFileName, 'rb')
read_pdf = PdfFileReader(pdf_file)
```

第一行以读取和向后查找模式打开文件。第一行实际上是 Python 的打开文件命令/函数，它仅以二进制模式打开非文本文件。第二行会将打开的文件传递给`PdfFileReader`类，该类将处理 PDF 文档。

1.  下一步是解密受密码保护的文件（如果有的话）：

```py
if password != '':
  read_pdf.decrypt(password)
```

如果在函数调用中提供了密码，我们将尝试使用该密码解密文件。

1.  现在我们将从文件中读取文本：

```py
text = []
for i in range(0,read_pdf.getNumPages()-1):
  text.append(read_pdf.getPage(i).extractText())
```

我们创建一个字符串列表，并将每一页的文本添加到该列表中。

1.  返回最终输出：

```py
return '\n'.join(text)
```

我们通过将列表中所有字符串对象的内容连接成一个新行来返回单个 `string` 对象。

1.  在与 `pdf.py` 同一文件夹中创建另一个名为 `TestPDFs.py` 的文件，并添加以下导入语句：

```py
import pdf
```

1.  现在我们将打印出来自几个文档的文本，其中一个有密码保护，另一个是普通文档：

```py
pdfFile = 'sample-one-line.pdf'
pdfFileEncrypted = 'sample-one-line.protected.pdf'
print('PDF 1: \n',pdf.getTextPDF(pdfFile))
print('PDF 2: \n',pdf.getTextPDF(pdfFileEncrypted,'tuffy'))
```

**输出**：食谱的前六个步骤仅创建一个 Python 函数，并且在控制台上不会生成任何输出。第七和第八个步骤将输出以下内容：

```py
This is a sample PDF document I am using to demonstrate in the tutorial.

This is a sample PDF document

password protected.
```

# 它是如何工作的…

`PyPDF2` 是一个纯 Python 库，我们用它来从 PDF 中提取内容。该库有更多的功能，例如裁剪页面、叠加图像以进行数字签名、创建新的 PDF 文件等等。然而，作为 NLP 工程师或进行任何文本分析任务时，你的目的是读取文件内容。在第 `2` 步中，重要的是以反向查找模式打开文件，因为 `PyPDF2` 模块尝试从文件的末尾读取内容。此外，如果 PDF 文件受到密码保护且未解密，Python 解释器将抛出 `PdfReadError` 错误。

# 在 Python 中读取 Word 文档

在本食谱中，我们将看到如何加载和读取 Word/DOCX 文档。用于读取 DOCX 文档的库更为全面，因为我们还可以查看段落边界、文本样式，并执行所谓的“运行”。我们将看到所有这些内容，因为它们在文本分析任务中非常重要。

如果你没有 Microsoft Word，你可以始终使用开源版本的 LibreOffice 和 OpenOffice 来创建和编辑 `.docx` 文件。

# 正在准备…

假设你已经在机器上安装了 `pip`，我们将使用 pip 安装一个名为 `python-docx` 的模块。不要将其与名为 `docx` 的另一个库混淆，它是完全不同的模块。我们将从 `python-docx` 库中导入 `docx` 对象。以下命令在命令行中执行时，将安装该库：

```py
pip install python-docx
```

成功安装库后，我们可以继续操作。在本食谱中，我们将使用一个测试文档，如果你已经下载了本章第一个食谱中提供的所有文档，你应该拥有相关文档。如果没有，请从 [`www.dropbox.com/sh/bk18dizhsu1p534/AABEuJw4TArUbzJf4Aa8gp5Wa?dl=0`](https://www.dropbox.com/sh/bk18dizhsu1p534/AABEuJw4TArUbzJf4Aa8gp5Wa?dl=0) 下载 `sample-one-line.docx` 文档。

现在我们可以开始了。

# 如何做…

1.  创建一个名为 `word.py` 的新 Python 文件，并添加以下 `import` 行：

```py
import docx
```

简单地导入 `python-docx` 模块中的 `docx` 对象。

1.  定义函数 `getTextWord`：

```py
def getTextWord(wordFileName):
```

该函数接受一个`string`类型的参数，`wordFileName`，它应该包含你想读取的 Word 文件的绝对路径。

1.  初始化`doc`对象：

```py
doc = docx.Document(wordFileName)
```

`doc`对象现在已加载你想要读取的 Word 文件。

1.  我们将从`doc`对象加载的文档中读取文本。为此，添加以下几行代码：

```py
fullText = []
for para in doc.paragraphs:
  fullText.append(para.text)
```

首先，我们初始化了一个字符串数组`fullText`。`for`循环按段落读取文档的文本，并将其附加到列表`fullText`中。

1.  现在我们将把所有片段/段落合并成一个字符串对象，并将其作为函数的最终输出返回：

```py
return '\n'.join(fullText)
```

我们使用分隔符`\n`将`fullText`数组中的所有元素连接起来，并返回结果对象。保存文件并退出。

1.  创建另一个文件，命名为`TestDocX.py`，并添加以下导入语句：

```py
import docx
import word
```

只需导入`docx`库和我们在前五个步骤中编写的`word.py`文件。

1.  现在我们将读取一个 DOCX 文档，并使用我们在`word.py`中编写的 API 打印出完整内容。写下以下两行：

```py
docName = 'sample-one-line.docx'
print('Document in full :\n',word.getTextWord(docName))
```

在第一行初始化文档路径，然后，使用 API 打印出完整文档。当你运行这部分时，应该会得到类似以下的输出：

```py
Document in full :
```

这是一个示例 PDF 文档，包含一些加粗文本、斜体文本和一些下划线文本。我们还嵌入了一个如下所示的标题：

```py
This is my TITLE.
This is my third paragraph.
```

1.  如前所述，Word/DOCX 文档是信息来源更加丰富的格式，相关库将提供比文本更多的信息。现在让我们看看段落信息。添加以下四行代码：

```py
doc = docx.Document(docName)
print('Number of paragraphs :',len(doc.paragraphs))
print('Paragraph 2:',doc.paragraphs[1].text)
print('Paragraph 2 style:',doc.paragraphs[1].style)
```

上述代码片段中的第二行给出了文档中段落的数量。第三行仅返回文档中的第二个段落，第四行将分析第二个段落的样式，在本例中是`Title`。当你运行时，这四行的输出将是：

```py
Number of paragraphs : 3
Paragraph 2: This is my TITLE.
Paragraph 2 style: _ParagraphStyle('Title') id: 4374023248
```

这很容易理解。

1.  接下来，我们将了解什么是 run。添加以下几行代码：

```py
print('Paragraph 1:',doc.paragraphs[0].text)
print('Number of runs in paragraph 1:',len(doc.paragraphs[0].runs))
for idx, run in enumerate(doc.paragraphs[0].runs):
  print('Run %s : %s' %(idx,run.text))
```

在这里，我们首先返回第一个段落；接着返回段落中的 run 数量。然后我们打印出每个 run。

1.  现在，为了识别每个 run 的样式，编写以下几行代码：

```py
print('is Run 0 underlined:',doc.paragraphs[0].runs[5].underline)
print('is Run 2 bold:',doc.paragraphs[0].runs[1].bold)
print('is Run 7 italic:',doc.paragraphs[0].runs[3].italic)
```

上述代码片段中的每一行分别检查下划线、加粗和斜体样式。在接下来的部分中，我们将看到最终的输出：

```py
Output: Document in full :
This is a sample PDF document with some text in BOLD, some in ITALIC and some underlined. We are also embedding a Title down below.
This is my TITLE.
This is my third paragraph.
Number of paragraphs : 3
Paragraph 2: This is my TITLE.
Paragraph 2 style: _ParagraphStyle('Title') id: 4374023248
Paragraph 1: This is a sample PDF document with some text in BOLD, some in ITALIC and some underlined. We're also embedding a Title down below.
Number of runs in paragraph 1: 8
Run 0 : This is a sample PDF document with
Run 1 : some text in BOLD
Run 2 : ,
Run 3 : some in ITALIC
Run 4 :  and
Run 5 : some underlined.
Run 6 :  We are also embedding a Title down below
Run 7 : .
is Run 0 underlined: True
is Run 2 bold: True
is Run 7 italic: True
```

# 它是如何工作的…

首先，我们在`word.py`文件中编写了一个函数，该函数将读取任何给定的 DOCX 文件，并将其完整内容返回为一个`string`对象。你看到的输出文本已经相当自解释了，但我想详细说明的是`Paragraph`和`Run`行。`.docx`文档的结构由`python-docx`库中的三种数据类型表示。最高级别是`Document`对象。每个文档内部都有多个段落。

每当我们看到新的一行或换行符时，它表示一个新段落的开始。每个段落包含多个`Run`，表示单词样式的变化。这里的样式指的是字体、大小、颜色以及其他样式元素，如粗体、斜体、下划线等。每当这些元素发生变化时，便开始一个新的运行。

# 从 PDF、DOCX 和纯文本文件中创建一个用户定义的语料库

对于这个例子，我们不会使用任何新的库或概念。我们将重新调用第一章中的语料库概念。只不过这次我们将创建自己的语料库，而不是使用从互联网上获得的语料库。

# 准备工作

在准备工作方面，我们将使用本章第一部分介绍的 Dropbox 文件夹中的一些文件。如果你已经下载了该文件夹中的所有文件，那么就没问题了。如果没有，请从[`www.dropbox.com/sh/bk18dizhsu1p534/AABEuJw4TArUbzJf4Aa8gp5Wa?dl=0`](https://www.dropbox.com/sh/bk18dizhsu1p534/AABEuJw4TArUbzJf4Aa8gp5Wa?dl=0)下载以下文件：

+   `sample_feed.txt`

+   `sample-pdf.pdf`

+   `sample-one-line.docx`

如果你没有按照本章的顺序进行操作，你需要回到本章的前两部分查看。我们将重用之前编写的两个模块`word.py`和`pdf.py`。本例更多的是应用我们在前两部分中做过的工作，以及来自第一章的语料库，而不是引入新概念。让我们继续实际的代码部分。

# 如何实现…

1.  创建一个新的 Python 文件，命名为`createCorpus.py`，并添加以下导入行开始：

```py
import os
import word, pdf
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
```

我们已经导入了`os`库来进行文件操作，以及我们在本章前两部分编写的`word`和`pdf`模块，还有`PlaintextCorpusReader`，这是本例的最终目标。

1.  现在让我们编写一个小函数，它将接受一个纯文本文件的路径作为输入，并将文件的完整文本返回为一个`string`对象。请添加以下几行：

```py
def getText(txtFileName):
  file = open(txtFileName, 'r')
  return file.read()
```

第一行定义了函数和输入参数。第二行以读取模式打开给定的文件（`open`函数的第二个参数`r`表示读取模式）。第三行读取文件的内容，并一次性将其返回为`string`对象。

1.  我们现在将在磁盘/文件系统上创建新的`corpus`文件夹。请添加以下三行：

```py
newCorpusDir = 'mycorpus/'
if not os.path.isdir(newCorpusDir):
  os.mkdir(newCorpusDir)
```

第一行是一个简单的`string`对象，包含新文件夹的名称。第二行检查磁盘上是否已存在同名的目录/文件夹。第三行指示`os.mkdir()`函数在磁盘上创建具有指定名称的目录。结果，在你的 Python 文件所在的工作目录中将创建一个名为`mycorpus`的新目录。

1.  现在我们将逐个读取这三个文件。首先是纯文本文件，添加以下行：

```py
txt1 = getText('sample_feed.txt')
```

调用之前编写的`getText()`函数，它将读取`sample_feed.txt`文件，并将输出返回到`txt1`字符串对象。

1.  现在我们将读取 PDF 文件。添加以下行：

```py
txt2 = pdf.getTextPDF('sample-pdf.pdf')
```

使用`pdf.py`模块的`getTextPDF()`函数，我们正在将`sample-pdf.pdf`文件的内容检索到`txt2`字符串对象中。

1.  最后，我们将通过添加以下行来读取 DOCX 文件：

```py
txt3 = word.getTextWord('sample-one-line.docx')
```

使用`word.py`模块的`getTextWord()`函数，我们正在将`sample-one-line.docx`文件的内容检索到`txt3`字符串对象中。

1.  下一步是将这三个字符串对象的内容写入磁盘文件。为此，请编写以下代码行：

```py
files = [txt1,txt2,txt3]
for idx, f in enumerate(files):
  with open(newCorpusDir+str(idx)+'.txt', 'w') as fout:
    fout.write(f)
```

+   **第一行**：从字符串对象创建一个数组，以便在接下来的`for`循环中使用

+   **第二行**：带有索引的`for`循环，遍历文件数组

+   **第三行**：这将以写模式（在`open`函数调用中的`w`选项）打开一个新文件

+   **第四行**：将字符串对象的内容写入文件

1.  现在我们将从`mycorpus`目录创建一个`PlainTextCorpus`对象，我们将文件存储在该目录中：

```py
newCorpus = PlaintextCorpusReader(newCorpusDir, '.*')
```

这是一个简单的单行指令，但在内部它执行了大量的文本处理，识别段落、句子、单词等等。两个参数是语料库目录的路径和要考虑的文件名模式（在这里我们要求语料库读取器考虑目录中的所有文件）。我们创建了一个用户定义的语料库。就这么简单！

1.  让我们看看我们的`PlainTextCorpusReader`是否正确加载。添加以下代码行进行测试：

```py
print(newCorpus.words())
print(newCorpus.sents(newCorpus.fileids()[1]))
print(newCorpus.paras(newCorpus.fileids()[0]))
```

第一行将打印包含语料库中所有单词的数组（已截断）。第二行将打印文件`1.txt`中的句子。第三行将打印文件`0.txt`中的段落：

```py
Output: ['Five', 'months', '.', 'That', "'", 's', 'how', ...]
[['A', 'generic', 'NLP'], ['(', 'Natural', 'Language', 'Processing', ')', 'toolset'], ...]
[[['Five', 'months', '.']], [['That', "'", 's', 'how', 'long', 'it', "'", 's', 'been', 'since', 'Mass', 'Effect', ':', 'Andromeda', 'launched', ',', 'and', 'that', "'", 's', 'how', 'long', 'it', 'took', 'BioWare', 'Montreal', 'to', 'admit', 'that', 'nothing', 'more', 'can', 'be', 'done', 'with', 'the', 'ailing', 'game', "'", 's', 'story', 'mode', '.'], ['Technically', ',', 'it', 'wasn', "'", 't', 'even', 'a', 'full', 'five', 'months', ',', 'as', 'Andromeda', 'launched', 'on', 'March', '21', '.']], ...]
```

# 它是如何工作的…

输出相当直接，并且如食谱的最后一步所解释的那样。特别的是，每个展示对象的特点。第一行是新语料库中所有单词的列表；它与句子/段落/文件等高级结构无关。第二行是文件`1.txt`中所有句子的列表，每个句子是该句子中单词的列表。第三行是段落的列表，每个段落对象又是一个句子的列表，而每个句子又是该句子中单词的列表，所有这些都来自文件`0.txt`。如你所见，在段落和句子中保持了大量的结构信息。

# 从 RSS 源中读取内容

**富网站摘要**（**RSS**）源是一种计算机可读格式，用于传输互联网上定期更新的内容。大多数提供这种格式信息的网站会提供更新内容，例如新闻文章、在线出版物等等。它以标准化格式让听众可以定期访问更新的源。

# 准备就绪

本教程的目标是读取这样的 RSS 源，并访问该源中某篇帖子的内容。为此，我们将使用 Mashable 的 RSS 源。Mashable 是一个数字媒体网站，简而言之，它是一个科技和社交媒体博客列表。该网站的 RSS 源地址是[`feeds.mashable.com/Mashable`](http://feeds.mashable.com/Mashable)。

此外，我们需要`feedparser`库才能读取 RSS 源。要在计算机上安装此库，只需打开终端并运行以下命令：

```py
pip install feedparser
```

拥有这个模块和有用的信息后，我们可以开始编写我们的第一个 Python RSS 源阅读器。

# 如何操作…

1.  创建一个名为`rssReader.py`的新文件，并添加以下导入：

```py
import feedparser
```

1.  现在，我们将 Mashable 的源加载到内存中。添加以下行：

```py
myFeed = feedparser.parse("http://feeds.mashable.com/Mashable")
```

`myFeed`对象包含 Mashable 的 RSS 源的第一页。该源将被下载并通过`feedparser`解析，以填充所有适当的字段。每篇帖子都将作为`myFeed`对象中`entries`列表的一部分。

1.  让我们检查标题并计算当前源中的帖子数：

```py
print('Feed Title :', myFeed['feed']['title'])
print('Number of posts :', len(myFeed.entries))
```

在第一行，我们从`myFeed`对象中获取源的标题，在第二行，我们计算`myFeed`对象中`entries`对象的长度。`entries`对象其实就是从解析的源中获取的所有帖子列表。运行时，输出类似于：

```py
Feed Title: Mashable
Number of posts : 30
```

`Title`将始终是 Mashable，并且在写这章时，Mashable 团队在源中最多放入 30 篇帖子。

1.  现在，我们将从`entries`列表中获取第一个`post`并将其标题打印到控制台：

```py
post = myFeed.entries[0]
print('Post Title :',post.title)
```

在第一行，我们物理地访问`entries`列表中的第一个元素，并将其加载到`post`对象中。第二行打印该帖子的标题。运行后，你应该得到类似以下的输出：

```py
Post Title: The moon literally blocked the sun on Twitter
```

我说的是类似的内容，而不是完全相同，因为源会不断更新。

1.  现在我们将访问帖子的原始 HTML 内容，并将其打印到控制台：

```py
content = post.content[0].value
print('Raw content :\n',content)
```

首先，我们访问帖子的内容对象及其实际值。然后，我们将其打印到控制台：

```py
Output: Feed Title: Mashable
Number of posts : 30
Post Title: The moon literally blocked the sun on Twitter
Raw content :
<img alt="" src="img/https%3A%2F%2Fblueprint-api-production.s3.amazonaws.com%2Fuploads%2Fcard%2Fimage%2F569570%2F0ca3e1bf-a4a2-4af4-85f0-1bbc8587014a.jpg" /><div style="float: right; width: 50px;"><a href="http://twitter.com/share?via=Mashable&text=The+moon+literally+blocked+the+sun+on+Twitter&url=http%3A%2F%2Fmashable.com%2F2017%2F08%2F21%2Fmoon-blocks-sun-eclipse-2017-twitter%2F%3Futm_campaign%3DMash-Prod-RSS-Feedburner-All-Partial%26utm_cid%3DMash-Prod-RSS-Feedburner-All-Partial" style="margin: 10px;">
<p>The national space agency threw shade the best way it knows how: by blocking the sun. Yep, you read that right. </p>
<div><div><blockquote>
<p>HA HA HA I've blocked the Sun! Make way for the Moon<a href="https://twitter.com/hashtag/SolarEclipse2017?src=hash">#SolarEclipse2017</a> <a href="https://t.co/nZCoqBlSTe">pic.twitter.com/nZCoqBlSTe</a></p>
<p>— NASA Moon (@NASAMoon) <a href="https://twitter.com/NASAMoon/status/899681358737539073">August 21, 2017</a></p>
</blockquote></div></div>
```

# 它是如何工作的…

大多数你在互联网上获取的 RSS 订阅源会遵循时间顺序排列，最新的帖子位于最上面。因此，在本食谱中访问的帖子将始终是订阅源提供的最新帖子。订阅源本身是不断变化的。所以，每次运行程序时，输出的格式将保持不变，但控制台上的帖子内容可能会有所不同，具体取决于订阅源更新的速度。此外，这里我们直接在控制台上显示的是原始 HTML，而不是清理后的内容。接下来，我们将查看如何解析 HTML 并仅从页面中提取我们需要的信息。再进一步，本食谱的附加内容可以是读取你选择的任何订阅源，将所有帖子存储到磁盘，并使用它创建一个纯文本语料库。不用说，你可以从前一个和下一个食谱中获取灵感。

# 使用 BeautifulSoup 进行 HTML 解析

大多数情况下，当你需要处理网页上的数据时，它会以 HTML 页面的形式存在。为此，我们认为有必要向你介绍 Python 中的 HTML 解析。虽然有许多 Python 模块可以完成这个任务，但在本食谱中，我们将演示如何使用`BeautifulSoup4`库进行 HTML 解析。

# 准备中

`BeautifulSoup4`包支持 Python 2 和 Python 3。我们需要在解释器中下载并安装这个包，才能开始使用它。和我们一直以来的做法一样，我们将使用 pip 安装工具来进行安装。运行以下命令：

```py
pip install beautifulsoup4
```

除了这个模块，你还需要从本章的 Dropbox 位置获取`sample-html.html`文件。如果你还没有下载这些文件，这里再给你链接：

[`www.dropbox.com/sh/bk18dizhsu1p534/AABEuJw4TArUbzJf4Aa8gp5Wa?dl=0`](https://www.dropbox.com/sh/bk18dizhsu1p534/AABEuJw4TArUbzJf4Aa8gp5Wa?dl=0)

# 如何操作…

1.  假设你已经安装了所需的包，开始时使用以下导入语句：

```py
from bs4 import BeautifulSoup
```

我们已经从`bs4`模块中导入了`BeautifulSoup`类，接下来我们将使用它来解析 HTML。

1.  让我们将 HTML 文件加载到`BeautifulSoup`对象中：

```py
html_doc = open('sample-html.html', 'r').read()
soup = BeautifulSoup(html_doc, 'html.parser')
```

在第一行，我们将`sample-html.html`文件的内容加载到`str`对象`html_doc`中。接着我们创建一个`BeautifulSoup`对象，将 HTML 文件的内容作为第一个参数传入，将`html.parser`作为第二个参数传入。我们指示它使用`html`解析器解析文档。这样，文档就会被加载到`soup`对象中，已解析并准备好使用。

1.  在这个` soup`对象上，第一个最简单且最有用的任务就是去除所有 HTML 标签，获取文本内容。添加以下代码行：

```py
print('\n\nFull text HTML Stripped:')
print(soup.get_text())
```

在`soup`对象上调用`get_text()`方法将提取文件中去除 HTML 标签的内容。如果你运行到目前为止编写的代码，你将得到以下输出：

```py
Full text HTML Stripped:
Sample Web Page

Main heading
This is a very simple HTML document
Improve your image by including an image.
Add a link to your favorite Web site.
This is a new sentence without a paragraph break, in bold italics.
This is purely the contents of our sample HTML document without any of the HTML tags.
```

1.  有时候，仅仅拥有纯粹的 HTML 去除内容是不够的。你可能还需要特定标签的内容。让我们来访问其中一个标签：

```py
print('Accessing the <title> tag :', end=' ')
print(soup.title)
```

`soup.title` 将返回文件中遇到的第一个 title 标签。以下行的输出将是：

```py
Accessing the <title> tag : <title>Sample Web Page</title>
```

1.  现在让我们只从标签中获取去除 HTML 的文本。我们将使用以下代码抓取 `<h1>` 标签的文本：

```py
print('Accessing the text of <H1> tag :', end=' ')
print(soup.h1.string)
```

命令 `soup.h1.string` 将返回被第一个 `<h1>` 标签包围的文本。该行的输出将是：

```py
Accessing the text of <H1> tag : Main heading
```

1.  现在我们将访问标签的属性。在这个例子中，我们将访问 `img` 标签的 `alt` 属性；添加以下代码行：

```py
print('Accessing property of <img> tag :', end=' ')
print(soup.img['alt'])
```

仔细观察；访问标签属性的语法与访问文本是不同的。当你运行这段代码时，你会得到以下输出：

```py
Accessing property of <img> tag : A Great HTML Resource
```

1.  最后，在 HTML 文件中可能会有多个相同类型的标签。仅使用 `.` 语法将只会获取到第一个实例。要获取所有实例，我们使用 `find_all()` 功能，如下所示：

```py
print('\nAccessing all occurences of the <p> tag :')
for p in soup.find_all('p'):
  print(p.string)
```

对 `BeautifulSoup` 对象调用的 `find_all()` 函数将接受标签名作为参数，遍历整个 HTML 树，并返回该标签的所有实例作为一个列表。我们在 `for` 循环中访问该列表并打印给定 `BeautifulSoup` 对象中所有 `<p>` 标签的内容/文本：

```py
Output: Full text HTML Stripped:

Sample Web Page

Main heading
This is a very simple HTML document
Improve your image by including an image.

Add a link to your favorite Web site.
 This is a new sentence without a paragraph break, in bold italics.

Accessing the <title> tag : <title>Sample Web Page</title>
Accessing the text of <H1> tag : Main heading
Accessing property of <img> tag : A Great HTML Resource

Accessing all occurences of the <p> tag :
This is a very simple HTML document
Improve your image by including an image.
None
```

# 它是如何工作的……

BeautifulSoup 4 是一个非常方便的库，用于解析任何 HTML 和 XML 内容。它支持 Python 内建的 HTML 解析器，但你也可以使用其他第三方解析器，比如 `lxml` 解析器和纯 Python `html5lib` 解析器。在这个示例中，我们使用了 Python 内建的 HTML 解析器。生成的输出几乎是自解释的，当然，前提是你知道 HTML 是什么以及如何编写简单的 HTML。
