# 第四章：正则表达式

在本章中，我们将涵盖以下实例：

+   正则表达式——学习使用 `*`、`+` 和 `?`

+   正则表达式——学习使用 `$` 和 `^`，以及单词的非开始和非结束匹配

+   搜索多个字面字符串和子字符串出现位置

+   学习创建日期正则表达式和字符集或字符范围

+   查找所有五个字符的单词，并在某些句子中创建缩写

+   学习编写你自己的正则表达式分词器

+   学习编写你自己的正则表达式词干提取器

# 介绍

在上一章中，我们看到了你在处理原始数据时可能需要执行的预处理任务。本章紧接着提供了一个很好的机会来介绍正则表达式。正则表达式是最简单和基础的工具之一，但也是最重要和强大的工具之一，它们用于在文本中匹配模式。在本章中，我们将学习它到底有多强大。

我们并不声称在本章结束后你会成为正则表达式编写的专家，这也许不是本书或本章的目标。本章的目的是介绍模式匹配的概念，作为文本分析的一种方式，而要开始学习，正则表达式是最好的工具。通过完成这些实例，你应该会对执行任何文本匹配、文本分割、文本搜索或文本提取操作充满信心。

让我们详细看一下前面提到的实例。

# 正则表达式——学习使用 `*`、`+` 和 `?`

我们从一个例子开始，详细说明如何在正则表达式中使用 `+` 和 `?` 操作符。这些简写操作符通常被称为通配符，但我更喜欢将它们称为零个或多个（`*`）、一个或多个（`+`）、以及零个或一个（`?`），以便区分。如果你从这个角度思考，它们的名称会更加直观。

# 准备工作

正则表达式库是 Python 包的一部分，无需安装额外的包。

# 如何做到……

1.  创建一个名为 `regex1.py` 的文件，并在其中添加以下 `import` 语句：

```py
import re
```

这行代码导入了 `re` 模块，它允许我们处理和应用正则表达式。

1.  在要应用给定模式进行匹配的文件中添加以下 Python 函数：

```py
def text_match(text, patterns):
```

这个函数接受两个参数；`text` 是输入的文本，`patterns` 是将在其上应用的匹配模式。

1.  现在，让我们定义函数。在函数下方添加以下几行代码：

```py
if re.search(patterns,  text):
  return 'Found a match!'
else:
  return('Not matched!')
```

`re.search()` 方法将给定模式应用于 `text` 对象，并根据应用结果返回真或假。到此为止，函数结束。

1.  让我们逐个应用这些通配符模式。我们从零个或一个开始：

```py
    print(text_match("ac", "ab?"))
    print(text_match("abc", "ab?"))
    print(text_match("abbc", "ab?"))
```

1.  我们来看这个模式 `ab?`。它的意思是一个 `a` 后面跟零个或一个 `b`。我们来看一下当我们执行这三行代码时输出会是什么：

```py
Found a match!
Found a match!
Found a match!
```

现在，所有的模式都找到了匹配。这些模式尝试匹配输入的一部分，而不是整个输入；因此，它们在所有三个输入中都找到了匹配。

1.  接下来，来看零个或多个！添加以下三行：

```py
print(text_match("ac", "ab*"))
print(text_match("abc", "ab*"))
print(text_match("abbc", "ab*"))
```

1.  相同的输入集，但不同的字符串。模式表示，`a` 后面跟着零个或多个 `b`。让我们看看这三行的输出：

```py
Found a match!
Found a match!
Found a match!
```

如你所见，所有文本都找到了匹配项。通常来说，任何匹配零个或一个通配符的也会匹配零个或多个。`?` 通配符是 `*` 的一个子集。

1.  现在，来看一下一个或多个的通配符。添加以下几行：

```py
print(text_match("ac", "ab+"))
print(text_match("abc", "ab+"))
print(text_match("abbc", "ab+"))
```

1.  相同的输入！只是模式中包含了 `+`（一个或多个通配符）。让我们看看输出：

```py
Not matched!
Found a match!
Found a match!
```

如你所见，第一个输入字符串没有找到匹配项。其余的都按预期找到了匹配。

1.  现在，更具体地指定重复次数，添加以下这一行：

```py
print(text_match("abbc", "ab{2}"))
```

模式表示 `a` 后面跟着正好两个 `b`。不用说，这个模式将在输入文本中找到匹配项。

1.  是时候来进行一系列重复操作了！添加以下这一行：

```py
print(text_match("aabbbbc", "ab{3,5}?"))
```

这也会匹配，因为我们有一个子串 `a` 后跟四个 `b`。

程序的输出完全没有什么意义。我们已经 `ana.ysed` 了每一个步骤的输出；因此，我们在这里不会再次打印出来。

# 它是如何工作的…

`re.search()` 函数是一个只会将给定模式作为测试应用的函数，并会返回测试结果的真或假。它不会返回匹配的值。对于这个，还有其他 `re` 函数，我们将在后续的食谱中学习。

# 正则表达式 – 学习如何使用 $ 和 ^，以及单词的非开始和非结束

以 (^) 开头并以 ($) 结尾的操作符是用于匹配输入文本开头或结尾的指示符。

# 准备工作

我们本可以重用之前食谱中的 `text_match()` 函数，但我们不打算导入外部文件，而是重新编写它。让我们看看食谱的实现。

# 如何实现…

1.  创建一个名为 `regex2.py` 的文件，并添加以下 `import` 语句：

```py
import re
```

1.  将这个 Python 函数添加到应该应用给定模式进行匹配的文件中：

```py
def text_match(text, patterns):
  if re.search(patterns,  text):
    return 'Found a match!'
  else:
    return('Not matched!')
```

这个函数接受两个参数；`text` 是输入文本，`patterns` 将应用于该文本以进行匹配，并返回是否找到匹配项。这个函数正是我们在之前食谱中编写的。

1.  让我们应用以下模式。我们从一个简单的以“开始于”和“结束于”开始：

```py
print("Pattern to test starts and ends with")
print(text_match("abbc", "^a.*c$"))
```

1.  让我们看看这个模式，`^a.*c$`。这意味着：以 `a` 开头，后跟零个或多个任意字符，并以 `c` 结尾。让我们看看执行这三行时的输出：

```py
Pattern to test starts and ends with

Found a match!
```

它当然找到了输入文本的匹配项。我们在这里介绍了一个新的 `.` 通配符。默认模式下，点号匹配除了换行符以外的任何字符；也就是说，当你说 `.*` 时，它意味着零个或多个任意字符。

1.  接下来，寻找一个模式，检查输入文本是否以一个单词开始。添加以下两行代码：

```py
print("Begin with a word")
print(text_match("Tuffy eats pie, Loki eats peas!", "^\w+"))
```

1.  `\w`表示任何字母数字字符和下划线。模式表示：以（`^`）任何字母数字字符（`\w`）开始，且它出现一次或多次（`+`）。输出结果：

```py
Begin with a word

Found a match!
```

正如预期的那样，模式找到了匹配。

1.  接下来，我们检查以某个单词和可选标点符号结尾的情况。添加以下几行代码：

```py
print("End with a word and optional punctuation")
print(text_match("Tuffy eats pie, Loki eats peas!", "\w+\S*?$"))
```

1.  该模式表示一个或多个`\w`字符，后面跟着零个或多个`\S`字符，且应该接近输入文本的末尾。为了理解`\S`（大写`S`），我们必须先了解`\s`，它代表所有的空白字符。`\S`是`\s`的反集，当它后面跟着`\w`时表示查找一个标点符号：

```py
End with a word and optional punctuation

Found a match!
```

我们在输入文本的末尾找到了与豌豆匹配的内容！

1.  接下来，找一个包含特定字符的单词。添加以下几行代码：

```py
print("Finding a word which contains character, not start or end of the word")
print(text_match("Tuffy eats pie, Loki eats peas!", "\Bu\B"))
```

对于解码这个模式，`\B`是`\b`的反集或反向模式。`\b`匹配一个空字符串，位于单词的开始或结束，我们已经知道什么是单词。因此，`\B`会匹配单词内部，它将匹配输入字符串中包含字符`u`的任何单词：

```py
Finding a word which contains character, not start or end of the word

Found a match!
```

我们在第一个单词`Tuffy`中找到了匹配。

这是程序的完整输出。我们已经详细查看过，所以我不再重复讲解：

```py
Pattern to test starts and ends with

Found a match!

Begin with a word

Found a match!

End with a word and optional punctuation

Found a match!

Finding a word which contains character, not start or end of the word

Found a match!
```

# 它是如何工作的……

除了“开始”和“结束”模式，我们还学习了通配符字符`.`和一些其他特殊序列，比如`\w`、`\s`、`\b`等。

# 搜索多个字面量字符串和子字符串出现位置

在这个例子中，我们将运行一些带有正则表达式的迭代函数。更具体地说，我们将在输入字符串上运行多个模式，使用`for`循环，还将对单个模式进行多次匹配。让我们直接看看如何做。

# 准备好

打开你的 PyCharm 编辑器或任何你使用的 Python 编辑器，准备好开始吧。

# 如何做到这一点……

1.  创建一个名为`regex3.py`的文件，并在其中添加以下`import`语句：

```py
import re
```

1.  添加以下两行 Python 代码来声明和定义我们的模式和输入文本：

```py
patterns = [ 'Tuffy', 'Pie', 'Loki' ]
text = 'Tuffy eats pie, Loki eats peas!'
```

1.  让我们编写第一个`for`循环。添加以下几行代码：

```py
for pattern in patterns:
  print('Searching for "%s" in "%s" -&gt;' % (pattern, text),)
  if re.search(pattern,  text):
    print('Found!')
  else:
    print('Not Found!')
```

这是一个简单的`for`循环，逐个遍历模式列表，并调用`re`的搜索函数。运行这段代码，你会发现输入字符串中的三个单词中有两个匹配成功。另外请注意，这些模式是区分大小写的；大写单词`Tuffy`！我们将在输出部分讨论结果。

1.  接下来，搜索一个子字符串并找到其位置。首先定义模式和输入文本：

```py
text = 'Diwali is a festival of lights, Holi is a festival of colors!'
pattern = 'festival'
```

前两行分别定义了输入文本和要搜索的模式。

1.  现在，我们的`for`循环将遍历输入文本并获取给定模式的所有出现位置：

```py
for match in re.finditer(pattern, text):
  s = match.start()
  e = match.end()
  print('Found "%s" at %d:%d' % (text[s:e], s, e))
```

1.  `finditer` 函数接收模式和输入文本作为参数，用来匹配该模式。在返回的列表中，我们将进行迭代。对于每个对象，我们将调用 `start` 和 `end` 方法，确定我们在哪个位置找到了匹配的模式。我们将在这里讨论这个代码块的输出。这个小代码块的输出将如下所示：

```py
Found "festival" at 12:20

Found "festival" at 42:50
```

两行输出！这表明我们在输入的两个地方找到了匹配的模式。第一次是在位置 `12:20`，第二次是在 `42:50`，如输出文本行所示。

这是程序的完整输出。我们已经详细查看过其中的一些部分，但我们将再次逐步查看：

```py
Searching for "Tuffy" in "Tuffy eats pie, Loki eats peas!" -&gt;

Found!

Searching for "Pie" in "Tuffy eats pie, Loki eats peas!" -&gt;

Not Found!

Searching for "Loki" in "Tuffy eats pie, Loki eats peas!" -&gt;

Found!

Found "festival" at 12:20

Found "festival" at 42:50
```

输出非常直观，至少前六行是这样的。我们搜索了单词 `Tuffy` 并找到了它。单词 `Pie` 没有找到（`re.search()` 函数是区分大小写的），然后找到了单词 `Loki`。最后两行我们在第六步中已经讨论过。我们不仅搜索了字符串，还指出了在给定输入中找到它们的索引。

# 它是如何工作的…

让我们再讨论一下我们到目前为止使用得非常频繁的 `re.search()` 函数。如你所见，在前面的输出中，单词 `pie` 是输入文本的一部分，但我们搜索的是大写单词 `Pie`，并且似乎找不到它。如果你在搜索函数调用中添加一个标志 `re.IGNORECASE`，那么它才会进行不区分大小写的搜索。语法将是 `re.search(pattern, string, flags=re.IGNORECASE)`。

现在，我们来看 `re.finditer()` 函数。函数的语法是 `re.finditer(pattern, string, flags=0)`。它返回一个迭代器，包含输入字符串中所有不重叠匹配的 `MatchObject` 实例。

# 学习创建日期正则表达式和字符集或字符范围

在这个教程中，我们将首先运行一个简单的日期正则表达式。与此同时，我们将学习 () 分组的意义。由于这对于一个教程来说内容太少，我们还将介绍一些其他内容，如方括号 []，它表示一个集合（我们将详细了解集合的定义）。

# 如何实现它…

1.  创建一个名为 `regex4.py` 的文件，并添加以下 `import` 行：

```py
import re
```

1.  让我们声明一个 `url` 对象，并编写一个简单的日期查找正则表达式来开始：

```py
url= "http://www.telegraph.co.uk/formula-1/2017/10/28/mexican-grand-prix-2017-time-does-start-tv-channel-odds-lewis1/"

date_regex = '/(\d{4})/(\d{1,2})/(\d{1,2})/'
```

`url` 是一个简单的字符串对象。`date_regex` 也是一个简单的字符串对象，但它包含一个正则表达式，该正则表达式可以匹配格式为 *YYYY/DD/MM* 或 *YYYY/MM/DD* 类型的日期。`\d` 表示从 0 到 9 的数字。我们已经学过了符号 {} 的用法。

1.  让我们将 `date_regex` 应用到 `url` 并查看输出。添加以下行：

```py
print("Date found in the URL :", re.findall(date_regex, url))
```

1.  一个新的 `re` 函数，`re.findall(pattern, input, flags=0)`，它同样接受模式、输入文本，并可选地接受标志（我们在之前的教程中学过区分大小写的标志）。让我们看看输出：

```py
Date found in the URL : [('2017', '10', '28')]
```

所以，我们在给定的输入字符串对象中找到了日期 2017 年 10 月 28 日。

1.  接下来是下一部分，我们将学习字符集符号`[]`。在代码中添加以下函数：

```py
def is_allowed_specific_char(string):
  charRe = re.compile(r'[^a-zA-Z0-9.]')
  string = charRe.search(string)
  return not bool(string)
```

这里的目的是检查输入字符串是否包含特定的一组字符或其他字符。在这里，我们采用了一种稍微不同的方法；首先，我们使用`re.compile`编译模式，这将返回一个`RegexObject`。然后，我们调用`RegexObject`的`search`方法来匹配已编译的模式。如果找到匹配项，`search`方法将返回一个`MatchObject`，否则返回`None`。现在，我们将注意力转向符号集`[]`。方括号内的模式意味着：不（`^`）是字符范围`a-z`、`A-Z`、`0-9`或`.`。实际上，这是方括号内所有内容的或操作。

1.  现在测试模式。我们在两种不同类型的输入上调用这个函数，一种是匹配的，一种是没有匹配的：

```py
print(is_allowed_specific_char("ABCDEFabcdef123450."))
print(is_allowed_specific_char("*&%@#!}{"))
```

1.  第一个字符集包含了所有允许的字符，而第二个字符集包含了所有不允许的字符。正如预期的那样，这两行的输出将是：

```py
True

False
```

模式将遍历输入字符串中的每一个字符，检查是否存在任何不允许的字符，如果找到，它将标记出来。你可以尝试在第一次调用`is_allwoed_specific_char()`时添加任何不允许的字符集，并自行检查结果。

这是程序的完整输出。我们已经详细查看过，因此不再重复。

```py
Date found in the URL : [('2017', '10', '28')]

True

False
```

# 它是如何工作的……

首先让我们讨论一下什么是“分组”。在任何正则表达式中，分组是指在模式声明中的括号`()`内所包含的内容。如果你查看日期匹配的输出，你会看到一个集合表示法，里面包含了三个字符串对象：`[('2017', '10', '28')]`。现在，仔细看一下声明的模式`/(\d{4})/(\d{1,2})/(\d{1,2})/`。日期的三个组件都被标记在分组表示法`()`内，因此这三个部分被单独识别出来。

现在，`re.findall()`方法将在给定的输入中查找所有匹配项。这意味着，如果输入文本中有更多日期，输出将是类似于`[('2017', '10', '28'), ('2015', '05', '12')]`的形式。

`[]`符号集本质上意味着：匹配集合表示法中所包含的任意字符。如果找到任何单一匹配项，模式就成立。

# 在一些句子中查找所有五个字符的单词并进行缩写。

我们已经通过之前的示例覆盖了所有我想要讲解的重要符号。接下来，我们将讨论一些较小的示例，这些示例更侧重于使用正则表达式完成特定任务，而不是解释符号。话虽如此，我们仍然会学习一些其他的符号。

# 如何做……

1.  创建一个名为`regex_assignment1.py`的文件，并在其中添加以下`import`行：

```py
import re
```

1.  添加以下两行 Python 代码来定义输入字符串，并应用缩写的替代模式：

```py
street = '21 Ramkrishna Road'
print(re.sub('Road', 'Rd', street))
```

1.  首先，我们要做的是缩写处理，为此我们使用 `re.sub()` 方法。要查找的模式是 `Road`，替换成的字符串是 `Rd`，输入字符串是 `street`。让我们看看输出：

```py
21 Ramkrishna Rd
```

显然，它按预期工作。

1.  现在，让我们在任何给定的句子中找到所有五个字符的单词。为此，添加这两行代码：

```py
text = 'Diwali is a festival of light, Holi is a festival of color!'
print(re.findall(r"\b\w{5}\b", text))
```

1.  声明一个字符串对象 `text`，并将句子放入其中。接下来，创建一个模式并使用 `re.findall()` 函数应用它。我们使用 `\b` 边界集来识别单词之间的边界，并使用 `{}` 符号来确保我们仅列出五个字符的单词。运行此程序，你将看到按预期匹配的单词列表：

```py
['light', 'color']
```

这是程序的完整输出。我们已经详细查看过它，所以我们不再重复：

```py
21 Ramkrishna Rd

['light', 'color']
```

# 它是如何工作的...

到现在为止，我相信你已经对正则表达式的符号和语法有了很好的理解。因此，在我们编写本配方时给出的解释已经足够了。接下来，让我们看一些更有趣的东西。看看 `findall()` 方法，你会看到像 `r&lt;pattern&gt;` 的符号。这叫做原始字符串符号；它有助于保持正则表达式的可读性。如果你不使用它，你必须为正则表达式中的所有反斜杠提供转义序列。例如，`r"\b\w{5}\b"` 和 `"\\b\\w{5}\\b"` 的功能完全相同。

# 学习编写你自己的正则表达式分词器

我们已经从上一章了解了令牌、分词器的概念以及为什么需要它们。我们还看到了如何使用 NLTK 模块的内置分词器。在这个配方中，我们将编写我们自己的分词器；它将逐步发展，模拟 `nltk.word_tokenize()` 的行为。

# 准备中

如果你已经准备好 Python 解释器和编辑器，那么你就已经准备好了。

# 如何实现...

1.  创建一个名为 `regex_tokenizer.py` 的文件，并在其中添加以下 `import` 行：

```py
import re
```

1.  让我们定义要分词的原始句子和第一个模式：

```py
raw = "I am big! It's the pictures that got small."

print(re.split(r' +', raw))
```

1.  这个模式将执行与前一章中看到的空格分词器相同的功能。让我们看一下输出：

```py
['I', 'am', 'big!', "It's", 'the', 'pictures', 'that', 'got', 'small.']
```

正如我们所见，我们的小模式完全按预期工作。

1.  现在，这还不够，对吧？我们希望将令牌分割成任何非单词字符，而不仅仅是 `' '` 字符。让我们尝试以下模式：

```py
print(re.split(r'\W+', raw))
```

1.  我们正在对所有非单词字符进行分割，即 `\W`。让我们看看输出：

```py
['I', 'am', 'big', 'It', 's', 'the', 'pictures', 'that', 'got', 'small', '']
```

我们确实已经分割了所有非单词字符（`' '`、`,`、`!` 等），但似乎它们已经完全从结果中删除了。看起来我们需要做一些额外的事情，且有所不同。

1.  `split` 似乎没有完成任务；让我们尝试一个不同的 `re` 函数，`re.findall()`。添加以下行：

```py
print(re.findall(r'\w+|\S\w*', raw))
```

1.  让我们运行并查看输出：

```py
['I', 'am', 'big', '!', 'It', "'s", 'the', 'pictures', 'that', 'got', 'small', '.']
```

看起来我们中奖了。

这是程序的完整输出。我们已经讨论过它了；让我们打印出来：

```py
['I', 'am', 'big!', "It's", 'the', 'pictures', 'that', 'got', 'small.']

['I', 'am', 'big', 'It', 's', 'the', 'pictures', 'that', 'got', 'small', '']

['I', 'am', 'big', '!', 'It', "'s", 'the', 'pictures', 'that', 'got', 'small', '.']
```

如你所见，我们逐步改进了我们的模式和方法，最终达到了最佳的结果。

# 它是如何工作的…

我们从对空格字符的简单`re.split`开始，然后通过非字母字符进行改进。最后，我们改变了方法；我们不再尝试分割，而是通过使用`re.findall`来匹配我们想要的内容，这完成了任务。

# 学会编写自己的正则表达式词干提取器

我们已经在上一章了解了词干/词形、词干提取器及其必要性。我们已经看到了如何使用 NLTK 模块的内置 Porter 词干提取器和 Lancaster 词干提取器。在这个示例中，我们将编写自己的正则表达式词干提取器，去除不需要的后缀，以找到正确的词干。

# 准备工作

正如我们在之前的词干提取器和词形还原器示例中所做的那样，我们需要先对文本进行标记化处理，然后再应用词干提取器。这正是我们将要做的。我们将重新使用上一个示例中的最终标记化模式。如果你还没有查看之前的示例，请先查看一下，然后你就可以开始这个示例了。

# 如何操作…

1.  创建一个名为`regex_tokenizer.py`的文件，并在其中添加以下`import`语句：

```py
import re
```

1.  我们将编写一个函数来完成词干提取的工作。让我们首先在这一步声明函数的语法，接下来我们会在下一步定义它：

```py
def stem(word):
```

这个函数应该接受一个字符串对象作为参数，并且返回一个字符串对象作为结果。词根提取，输出！

1.  让我们定义`stem()`函数：

```py
splits = re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$', word)
stem = splits[0][0]
return stem
```

我们正在对输入的单词应用`re.findall()`函数，返回两个组作为输出。第一个是词根，第二个是可能的后缀。我们从函数调用中返回第一个组作为结果。

1.  让我们定义输入句子并进行标记化。添加以下几行：

```py
raw = "Keep your friends close, but your enemies closer."
tokens = re.findall(r'\w+|\S\w*', raw)
print(tokens)
```

1.  让我们运行并查看输出：

```py
['Keep', 'your', 'friends', 'close', ',', 'but', 'your', 'enemies', 'closer', '.']
```

看起来我们已经得到了可以进行词干提取的标记。

1.  让我们将`stem()`方法应用于我们刚刚生成的标记列表。添加以下`for`循环：

```py
for t in tokens:
  print("'"+stem(t)+"'")
```

我们只是遍历所有的标记，并逐一打印返回的词干。我们将在接下来的输出部分看到结果并进行讨论。

让我们查看整个代码的输出：

```py
['Keep', 'your', 'friends', 'close', ',', 'but', 'your', 'enemies', 'closer', '.']

'Keep'

'your'

'friend'

'close'

','

'but'

'your'

'enem'

'closer'

'.'
```

我们的词干提取器似乎做得相当不错。然而，我认为我给词干提取器提供的是一个看起来简单的句子。

# 它是如何工作的…

再次，我们使用`re.findall()`函数来获取所需的输出，尽管你可能需要仔细查看第一个组的正则表达式模式。我们使用了一个非贪婪的通配符匹配（`.*?`），否则它会贪婪地吞噬整个单词，导致无法识别后缀。此外，输入的开始和结束是必须的，以匹配整个输入单词并进行分割。
