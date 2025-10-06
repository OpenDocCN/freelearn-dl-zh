# 2<st c="0">

# 代码实验室 – 整个 RAG 管道<st c="2">

本代码实验室为本书中其余代码奠定了基础。<st c="35"> <st c="109">我们将花费整个章节的时间，为您提供整个</st> **<st c="164">检索增强生成</st>** <st c="194">(</st>**<st c="196">RAG</st>**<st c="199">**)管道。<st c="212">然后，随着我们逐步阅读本书，我们将查看代码的不同部分，并在过程中添加增强功能，以便您全面了解代码如何演变以解决更多和更复杂的难题。</st> <st c="425">

我们将花费这一章的时间，逐一介绍 RAG 管道的每个组件，包括以下方面：<st c="444"> <st c="538">

+   无界面<st c="556">

+   在 OpenAI 上设置一个大型语言模型（LLM）账户<st c="569"> <st c="618">

+   安装所需的<st c="629">Python 包</st> <st c="654">

+   通过网络爬虫、分割文档和嵌入块<st c="669">来索引数据</st> <st c="736">

+   使用向量相似度搜索检索相关文档<st c="746">

+   通过将检索到的上下文整合到 LLM 提示中生成响应<st c="807">

随着我们逐步分析代码，您将通过使用 LangChain、Chroma DB 和 OpenAI 的 API 等工具，以编程方式全面理解 RAG 过程中的每一步。<st c="878">这将为您提供一个坚实的基础，我们将在后续章节中在此基础上构建，增强和改进代码，以解决越来越复杂的难题。</st> <st c="1065">这将为您提供一个坚实的基础，我们将在后续章节中在此基础上构建，增强和改进代码，以解决越来越复杂的难题。</st> <st c="1215">

在后续章节中，我们将探讨可以帮助改进和定制管道以适应不同用例的技术，并克服在构建由 RAG 驱动的应用程序时出现的常见挑战。</st> <st c="1232">让我们深入其中，<st c="1434">开始构建！</st>

# 技术要求<st c="1467">

本章的代码可在以下位置找到：<st c="1490"> <st c="1530">[`github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG/tree/main/Chapter_02`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG/tree/main/Chapter_02)</st> <st c="1536">](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG/tree/main/Chapter_02)

<st c="1633">您需要在已配置为运行 Jupyter 笔记本的环境下运行本章的代码。</st> <st c="1738">熟悉 Jupyter 笔记本是使用本书的先决条件，而且很难在简短的文字中涵盖。</st> <st c="1874">设置笔记本环境有众多方法。</st> <st c="1932">有在线版本，可以下载的版本，大学为学生提供的笔记本环境，以及您可以使用的不同界面。</st> <st c="2082">如果您在公司进行这项操作，他们可能有一个您需要熟悉的环境。</st> <st c="2191">这些选项的设置指令各不相同，而且这些指令经常变化。</st> <st c="2295">如果您需要更新关于此类环境的知识，可以从 Jupyter 网站（[<st c="2407">https://docs.jupyter.org/en/latest/</st>](https://docs.jupyter.org/en/latest/)）开始。</st> <st c="2442">从这里开始，然后向您最喜欢的语言模型请求更多帮助以设置您的环境。</st> <st c="2521">。</st>

<st c="2528">我该使用什么？</st> <st c="2544">当我使用我的 Chromebook，通常在旅行时，我会在云环境中设置一个笔记本。</st> <st c="2655">我更喜欢 Google Colab 或他们的 Colab Enterprise 笔记本，您可以在 Google Cloud Platform 的 Vertex AI 部分找到。</st> <st c="2784">但这些环境需要付费，如果您活跃使用，通常每月超过 20 美元。</st> <st c="2866">如果您的活跃程度像我一样，每月可能超过 1000 美元！</st>

<st c="2925">作为一个成本效益的替代方案，当我很活跃时，我会在我的 Mac 上使用 Docker Desktop，它本地托管一个 Kubernetes 集群，并在集群中设置我的笔记本环境。</st> <st c="3110">所有这些方法都有一些环境要求，这些要求经常变化。</st> <st c="3196">最好做一点研究，找出最适合您情况的方法。</st> <st c="3282">对于基于 Windows 的计算机也有类似的解决方案。</st>

<st c="3338">最终，主要要求是找到一个您可以使用 Python 3 运行 Jupyter 笔记本的环境。</st> <st c="3457">我们将提供的代码将指示您需要安装的其他包。</st>

<st c="3541">注意</st>

<st c="3546">所有这些代码都假设您正在 Jupyter 笔记本中工作。</st> <st c="3611">您可以直接在 Python 文件（</st>`<st c="3656">.py</st>`<st c="3659">）中这样做，但您可能需要对其进行一些修改。</st> <st c="3702">在笔记本中运行它允许您逐个单元格地执行，并查看每个点发生的情况，以便更好地理解整个过程。</st> <st c="3843">。</st>

# <st c="3858">没有界面！</st>

<st c="3872">在下面的编码示例中，我们不会处理接口；我们将在</st> *<st c="3970">第六章</st>*<st c="3979">中介绍这一点。同时，我们将创建一个表示用户会输入的提示字符串变量，并将其用作完整接口输入的占位符。</st>

# <st c="4144">设置大型语言模型（LLM）账户</st>

<st c="4192">对于</st> <st c="4200">公众，OpenAI 的 ChatGPT 模型</st> <st c="4240">目前是最受欢迎和最知名的 LLM。</st> <st c="4293">然而，市场上还有许多其他 LLM，适用于各种用途。</st> <st c="4383">您并不总是需要使用最昂贵、最强大的 LLM。</st> <st c="4452">一些 LLM 专注于一个领域，例如 Meditron LLM，它是 Llama 2 的专注于医学研究的微调版本。</st> <st c="4575">如果您在医学领域，您可能想使用该 LLM，因为它可能在您的领域内比大型通用 LLM 表现得更好。</st> <st c="4703">通常，LLM 可以用作其他 LLM 的二次检查，因此在这些情况下您可能需要不止一个。</st> <st c="4805">我强烈建议您不要只使用您已经使用过的第一个 LLM，而要寻找最适合您需求的 LLM。</st> <st c="4933">但为了使本书早期内容更简单，我将讨论如何设置</st> <st c="5021">OpenAI 的 ChatGPT：</st>

1.  <st c="5038">访问 OpenAI</st> **<st c="5049">API</st>** <st c="5052">部分：</st> [<st c="5084">https://openai.com/api/</st>](https://openai.com/api/)<st c="5107">。</st>

1.  <st c="5108">如果您尚未设置账户，请现在就设置。</st> <st c="5159">网页可能会经常更改，但请查找注册位置。</st>

<st c="5220">警告</st>

<st c="5228">使用 OpenAI 的 API 需要付费！</st> <st c="5261">请谨慎使用！</st>

1.  <st c="5278">一旦您完成注册，请访问以下文档</st> [<st c="5329">https://platform.openai.com/docs/quickstart</st>](https://platform.openai.com/docs/quickstart) <st c="5372">并按照说明设置您的第一个</st> <st c="5422">API 密钥。</st>

1.  <st c="5430">在创建 API 密钥时，请给它一个容易记住的名字，并选择您想要实施的权限类型（</st>**<st c="5540">全部</st>**<st c="5544">、**<st c="5546">受限</st>**<st c="5556">或</st> **<st c="5561">只读</st>**<st c="5570">）。</st> <st c="5574">如果您不知道选择哪个选项，目前最好选择**<st c="5638">全部</st>** <st c="5641">。</st> <st c="5651">然而，请注意其他选项——您可能希望与其他团队成员分担各种责任，但限制某些类型的访问：</st>

    1.  **<st c="5800">全部</st>**<st c="5804">：此密钥将具有对所有</st> <st c="5858">OpenAI API 的读写访问权限。</st>

    1.  **<st c="5870">受限</st>**<st c="5881">：将显示可用 API 列表，为您提供对密钥可以访问哪些 API 的细粒度控制。</st> <st c="5997">您可以选择为每个 API 提供只读或写入访问权限。</st> <st c="6066">请确保您至少已启用在这些演示中将使用的模型和嵌入 API。</st>

    1.  **<st c="6160">只读</st>**<st c="6170">：此选项为您提供对所有 API 的只读访问权限。</st>

1.  <st c="6224">复制提供的密钥。</st> <st c="6248">您将很快将其添加到代码中。</st> <st c="6288">在此期间，请记住，如果此密钥与他人共享，任何获得此密钥的人都可以使用它，并且您将付费。</st> <st c="6429">因此，这是一个您希望视为绝密并采取适当预防措施以防止未经授权使用的密钥。</st>

1.  <st c="6550">OpenAI API 要求您提前购买积分才能使用 API。</st> <st c="6621">购买您感到舒适的金额，并且为了更安全，请确保</st> **<st c="6696">启用自动充值</st>** <st c="6716">选项已关闭。</st> <st c="6732">这将确保您</st> <st c="6737">只花费您打算花费的金额。</st>

<st c="6796">有了这些，您已经设置了将作为您 RAG 管道</st> *<st c="6865">大脑</st> <st c="6871">的关键组件：LLM！</st> <st c="6903">接下来，我们将设置您的开发环境，以便您可以连接到</st> <st c="6980">LLM。</st>

# <st c="6988">安装必要的软件包</st>

<st c="7022">确保这些软件包已安装到您的 Python 环境中。</st> <st c="7090">在笔记本的第一个单元中添加以下代码行：</st>

```py
 %pip install langchain_community
%pip install langchain_experimental
%pip install langchain-openai
%pip install langchainhub
%pip install chromadb
%pip install langchain
%pip install beautifulsoup4
```

<st c="7355">前面的代码使用</st> `<st c="7419">pip</st>` <st c="7422">包管理器安装了几个 Python 库，这是运行我提供的代码所必需的。</st> <st c="7496">以下是每个库的</st> <st c="7518">分解：</st>

+   `<st c="7531">langchain_community</st>`<st c="7551">：这是一个</st> <st c="7563">由社区驱动的 LangChain 库的软件包，LangChain 是一个用于构建具有 LLMs 应用程序的开源框架。</st> <st c="7687">它提供了一套工具和组件，用于与 LLMs 协同工作并将它们集成到</st> <st c="7777">各种应用程序中。</st>

+   `<st c="7798">langchain_experimental</st>`<st c="7821">：</st> `<st c="7828">langchain_experimental</st>` <st c="7850">库</st> <st c="7858">提供了核心 LangChain 库之外的一些额外功能和工具，这些功能和工具尚未完全稳定或适用于生产，但仍可用于实验</st> <st c="8028">和探索。</st>

+   `<st c="8044">langchain-openai</st>`<st c="8061">：这个</st> <st c="8069">包提供了 LangChain 与 OpenAI 语言模型之间的集成。</st> <st c="8146">它允许你轻松地将 OpenAI 的模型，如 ChatGPT 4 或 OpenAI 嵌入服务，集成到你的</st> <st c="8261">LangChain 应用程序中。</st>

+   `<st c="8284">langchainhub</st>`<st c="8297">：这个</st> <st c="8305">包提供了一组预构建的组件和模板，用于 LangChain 应用程序。</st> <st c="8401">它包括各种代理、内存组件和实用函数，可用于加速基于 LangChain 的应用程序的开发。</st>

+   `<st c="8549">chromadb</st>`<st c="8558">：这是 Chroma DB 的</st> <st c="8573">包名，Chroma DB 是一个高性能的嵌入/向量数据库，旨在进行高效的相似性搜索和检索。</st>

+   `<st c="8701">langchain</st>`<st c="8711">：这是</st> <st c="8725">核心 LangChain 库本身。</st> <st c="8757">它提供了一个框架和一系列抽象，用于构建基于 LLM 的应用程序。</st> <st c="8844">LangChain 包括构建有效的 RAG 管道所需的所有组件，包括提示、内存管理、代理以及与其他各种外部工具和服务集成。</st>

<st c="9028">在运行前面的第一行之后，你需要重启内核才能访问你刚刚在环境中安装的所有新包。</st> <st c="9190">根据你所在的环境，这可以通过多种方式完成。</st> <st c="9271">通常，你会看到一个可以使用的刷新按钮，或者菜单中的</st> **<st c="9329">重启内核</st>** <st c="9343">选项。</st>

<st c="9363">如果你找不到重启内核的方法，请添加此单元格并</st> <st c="9439">运行它：</st>

```py
 import IPython
app = IPython.Application.instance(;
app.kernel.do_shutdown(True)
```

<st c="9527">这是一个在 IPython 环境（笔记本）中执行内核重启的代码版本（注意：通常不需要它，但这里提供以备不时之需）。</st> <st c="9622">你不应该需要它，但它在这里供你使用。</st> <st c="9673">以防万一！</st>

<st c="9681">一旦安装了这些包并重启了你的内核，你就可以开始编码了！</st> <st c="9779">让我们从导入你环境中刚刚安装的许多包开始。</st>

## <st c="9866">导入</st>

<st c="9874">现在，让我们导入所有执行 RAG 相关任务所需的</st> <st c="9923">库。</st> <st c="9955">我在每个导入组顶部提供了注释，以指示这些导入与 RAG 的哪个领域相关。</st> <st c="10074">结合以下列表中的描述，这为你的第一个</st> <st c="10201">RAG 管道</st>提供了基本介绍：

```py
 import os
from langchain_community.document_loaders import WebBaseLoader
import bs4
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
```

<st c="10644">让我们逐一查看</st> <st c="10672">这些导入：</st>

+   `<st c="10686">import os</st>`<st c="10696">: 这</st> <st c="10703">提供了一种与操作系统交互的方式。</st> <st c="10758">这对于执行诸如访问环境变量和操作</st> <st c="10854">文件路径等操作非常有用。</st>

+   `<st c="10865">from langchain_community.document_loaders import WebBaseLoader</st>`<st c="10928">: `<st c="10935">WebBaseLoader</st>` <st c="10948">类是一个文档加载器，可以获取并加载网页</st> <st c="11010">作为文档。</st>

+   `<st c="11023">import bs4</st>`<st c="11034">: `<st c="11041">bs4</st>` <st c="11044">模块，代表</st> **<st c="11070">Beautiful Soup 4</st>**<st c="11086">，是一个流行的网络抓取和解析 HTML</st> <st c="11143">或 XML 文档的库。</st> <st c="11161">由于我们将处理网页，这为我们提供了一个简单的方法来分别提取标题、内容和</st> <st c="11266">头部信息。</st>

+   `<st c="11285">import openai</st>`<st c="11299">: 这提供了与 OpenAI 语言模型和 API 交互的接口。</st> <st c="11371">它允许我们使用 OpenAI 的模型与 LangChain 直接交互。</st>

+   `<st c="11380">from langchain_openai import ChatOpenAI, OpenAIEmbeddings</st>`<st c="11438">: 这导入了</st> `<st c="11459">ChatOpenAI</st>` <st c="11469">(用于 LLM) 和</st> `<st c="11488">OpenAIEmbeddings</st>` <st c="11504">(用于嵌入)，它们是使用 OpenAI 模型并直接与 LangChain 工作的特定语言模型和嵌入的实现。</st>

+   `<st c="11655">from langchain import hub</st>`<st c="11681">: `<st c="11688">hub</st>` <st c="11691">组件提供了访问各种预构建组件和工具的途径，用于与</st> <st c="11781">语言模型一起工作。</st>

+   `<st c="11797">from langchain_core.output_parsers import StrOutputParser</st>`<st c="11855">: 此组件解析语言模型生成的输出并提取相关信息。</st> <st c="11962">在这种情况下，它假定语言模型的输出是一个字符串，并返回</st> <st c="12044">它本身。</st>

+   `<st c="12053">from langchain_core.runnables import RunnablePassthrough</st>`<st c="12110">: 此组件将问题或查询直接传递，不进行任何修改。</st> <st c="12192">它允许将问题直接用于链的后续步骤。</st>

+   `<st c="12269">Import chromadb</st>`<st c="12285">: 如前所述，</st> `<st c="12313">chromadb</st>` <st c="12321">导入 Chroma DB 向量存储库，这是一个为高效相似性搜索和检索而设计的高性能嵌入/向量数据库。</st>

+   `<st c="12458">from langchain_community.vectorstores import Chroma</st>`<st c="12510">: 这提供了使用 LangChain 与 Chroma 向量数据库交互的接口。</st> <st c="12584">Chroma 是一个高性能的嵌入/向量数据库，专为高效的相似性搜索和检索而设计。</st>

+   `<st c="12600">from langchain_experimental.text_splitter import SemanticChunker</st>`<st c="12665">：文本分割器通常是一个函数，我们使用它根据指定的块大小和重叠来将文本分割成小块。</st> <st c="12801">这个分割器被称为</st> `<st c="12825">SemanticChunker</st>`<st c="12840">，是 Langchain_experimental</st> <st c="12897">库提供的一个实验性文本分割工具。</st> <st c="12929">SemanticChunker</st> <st c="12949">的主要目的是将长文本分解成更易于管理的片段，同时保留每个片段的</st> <st c="13040">语义连贯性和上下文。</st>

<st c="13086">这些导入提供了设置您的 RAG 管道所需的 Python 基本包。</st> <st c="13188">您的下一步将是将您的环境连接到</st> <st c="13242">OpenAI 的 API。</st>

## <st c="13255">OpenAI 连接</st>

<st c="13273">以下代码行是一个非常</st> <st c="13310">简单的示例，展示了您的 API 密钥如何被系统接收。</st> <st c="13386">然而，这不是使用 API 密钥的安全方式。</st> <st c="13439">有许多更安全的方式来完成这项任务。</st> <st c="13485">如果您有偏好，现在就实施它，否则，我们将在</st> *<st c="13613">第五章</st>**<st c="13622">中介绍一种流行的更安全的方法。</st>*

<st c="13623">您需要将</st> `<st c="13654">sk-###################</st>` <st c="13676">替换为您实际的 OpenAI</st> <st c="13701">API 密钥：</st>

```py
 os.environ['OPENAI_API_KEY'] = 'sk-###################'
openai.api_key = os.environ['OPENAI_API_KEY']
```

<st c="13811">重要</st>

<st c="13821">这只是一个简单的示例；请使用安全的方法来隐藏您的</st> <st c="13895">API 密钥！</st>

<st c="13903">您可能已经猜到了，这个 OpenAI API 密钥将被用来连接到 ChatGPT LLM。</st> <st c="13999">但 ChatGPT 并不是我们将从 OpenAI 使用的唯一服务。</st> <st c="14060">这个 API 密钥也用于访问 OpenAI 嵌入服务。</st> <st c="14126">在下一节中，我们将专注于 RAG 过程的索引阶段编码，我们将利用 OpenAI 嵌入服务将您的内容转换为向量嵌入，这是 RAG 管道的关键方面。</st>

# <st c="14336">索引</st>

下几个步骤代表的是<st c="14345">索引</st> <st c="14379">阶段，在这个阶段我们获取目标数据，对其进行预处理，并将其矢量化。</st> <st c="14462">这些</st> <st c="14467">步骤通常是在</st> <st c="14489">离线</st> <st c="14496">完成的，这意味着它们是为了</st> <st c="14523">为后续的应用使用做准备。</st> <st c="14564">但在某些情况下，实时完成所有这些步骤可能是有意义的，例如在数据变化迅速的环境中，所使用的数据相对较小。</st> <st c="14725">在这个特定的例子中，步骤如下：</st> <st c="14767">。

1.  <st c="14778">网页加载</st> <st c="14791">和抓取。</st>

1.  将数据分割成 Chroma DB<st c="14804">向量化算法</st> <st c="14865">可消化的块。</st>

1.  <st c="14887">嵌入和索引</st> <st c="14911">这些块。</st>

1.  将这些块和嵌入添加到 Chroma DB<st c="14924">向量存储。</st> <st c="14977">。

让我们从第一步开始：网页加载<st c="14990">和抓取。</st> <st c="15036">。

## <st c="15049">网页加载和抓取</st>

首先，我们需要<st c="15074">拉取</st> <st c="15101">我们的数据。</st> <st c="15114">这当然可以是任何东西，但我们必须</st> <st c="15140">从某个地方开始！</st>

对于我们的例子，我提供了一个基于 LangChain 提供的某些内容的网页示例。</st> <st c="15179">我采用了 LangChain 在</st> <st c="15265">第一章</st><st c="15274">中提供的原始结构。</st> <st c="15352">在</st> [<st c="15355">https://lilianweng.github.io/posts/2023-06-23-agent/</st>](https://lilianweng.github.io/posts/2023-06-23-agent/)<st c="15407">。</st>

如果你在阅读时该网页仍然可用，你也可以尝试那个网页，但务必将你用于查询内容的提问改为更适合该页面上内容的提问。</st> <st c="15609">如果你更改网页，你还需要重新启动你的内核；否则，如果你重新运行加载器，它将包含两个网页的内容。</st> <st c="15751">这可能正是你想要的，但我只是让你知道！</st> <st c="15799">。

我还鼓励你尝试使用其他网页，看看这些其他网页会带来什么挑战。</st> <st c="15808">与大多数网页相比，这个例子涉及的数据非常干净，而大多数网页通常充满了你不想看到的广告和其他内容。</st> <st c="15914">但也许你可以找到一个相对干净的博客文章并将其拉取进来？</st> <st c="16068">也许你可以自己创建一个？</st> <st c="16138">尝试不同的网页</st> <st c="16169">并看看结果！</st>

```py
 loader = WebBaseLoader(
    web_paths=("https://kbourne.github.io/chapter1.html",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
           class_=("post-content", "post-title",
                   "post-header")
        )
    ),
)
docs = loader.load()
```

<st c="16407">前面的</st> <st c="16421">代码开始使用</st> `<st c="16449">WebBaseLoader</st>` <st c="16462">类从</st> `<st c="16478">langchain_community document_loaders</st>` <st c="16514">模块</st> <st c="16521">加载网页作为文档。</st> <st c="16554">让我们分解一下：</st>

1.  <st c="16574">创建</st> `<st c="16588">WebBaseLoader</st>` <st c="16601">实例：The</st> `<st c="16616">WebBaseLoader</st>` <st c="16629">类使用以下参数实例化：</st>

    +   `<st c="16682">web_paths</st>`<st c="16692">：一个包含要加载的网页 URL 的元组。</st> <st c="16754">在这种情况下，它包含一个单独的</st> <st c="16789">URL：</st> `<st c="16794">https://kbourne.github.io/chapter1.html</st>`<st c="16833">。</st>

    +   `<st c="16834">bs_kwargs</st>`<st c="16844">：一个字典，包含要传递给</st> `<st c="16901">BeautifulSoup</st>` <st c="16914">解析器的关键字参数。</st>

    +   `<st c="16922">parse_only</st>`<st c="16933">：一个</st> `<st c="16938">bs4.SoupStrainer</st>` <st c="16954">对象指定了要解析的 HTML 元素。</st> <st c="17000">在这种情况下，它被设置为仅解析具有 CSS 类别的元素，例如</st> `<st c="17081">post-content</st>`<st c="17093">,</st> `<st c="17095">post-title</st>`<st c="17105">,</st> <st c="17107">和</st> `<st c="17111">post-header</st>`<st c="17122">。</st>

1.  <st c="17123">The</st> `<st c="17128">WebBaseLoader</st>` <st c="17141">实例启动一系列步骤，代表将文档加载到您的环境中：在</st> `<st c="17274">loader</st>`<st c="17280">上调用 load 方法，这是</st> `<st c="17286">WebBaseLoader</st>` <st c="17299">实例，它将指定的网页作为文档获取和加载。</st> <st c="17370">内部，</st> `<st c="17382">loader</st>` <st c="17388">做了很多工作！</st>

    <st c="17404">以下是基于这段小代码所执行的步骤：</st>

    1.  <st c="17476">向指定的 URL 发送 HTTP 请求以获取</st> <st c="17532">网页。</st>

    1.  <st c="17542">使用</st> `<st c="17590">BeautifulSoup</st>`<st c="17603">解析网页的 HTML 内容，仅考虑由</st> `<st c="17652">parse_only</st>` <st c="17662">参数指定的元素。</st>

    1.  <st c="17673">从解析的</st> <st c="17725">HTML 元素中提取相关文本内容。</st>

    1.  <st c="17739">为包含提取的文本内容和元数据（如</st> <st c="17856">源 URL）的每个网页创建</st> <st c="17748">Document</st> <st c="17756">对象。</st>

<st c="17867">生成的</st> `<st c="17882">Document</st>` <st c="17890">对象存储在</st> `<st c="17917">docs</st>` <st c="17921">变量中，以便在</st> <st c="17950">我们的代码中进一步使用！</st>

<st c="17959">我们传递给</st> `<st c="17995">bs4</st>` <st c="17998">(</st>`<st c="18000">post-content</st>`<st c="18012">,</st> `<st c="18014">post-title</st>`<st c="18024">, 和</st> `<st c="18030">post-header</st>`<st c="18041">) 的类是 CSS 类。</st> <st c="18061">如果您正在使用没有这些 CSS 类的 HTML 页面，这将不起作用。</st> <st c="18149">因此，如果您使用不同的 URL 并且没有获取数据，请查看您正在爬取的 HTML 中的 CSS 标签。</st> <st c="18279">许多网页确实使用这种模式，但并非所有！</st> <st c="18328">爬取网页会带来许多挑战</st> <st c="18372">，就像这样。</st>

<st c="18382">一旦您从数据源收集了文档，您需要对其进行预处理。</st> <st c="18474">在这种情况下，这</st> <st c="18493">涉及到分割。</st>

## <st c="18512">分割</st>

<st c="18522">如果您正在使用提供的 URL，您</st> <st c="18562">将只会解析具有</st> `<st c="18600">post-content</st>`<st c="18612">,</st> `<st c="18614">post-title</st>`<st c="18624">, 和</st> `<st c="18630">post-header</st>` <st c="18641">CSS 类</st> <st c="18655">的元素。</st> <st c="18655">这将从主要文章主体（通常通过</st> `<st c="18744">post-content</st>` <st c="18756">类）提取文本内容，博客文章的标题（通常通过</st> `<st c="18819">post-title</st>` <st c="18829">类）以及任何标题信息（通常通过</st> `<st c="18892">post-header</st>` <st c="18903">类）。</st>

<st c="18911">如果您好奇，这是该文档在网页上的样子（</st>*<st c="18988">Figure 2</st>**<st c="18997">.1</st>*<st c="18999">）：</st>

![Figure 2.1 – A web page that we will process](img/B22475_02_01.jpg)

<st c="21860">Figure 2.1 – A web page that we will process</st>

<st c="21904">它也涉及到很多页面！</st> <st c="21934">这里的内容也很多，对于 LLM 直接处理来说太多了。</st> <st c="22007">因此，我们需要将文档分割成</st> <st c="22051">可消化的块：</st>

```py
 text_splitter = SemanticChunker(OpenAIEmbeddings())
splits = text_splitter.split_documents(docs)
```

<st c="22166">LangChain 中有很多文本分割器可用，但我选择从一种实验性但非常有趣的选项开始，称为</st> `<st c="22300">SemanticChunker</st>`<st c="22315">。正如我之前提到的，当谈到导入时，</st> `<st c="22376">SemanticChunker</st>` <st c="22391">专注于将长文本分解成更易于管理的片段，同时保留每个片段的语义连贯性和上下文。</st>

<st c="22521">其他文本分割器通常采用任意长度的块，这不是上下文感知的，当重要内容被分割器分割时，这会引发问题。</st> <st c="22691">有方法可以解决这个问题，我们将在</st> *<st c="22749">第十一章</st>*<st c="22759">中讨论，但到目前为止，只需知道</st> `<st c="22789">SemanticChunker</st>` <st c="22804">专注于考虑上下文，而不仅仅是块中的任意长度。</st> <st c="22889">还应注意的是，它仍然被视为实验性的，并且正在持续开发中。</st> <st c="22993">在第</st> *<st c="22996">第十一章</st>*<st c="23006">中，我们将对其进行测试，与可能的其他最重要的文本分割器</st> `<st c="23092">RecursiveCharacter TextSplitter</st>`<st c="23123">进行比较，看看哪个分割器与</st> <st c="23164">此内容配合得最好。</st>

<st c="23177">还应注意的是，你在这段代码中使用的</st> `<st c="23211">SemanticChunker</st>` <st c="23226">分割器使用的是</st> `<st c="23262">OpenAIEmbeddings</st>`<st c="23278">，处理嵌入需要付费。</st> <st c="23326">目前，OpenAI 的嵌入模型每百万个标记的成本在 0.02 美元到 0.13 美元之间，具体取决于你使用的模型。</st> <st c="23446">在撰写本文时，如果你没有指定嵌入模型，OpenAI 将默认使用</st> `<st c="23530">text-embedding-ada-002</st>` <st c="23552">模型，每百万个标记的成本为 0.02 美元。</st> <st c="23609">如果你想避免成本，可以回退到</st> `<st c="23653">RecursiveCharacter TextSplitter</st>`<st c="23684">，我们将在</st> *<st c="23713">第十一章</st>*<st c="23723">中介绍。</st>

<st c="23724">我鼓励你尝试不同的分割器，看看会发生什么！</st> <st c="23803">例如，你认为你从</st> `<st c="23857">RecursiveCharacter TextSplitter</st>` <st c="23888">》中获得的结果比从</st> `<st c="23899">SemanticChunker</st>`<st c="23914">》获得的结果更好吗？</st> <st c="23941">也许在你的特定情况下，速度比质量更重要——哪一个更快？</st>

<st c="24030">一旦将内容分块，下一步就是将其转换为我们已经讨论了很多的向量嵌入！</st> <st c="24146">！</st>

## <st c="24157">嵌入和索引块</st>

<st c="24191">接下来的几个步骤</st> <st c="24211">代表检索和生成步骤，我们将使用 Chroma DB 作为向量数据库。</st> <st c="24284">正如之前多次提到的，Chroma DB 是一个非常好的向量存储！</st> <st c="24309">我选择这个向量存储是因为它易于本地运行，并且对于此类演示效果良好，但它确实是一个相当强大的向量存储。</st> <st c="24377">如您所回忆的，当我们讨论词汇和向量存储与向量数据库之间的区别时，Chroma DB 确实既是！</st> <st c="24521">尽管如此，Chroma 只是您向量存储的许多选项之一。</st> <st c="24660">在第</st> *<st c="24723">7 章</st>*<st c="24732">中，我们将讨论许多向量存储选项以及选择其中一个而不是另一个的原因。</st> <st c="24825">其中一些选项甚至提供免费的向量</st> <st c="24872">嵌入生成。</st>

<st c="24893">我们在这里也使用 OpenAI 嵌入，它将使用您的 OpenAI 密钥将您的数据块发送到 OpenAI API，将它们转换为嵌入，并以数学形式发送回来。</st> <st c="25099">请注意，这</st> *<st c="25114">确实</st>* <st c="25118">需要付费！</st> <st c="25131">每个嵌入的费用是几分之一便士，但这是值得注意的。</st> <st c="25203">因此，如果您预算紧张，请谨慎使用此代码！</st> <st c="25288">在第</st> *<st c="25291">7 章</st>*<st c="25300">中，我们将回顾一些使用免费向量服务免费生成这些嵌入的方法</st> <st c="25391">的方法：</st>

```py
 vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
```

<st c="25524">首先，我们使用</st> `<st c="25542">Chroma.from_documents</st>` <st c="25596">方法创建 Chroma 向量存储，该方法用于从分割文档创建 Chroma 向量存储。</st> <st c="25679">这是我们创建 Chroma 数据库的许多方法之一。</st> <st c="25747">这通常取决于来源，但针对这种方法，它需要以下参数：</st> <st c="25830">以下参数：</st>

+   `<st c="25851">文档</st>`<st c="25861">：从上一个代码片段中获得的分割文档（分割）列表</st>

+   `<st c="25940">嵌入</st>`<st c="25950">：OpenAIEmbeddings 类的实例，用于生成文档的嵌入</st>

<st c="26050">在内部，该方法执行以下操作：</st>

1.  <st c="26096">它遍历分割列表中的每个</st> `<st c="26119">Document</st>` <st c="26127">对象。</st>

1.  <st c="26154">对于每个</st> `<st c="26164">Document</st>` <st c="26172">对象，它使用提供的</st> `<st c="26202">OpenAIEmbeddings</st>` <st c="26218">实例生成一个</st> `<st c="26243">嵌入向量</st>`。</st>

1.  <st c="26260">它将文档文本及其对应的嵌入向量存储在 Chroma</st> <st c="26342">向量数据库中。</st>

<st c="26358">在这个阶段，你现在有一个名为</st> `<st c="26412">vectorstore</st>`<st c="26423">的向量数据库，里面充满了嵌入，这些是…？</st> <st c="26467">没错——是你刚刚爬取的网页上所有内容的数学表示！</st> <st c="26569">太酷了！</st>

<st c="26577">但下一部分是什么——一个检索器？</st> <st c="26620">这是狗类的吗？</st> <st c="26651">不是。</st> <st c="26657">这是创建你将用于在新向量数据库上执行向量相似性搜索的机制。</st> <st c="26773">你直接在</st> `<st c="26786">as_retriever</st>` <st c="26798">方法上调用</st> `<st c="26819">vectorstore</st>` <st c="26830">实例来创建检索器。</st> <st c="26865">检索器是一个提供方便接口以执行这些相似性搜索，并根据</st> <st c="27042">这些搜索从向量数据库中检索相关文档的对象。</st>

<st c="27057">如果你只想执行文档检索过程，你可以。</st> <st c="27127">这并不是代码的官方部分，但如果你想测试这个，请在一个额外的单元中添加它并</st> <st c="27232">运行它：</st>

```py
 query = "How does RAG compare with fine-tuning?" relevant_docs = retriever.get_relevant_documents(query)
relevant_docs
```

<st c="27358">输出</st> <st c="27370">应该是我在此代码中稍后列出当我指出传递给 LLM 的内容时，但它本质上是一个存储在</st> `<st c="27512">vectorstore</st>` <st c="27523">向量数据库中的内容列表，该数据库与</st> <st c="27564">查询最相似。</st>

<st c="27574">你不觉得印象深刻吗？</st> <st c="27597">这是一个简单的例子，但这是你用来访问数据和为你的组织超级充电生成式 AI 应用的基础工具！</st>

<st c="27791">然而，在这个应用阶段，你只创建了接收器。</st> <st c="27871">你还没有在 RAG 管道中使用它。</st> <st c="27921">我们将在下一部分回顾如何做到这一点！</st>

# <st c="27956">检索和生成</st>

<st c="27981">在代码中，检索和生成阶段</st> <st c="28030">被组合在我们设置的链中，以表示整个 RAG 流程。</st> <st c="28108">这利用了来自</st> **<st c="28153">LangChain Hub</st>**<st c="28166">的预构建组件，例如</st> **<st c="28176">提示模板</st>**<st c="28192">，并将它们与选定的 LLM 集成。</st> <st c="28235">我们还将</st> <st c="28247">利用</st> **<st c="28260">LangChain 表达式语言</st>** <st c="28289">(</st>**<st c="28291">LCEL</st>**<st c="28295">) 来</st> <st c="28301">定义一个操作链，根据输入问题检索相关文档，格式化检索内容，并将其输入到 LLM 以生成响应。</st> <st c="28473">总的来说，我们在检索和生成中采取的步骤</st> <st c="28532">如下：</st>

1.  <st c="28543">接收一个</st> <st c="28554">用户查询。</st>

1.  <st c="28565">将那个</st> <st c="28581">用户查询向量化。</st>

1.  <st c="28592">对向量存储执行相似度搜索，以找到与用户查询向量最接近的向量及其</st> <st c="28712">相关内容。</st>

1.  <st c="28731">将检索到的内容传递给一个提示模板，这个过程被称为</st> <st c="28799">激活</st> <st c="28811">。</st>

1.  <st c="28812">将那个</st> *<st c="28823">激活的</st>* <st c="28831">提示传递给</st> <st c="28842">LLM。</st>

1.  <st c="28850">一旦你</st> <st c="28859">从 LLM 收到响应，将其呈现给</st> <st c="28907">用户。</st>

<st c="28916">从编码的角度来看，我们将首先定义提示模板，以便在接收到用户查询时有所依据。</st> <st c="29058">我们将在下一节中介绍这一点。</st>

## <st c="29097">来自 LangChain Hub 的提示模板</st>

<st c="29137">LangChain Hub</st> <st c="29155">是一个包含预构建组件和模板的集合，可以轻松集成到 LangChain 应用程序中。</st> <st c="29269">它提供了一个集中式存储库，用于</st> <st c="29310">共享和发现可重用组件，例如提示、代理和实用工具。</st> <st c="29395">在此，我们从 LangChain Hub 调用一个提示模板，并将其分配给</st> `<st c="29477">prompt</st>`<st c="29483">，这是一个表示我们将传递给</st> <st c="29537">LLM</st> 的提示模板：

```py
 prompt = hub.pull("jclemens24/rag-prompt")
print(prompt)
```

<st c="29602">此代码使用 LangChain 中心的</st> `<st c="29684">pull</st>` <st c="29688">方法从</st> `<st c="29703">hub</st>` <st c="29706">模块中检索预构建的提示模板。</st> <st c="29715">提示模板通过</st> `<st c="29756">jclemens24/rag-prompt</st>` <st c="29777">字符串进行标识。</st> <st c="29786">此标识符遵循</st> *<st c="29814">仓库/组件</st> <st c="29834">约定，其中</st> *<st c="29853">仓库</st> <st c="29863">代表托管组件的组织或用户，而</st> *<st c="29927">组件</st> <st c="29936">代表被拉取的具体组件。</st> <st c="29985">`<st c="29989">rag-prompt</st>` <st c="29999">组件表明它是一个为</st> <st c="30048">RAG 应用</st>设计的提示。</st>

<st c="30065">如果你使用</st> `<st c="30099">print(prompt)</st>`<st c="30112">打印提示信息，你可以看到这里使用了什么，以及</st> <st c="30165">输入的内容：</st>

```py
 input_variables=['context', 'question'] messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="You are an assistant for question-answering tasks. Use the following pieces of retrieved-context to answer the question. If you don't know the answer, just say that you don't know.\nQuestion: {question} \nContext: {context} \nAnswer:"))]
```

<st c="30564">这是传递给 LLM 的提示信息的初始部分，它在这个例子中告诉</st> <st c="30659">它：</st>

```py
 "You are an assistant for question-answering tasks. Use the following pieces of retrieved-context to answer the question. If you don't know the answer, just say that you don't know. Question: {question}
Context: {context}
Answer:"
```

<st c="30898">稍后，你将添加</st> <st c="30913">问题</st> <st c="30926">和</st> `<st c="30931">上下文</st>` <st c="30938">变量来</st> *<st c="30952">填充</st> <st c="30959">提示信息，但以这种格式开始可以优化它以更好地适用于</st> <st c="31034">RAG 应用。</st>

<st c="31051">注意</st>

<st c="31056">`<st c="31061">jclemens24/rag-prompt</st>` <st c="31082">字符串是预定义起始提示信息的一个版本。</st> <st c="31141">访问 LangChain 中心以找到更多选项——你甚至可能找到一个更适合你</st> <st c="31229">需求</st>的：[<st c="31236">https://smith.langchain.com/hub/search?q=rag-prompt</st>](https://smith.langchain.com/hub/search?q=rag-prompt)<st c="31287">。</st>

<st c="31288">你也可以使用自己的！</st> <st c="31316">在撰写本文时，我可以数出超过 30 个选项！</st>

<st c="31367">提示模板是 RAG 管道的关键部分，因为它代表了如何与 LLM 通信以获取你寻求的响应。</st> <st c="31513">但在大多数 RAG 管道中，将提示信息转换为可以与提示模板一起工作的格式并不像只是传递一个字符串那样简单。</st> <st c="31673">在这个例子中，</st> `<st c="31694">上下文</st>` <st c="31701">变量代表我们从检索器获取的内容，但还不是字符串格式</st> <st c="31795">！</st> <st c="31800">我们将逐步说明如何将检索到的内容转换为所需的正确字符串格式。</st>

## <st c="31901">格式化函数以匹配下一步输入</st>

<st c="31964">首先，我们将设置一个</st> <st c="31988">函数，该函数接受检索到的文档列表（docs）</st> <st c="32048">作为输入：</st>

```py
 def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
```

<st c="32133">在这个函数内部，使用了一个生成器表达式，</st> `<st c="32180">(doc.page_content for doc in docs)</st>`<st c="32214">，用于从每个文档对象中提取</st> `<st c="32239">page_content</st>` <st c="32251">属性。</st> <st c="32289">`<st c="32293">page_content</st>` <st c="32305">属性代表每个文档的</st> `<st c="32347">文本内容。</st>

<st c="32361">注意</st>

<st c="32366">在这种情况下，一个</st> *<st c="32383">文档</st>* <st c="32391">并不是你之前爬取的整个文档。</st> <st c="32445">它只是其中的一小部分，但我们通常称</st> <st c="32503">这些文档。</st>

<st c="32519">`<st c="32524">join</st>` <st c="32528">方法被调用在</st> `<st c="32553">\n\n</st>` <st c="32557">字符串上，用于将每个文档的内容之间插入两个换行符来连接</st> `<st c="32580">page_content</st>` <st c="32592">。</st> <st c="32671">格式化的字符串由`<st c="32711">format_docs</st>` <st c="32722">函数返回，以表示字典中通过管道输入到提示对象中的`<st c="32749">context</st>` <st c="32756">键。</st>

<st c="32816">此函数的目的是将检索器的输出格式化为字符串格式，以便在检索器步骤之后，在链中的下一步中使用。</st> <st c="32995">我们稍后会进一步解释这一点，但像这样的简短函数对于 LangChain 链来说通常是必要的，以便在整个</st> <st c="33150">链中匹配输入和输出。</st>

<st c="33163">接下来，在我们能够创建我们的 LangChain 链之前，我们将回顾最后一步 – 那就是定义我们将要在</st> <st c="33283">该链中使用的 LLM。</st>

## <st c="33294">定义你的 LLM</st>

<st c="33312">让我们设置你将使用的</st> <st c="33329">LLM 模型：</st>

```py
 llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
```

<st c="33411">前面的代码创建了一个来自</st> `<st c="33484">langchain_openai</st>` <st c="33500">模块的</st> `<st c="33458">ChatOpenAI</st>` <st c="33468">类的实例，该模块作为 OpenAI 语言模型的接口，具体是</st> `<st c="33583">GPT-4o mini</st>` <st c="33583">模型。</st> <st c="33603">尽管这个模型较新，但它以比旧模型大幅折扣的价格发布。</st> <st c="33699">使用这个模型可以帮助你降低推理成本，同时仍然允许你使用最新的模型！</st> <st c="33805">如果你想尝试 ChatGPT 的不同版本，例如</st> `<st c="33870">gpt-4</st>`<st c="33875">，你只需更改模型名称。</st> <st c="33913">在 OpenAI API 网站上查找最新的模型 – 他们经常添加！</st>

## <st c="33987">使用 LCEL 设置 LangChain 链</st>

<st c="34027">这个</st> *<st c="34033">链</st>* <st c="34038">是以</st> <st c="34045">LangChain 特有的代码格式</st> <st c="34087">LCEL</st> <st c="34094">编写的。</st> <st c="34159">从现在开始，你将看到我会在代码中一直使用 LCEL。</st> <st c="34159">这不仅使代码更容易阅读和更简洁，而且开辟了专注于提高你</st> <st c="34308">LangChain 代码的速度和效率的新技术。</st>

<st c="34323">如果你遍历这个链，你会看到它提供了整个</st> <st c="34416">RAG 过程</st>的绝佳表示：

```py
 rag_chain = (
    {"context": retriever | format_docs,
     "question": RunnablePassthrough()}
         | prompt
         | llm
         | StrOutputParser()
)
```

<st c="34551">所有这些组件都已经描述过了，但为了总结，</st> `<st c="34627">rag_chain</st>` <st c="34636">变量代表了一个使用 LangChain 框架的操作链。</st> <st c="34710">让我们遍历链的每一步，深入挖掘每个点正在发生的事情：</st> <st c="34787">：</st>

1.  `<st c="34990">rag_chain</st>` <st c="34999">变量稍后我们将传递一个“问题”。如前述代码所示，链从定义了两个键的字典开始：</st> `<st c="35142">"context"</st>` <st c="35151">和</st> `<st c="35156">"question"</st>`<st c="35166">。问题部分相当直接，但上下文从何而来？</st> <st c="35251">“</st> `<st c="35255">"context"</st>` <st c="35264">键分配的结果是</st> `<st c="35299">retriever</st>` <st c="35308">|</st> `<st c="35311">format_docs</st>` <st c="35322">操作的结果。</st>

+   <st c="37050">我们可以看到另一个管道（</st>`<st c="37076">|</st>`<st c="37078">）后面跟着</st> `<st c="37096">提示</st>` <st c="37102">对象，我们将</st> *<st c="37118">管道</st>* <st c="37122">变量（在一个字典中）放入那个提示对象。</st> <st c="37180">这被称为提示的填充。</st> <st c="37219">如前所述，</st> `<st c="37248">提示</st>` <st c="37254">对象是一个提示模板，它定义了我们将要传递给 LLM 的内容，并且通常包括首先填充/填充的输入变量（上下文和问题）。</st> <st c="37423">这一步骤的结果是完整的提示文本，作为字符串，变量填充了上下文和问题的占位符。</st> <st c="37564">然后，我们又有另一个管道（</st>`<st c="37592">|</st>`<st c="37594">）和</st> `<st c="37604">llm</st>` <st c="37607">对象，这是我们之前定义的。</st> <st c="37640">正如我们已经看到的，链中的这一步取前一步的输出，即包含前几步所有信息的提示字符串。</st> <st c="37811"></st> `<st c="37815">llm</st>` <st c="37818">对象代表我们设置的</st> `<st c="37855">语言模型</st>` <st c="37889">ChatGPT 4o</st>`<st c="37899">。格式化的提示字符串作为输入传递给语言模型，根据提供的上下文</st> `<st c="38028">和问题</st>` <st c="38041">生成响应。</st> *   <st c="38041">这似乎已经足够了，但当你使用 LLM API 时，它不仅仅发送你可能在 ChatGPT 中输入文本时看到的文本。</st> <st c="38202">它是以 JSON 格式发送的，并包含很多其他数据。</st> <st c="38271">因此，为了使事情简单，我们将</st> *<st c="38314">管道</st>* <st c="38318">LLM 的输出传递到下一步，并使用 LangChain 的</st> `<st c="38373">StrOutputParser()</st>` <st c="38390">对象。</st> <st c="38399">请注意，</st> `<st c="38409">StrOutputParser()</st>` <st c="38426">是 LangChain 中的一个实用类，它将语言模型的关键输出解析为字符串格式。</st> <st c="38530">它不仅去除了你现在不想处理的所有信息，而且还确保生成的响应以</st> `<st c="38677">字符串</st>` <st c="38686">的形式返回。</st>

<st c="38686">让我们花点时间来欣赏我们刚才所做的一切。</st> <st c="38750">我们使用 LangChain 创建的这个</st> *<st c="38755">链</st>* <st c="38760">代表了整个 RAG 管道的核心代码，而且它只有几个</st> `<st c="38863">字符串</st>` <st c="38863">那么长！</st>

<st c="38876">当用户使用您的应用程序时，它将从用户查询开始。</st> <st c="38949">但从编程的角度来看，我们设置了所有其他内容，以便我们可以正确处理查询。</st> <st c="39048">此时，我们已经准备好接受用户查询，所以让我们回顾一下我们代码中的最后一步。</st>

# `<st c="39145">提交 RAG 问题</st>`

`<st c="39175">到目前为止，你已经</st>` `<st c="39192">定义了链，但你还没有运行它。</st>` `<st c="39236">所以，让我们用你输入的查询运行整个 RAG 管道，一行代码即可：</st>` `<st c="39314"></st>`

```py
 rag_chain.invoke("What are the advantages of using RAG?")
```

如同在遍历链中发生的事情时提到的，`<st c="39383">"使用 RAG 的优势是什么?"</st>` `<st c="39446">是我们一开始要传递给链的字符串。</st>` `<st c="39485">链中的第一步期望这个字符串作为</st> *<st c="39606">问题</st>* `<st c="39614">我们在上一节讨论的作为两个期望变量之一。</st>` `<st c="39690">在某些应用中，这可能不是正确的格式，需要额外的函数来准备，但在这个应用中，它已经是我们期望的字符串格式，所以我们直接传递给那个</st>` `<st c="39905">RunnablePassThrough()</st>` `<st c="39926">对象。</st>`

`<st c="39934">将来，这个提示将包括来自用户界面的查询，但现在，我们将它表示为这个变量字符串。</st>` `<st c="40065">请记住，这不仅仅是 LLM 会看到的唯一文本；你之前添加了一个更健壮的提示，由</st>` `<st c="40169">prompt</st>` `<st c="40175">定义，并通过</st>` `<st c="40204">"context"</st>` `<st c="40213">和</st>` `<st c="40218">"</st>``<st c="40219">question"</st>` `<st c="40228">变量来填充。</st>`

`<st c="40239">这就是从编程角度的全部内容了！</st>` `<st c="40281">但当你运行代码时会发生什么呢？</st>` `<st c="40321">让我们回顾一下从这个 RAG</st>` `<st c="40374">管道代码中可以预期的输出。</st>`

# `<st c="40388">最终输出</st>`

`<st c="40401">最终的输出将看起来像这样：</st>` `<st c="40405"></st>` `<st c="40439"></st>`

```py
 "The advantages of using Retrieval Augmented Generation (RAG) include:\n\n1\. **Improved Accuracy and Relevance:** RAG enhances the accuracy and relevance of responses generated by large language models (LLMs) by fetching and incorporating specific information from databases or datasets in real time. This ensures outputs are based on both the model's pre-existing knowledge and the most current and relevant data provided.\n\n2\. **Customization and Flexibility:** RAG allows for the customization of responses based on domain-specific needs by integrating a company's internal databases into the model's response generation process. This level of customization is invaluable for creating personalized experiences and for applications requiring high specificity and detail.\n\n3\. **Expanding Model Knowledge Beyond Training Data:** RAG overcomes the limitations of LLMs, which are bound by the scope of their training data. By enabling models to access and utilize information not included in their initial training sets, RAG effectively expands the knowledge base of the model without the need for retraining. This makes LLMs more versatile and adaptable to new domains or rapidly evolving topics."
```

`<st c="41649">这包含了一些</st>` `<st c="41664">基本的格式化，所以当它显示时，它将看起来像这样（包括项目符号和</st>` `<st c="41762">粗体文本）：</st>`

`<st c="41775">使用检索增强生成（</st>``<st c="41832">RAG）的优势包括：</st>`

+   `<st c="41846">提高准确性和相关性：RAG 通过实时从数据库或数据集中检索并整合特定信息，增强了大型语言模型（LLM）生成的响应的准确性和相关性。</st>` `<st c="42067">这确保了输出基于模型预先存在的知识和最新且相关的</st>` `<st c="42175">数据。</st>`

+   `<st c="42189">定制和灵活性：通过将公司的内部数据库集成到模型的响应生成过程中，RAG 允许根据特定领域的需求定制响应。</st>` `<st c="42390">这种程度的定制对于创建个性化的体验以及需要高度特定性和详细的应用程序来说是无价的。</st>`

+   `<st c="42529">扩展模型知识超越训练数据：RAG 克服了 LLMs 的限制，LLMs 受限于其训练数据的范围。</st> <st c="42670">通过使模型能够访问和利用其初始训练集之外的信息，RAG 有效地扩展了模型的知识库，而无需重新训练。</st> <st c="42857">这使得 LLMs 更加灵活，能够适应新的领域或快速</st>` `<st c="42928">发展的主题。</st>`

<st c="42944">在你的用例中，你需要通过提出诸如，一个更便宜的模式能否以显著降低的成本完成足够好的工作等问题来做出决策？</st> <st c="43108">或者我需要额外花钱以获得更稳健的响应？</st> <st c="43176">你的提示可能要求非常简短，但你最终得到的响应与较便宜的模式一样短，那么为什么还要额外花钱呢？</st> <st c="43334">这在使用这些模型时是一个常见的考虑因素，在许多情况下，最大的、最昂贵的模型并不总是满足应用需求所必需的。</st>

<st c="43518">以下是 LLM 在结合之前 RAG 重点提示时将看到的内容：</st> <st c="43544">如下：</st> <st c="43598">（提示内容）</st>

```py
 "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Question:    What are the Advantages of using RAG? Context:    Can you imagine what you could do with all of the benefits mentioned above, but combined with all of the data within your company, about everything your company has ever done, about your customers and all of their interactions, or about all of your products and services combined with a knowledge of what a specific customer's needs are? You do not have to imagine it, that is what RAG does! Even smaller companies are not able to access much of their internal data resources very effectively. Larger companies are swimming in petabytes of data that are not readily accessible or are not being fully utilized. Before RAG, most of the services you saw that connected customers or employees with the data resources of the company were just scratching the surface of what is possible compared to if they could access ALL of the data in the company. With the advent of RAG and generative AI in general, corporations are on the precipice of something really, really big. Comparing RAG with Model Fine-Tuning#\nEstablished Large Language Models (LLM), what we call the foundation models, can be learned in two ways:\n Fine-tuning - With fine-tuning, you are adjusting the weights and/or biases that define the model\'s intelligence based
[TRUNCATED FOR BREVITY!]
Answer:"
```

<st c="45116">正如你所见，上下文相当大——它返回了原始文档中最相关的所有信息，以帮助 LLM 确定如何回答新问题。</st> <st c="45293">上下文</st> <st c="45304">是向量相似度搜索返回的内容，我们将在第八章中更深入地讨论这一点。</st>

# <st c="45415">完整代码</st>

<st c="45429">以下是代码的完整内容：</st> <st c="45450">如下：</st>

```py
 %pip install langchain_community
%pip install langchain_experimental
%pip install langchain-openai
%pip install langchainhub
%pip install chromadb
%pip install langchain
%pip install beautifulsoup4
```

<st c="45661">在运行以下代码之前，请重新启动内核：</st> <st c="45700">如下：</st>

```py
 import os
from langchain_community.document_loaders import WebBaseLoader
import bs4
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
os.environ['OPENAI_API_KEY'] = 'sk-###################'
openai.api_key = os.environ['OPENAI_API_KEY']
#### INDEXING ####
loader = WebBaseLoader(
    web_paths=("https://kbourne.github.io/chapter1.html",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                       class_=("post-content",
                               "post-title",
                               "post-header")
                   )
         ),
)
docs = loader.load()
text_splitter = SemanticChunker(OpenAIEmbeddings())
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
#### RETRIEVAL and GENERATION ####
prompt = hub.pull("jclemens24/rag-prompt")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
llm = ChatOpenAI(model_name="gpt-4o-mini")
rag_chain = (
    {"context": retriever | format_docs,
     "question": RunnablePassthrough()}
         | prompt
         | llm
         | StrOutputParser()
)
rag_chain.invoke("What are the Advantages of using RAG?")
```

# <st c="47070">摘要</st>

<st c="47078">本章提供了一个全面的代码实验室，介绍了完整 RAG 管道的实现过程。</st> <st c="47193">我们首先安装了必要的 Python 包，包括 LangChain、Chroma DB 以及各种 LangChain 扩展。</st> <st c="47313">然后，我们学习了如何设置 OpenAI API 密钥，使用</st> `<st c="47400">WebBaseLoader</st>`<st c="47413">从网页中加载文档，并使用 BeautifulSoup 预处理 HTML 内容以提取</st> <st c="47477">相关部分。</st>

<st c="47495">接下来，使用 LangChain 实验模块中的</st> `<st c="47563">SemanticChunker</st>` <st c="47578">将加载的文档分成可管理的块。</st> <st c="47617">然后，这些块被嵌入到 OpenAI 的嵌入模型中，并存储在 Chroma DB</st> <st c="47734">向量数据库中。</st>

<st c="47750">接下来，我们介绍了检索器的概念，它用于根据给定的查询在嵌入的文档上执行向量相似度搜索。</st> <st c="47901">我们逐步了解了 RAG 的检索和生成阶段，在这个案例中，它们通过 LCEL 结合成一个 LangChain 链。</st> <st c="48035">该链集成了来自 LangChain Hub 的预构建提示模板、选定的 LLM 以及用于格式化检索文档和解析</st> <st c="48192">LLM 输出的实用函数。</st>

<st c="48204">最后，我们学习了如何向 RAG 流水线提交问题，并接收一个包含检索上下文的生成响应。</st> <st c="48344">我们看到了 LLM 模型的输出，并讨论了基于准确性、深度和成本选择适当模型的关键考虑因素。</st>

<st c="48484">最后，RAG 流水线的完整代码已经提供！</st> <st c="48545">这就完了——你现在可以关闭这本书，仍然能够构建一个完整的 RAG 应用程序。</st> <st c="48643">祝你好运！</st> <st c="48654">但在你离开之前，还有许多概念需要复习，以便你能够优化你的 RAG 流水线。</st> <st c="48757">如果你在网上快速搜索</st> `<st c="48797">RAG 问题</st>` <st c="48813">或类似的内容，你可能会发现数百万个问题和问题被突出显示，其中 RAG 应用程序在除了最简单的应用程序之外的所有应用程序中都存在问题。</st> <st c="48982">还有许多其他 RAG 可以解决的问题需要调整刚刚提供的代码。</st> <st c="49086">本书的其余部分致力于帮助你建立知识，这将帮助你克服任何这些问题，并形成许多新的解决方案。</st> <st c="49234">如果你遇到类似的挑战，不要绝望！</st> <st c="49281">有一个解决方案！</st> <st c="49302">这可能会需要花费时间去超越</st> *<st c="49343">第二章</st>*<st c="49352">！</st>

<st c="49353">在下一章中，我们将讨论我们在</st> *<st c="49438">第一章</st>* <st c="49447">中讨论的一些实际应用，并深入探讨它们在各个组织中的实现方式。</st> <st c="49531">我们还将提供一些与 RAG 最常见实际应用之一相关的动手代码：提供 RAG 应用程序引用的内容来源</st> <st c="49709">给你。</st>
