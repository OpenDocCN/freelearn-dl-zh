# 3

# 数据工程

本章将更深入地探索 LLM Twin 项目。我们将学习如何设计和实现数据收集管道，以收集我们将用于所有 LLM 用例（如微调或推理）的原始数据。由于这不是一本关于数据工程的书籍，我们将使本章简短，并仅关注严格必要的收集所需原始数据的内容。从第四章开始，我们将专注于 LLM 和 GenAI，探讨其理论和具体实现细节。

当在玩具项目或进行研究时，你通常有一个静态数据集与之工作。但在我们的 LLM Twin 用例中，我们希望模拟一个真实世界的场景，我们必须自己收集和整理数据。因此，实现我们的数据管道将连接有关端到端 ML 项目如何工作的各个点。本章将探讨如何设计和实现一个**提取、转换、加载**（**ETL**）管道，该管道爬取多个社交平台（如 Medium、Substack 或 GitHub），并将收集到的数据聚合到一个 MongoDB 数据仓库中。我们将向您展示如何实现各种爬取方法、标准化数据并将其加载到数据仓库中。

我们将首先设计 LLM Twin 的数据收集管道，并解释 ETL 管道的架构。之后，我们将直接进入管道的实现，从 ZenML 开始，它将协调整个流程。我们将研究爬虫实现，并了解如何根据提供的链接的领域实现一个调度层，该层实例化正确的爬虫类，同时遵循软件最佳实践。接下来，我们将学习如何单独实现每个爬虫。此外，我们还将向您展示如何在 MongoDB 之上实现数据层，以结构化所有我们的文档并与数据库交互。

最后，我们将探讨如何使用 ZenML 运行数据收集管道，并从 MongoDB 查询收集到的数据。

因此，在本章中，我们将研究以下主题：

+   设计 LLM Twin 的数据收集管道

+   实现 LLM Twin 的数据收集管道

+   将原始数据收集到数据仓库中

到本章结束时，您将了解如何设计和实现一个 ETL 管道，以提取、转换和加载准备就绪的原始数据，以便被 ML 应用程序摄取。

# 设计 LLM Twin 的数据收集管道

在深入实施之前，我们必须了解 LLM Twin 的数据收集 ETL 架构，如图 3.1 所示。我们必须探索我们将爬取哪些平台以提取数据，以及我们将如何设计我们的数据结构和流程。然而，第一步是理解我们的数据收集管道如何映射到 ETL 流程。

ETL 管道涉及三个基本步骤：

1.  我们**提取**来自各种来源的数据。我们将从 Medium、Substack 和 GitHub 等平台爬取数据以收集原始数据。

1.  我们通过清理和标准化数据将其**转换**为适合存储和分析的统一格式。

1.  我们将**加载**转换后的数据到数据仓库或数据库中。

对于我们的项目，我们使用 MongoDB 作为我们的 NoSQL 数据仓库。尽管这不是一个标准的方法，但我们将很快解释选择这一方法的原因。

![图片](img/B31105_03_01.png)

图 3.1：LLM Twin 的数据收集 ETL 管道架构

我们希望设计一个 ETL 管道，它以用户和链接列表作为输入。然后，它单独爬取每个链接，标准化收集的内容，并将其保存到 MongoDB 数据仓库中特定作者的目录下。

因此，数据收集管道的签名将如下所示：

+   **输入：** 链接列表及其关联的用户（作者）

+   **输出：** 存储在 NoSQL 数据仓库中的原始文档列表

我们将“`user`”和“`author`”互换使用，因为在 ETL 管道的大多数场景中，用户是提取内容的作者。然而，在数据仓库内部，我们只有用户集合。

ETL 管道将检测每个链接的域，根据这个域它将调用一个专门的爬虫。我们为三种不同的数据类别实现了四个不同的爬虫，如*图 3.2*所示。首先，我们将探讨本书中我们将要工作的三个基本数据类别。我们收集的所有文档都可以归结为文章、仓库（或代码）和帖子。数据来源并不重要。我们主要对文档的格式感兴趣。在大多数情况下，我们都需要对这些数据类别进行不同的处理。因此，我们为每个类别创建了一个不同的域实体，每个实体在 MongoDB 中都有自己的类和集合。由于我们在文档的元数据中保存了源 URL，我们仍然知道它的来源，并且可以在我们的 GenAI 用例中引用它。

![图片](img/B31105_03_02.png)

图 3.2：爬虫和数据类别之间的关系

我们的基础代码支持四种不同的爬虫：

+   **Medium 爬虫**：用于从 Medium 收集数据。它输出一个文章文档。它登录 Medium 并爬取文章链接的 HTML。然后，它从 HTML 中提取、清理和标准化文本，并将标准化的文章文本加载到 NoSQL 数据仓库中。

+   **自定义文章爬虫**：它执行与 Medium 爬虫类似的步骤，但是一个更通用的实现，用于从各种网站收集文章。因此，由于它不实现任何特定平台的特性，它不执行登录步骤，而是盲目地从特定链接收集所有 HTML。这对于在线免费提供的文章来说足够了，您可以在 Substack 和人们的博客上找到这些文章。当链接的域名与其它支持的爬虫不关联时，我们将使用这个爬虫作为安全网。例如，当提供 Substack 链接时，它将默认使用自定义文章爬虫，但当提供 Medium URL 时，它将使用 Medium 爬虫。

+   **GitHub 爬虫**：此爬虫从 GitHub 收集数据。它输出一个仓库文档。它克隆仓库，解析仓库文件树，清理和标准化文件，并将它们加载到数据库中。

+   **LinkedIn 爬虫**：此爬虫用于从 LinkedIn 收集数据。它输出多个帖子文档。它登录 LinkedIn，导航到用户的动态，并爬取用户的所有最新帖子。对于每篇帖子，它提取其 HTML，清理和标准化它，并将其加载到 MongoDB 中。

在下一节中，我们将详细检查每个爬虫的实现。现在，请注意，每个爬虫以特定方式访问特定的平台或网站，并从中提取 HTML。之后，所有爬虫都会解析 HTML，从中提取文本，并清理和标准化它，以便可以在相同接口下存储在数据仓库中。

通过将所有收集到的数据减少到三个数据类别，而不是为每个新的数据源创建一个新的数据类别，我们可以轻松地将此架构扩展到多个数据源，而无需付出太多努力。例如，如果我们想开始从 X 收集数据，我们只需要实现一个新的爬虫，该爬虫输出一个帖子文档，这就足够了。其余的代码将保持不变。否则，如果我们引入了源维度到类和文档结构中，我们就需要在所有下游层中添加代码以支持任何新的数据源。例如，我们可能需要为每个新的源实现一个新的文档类，并调整特征管道以支持它。

对于我们的概念验证，爬取几百份文档就足够了，但如果我们想将其扩展到现实世界的产品，我们可能需要更多的数据源来爬取。LLMs 对数据有很高的需求。因此，为了获得理想的结果，您需要数千份文档，而不仅仅是几百份。但在许多项目中，实现一个不是最精确的端到端项目版本，并在之后迭代中对其进行改进是一个很好的策略。因此，通过使用这种架构，您可以在未来的迭代中轻松添加更多数据源以收集更大的数据集。关于 LLM 微调和数据集大小将在下一章中详细讨论。

**ETL 过程是如何与特征管道连接的？** 特征管道从 MongoDB 数据仓库中摄取原始数据，进一步清理它，将其处理成特征，并将其存储在 Qdrant 向量数据库中以使其可用于 LLM 训练和推理管道。**第四章**提供了有关特征管道的更多信息。ETL 过程独立于特征管道。两个管道严格通过 MongoDB 数据仓库相互通信。因此，数据收集管道可以为 MongoDB 写入数据，而特征管道可以独立地并且在不同的时间表上读取它。

**为什么我们选择 MongoDB 作为数据仓库？** 使用像 MongoDB 这样的事务型数据库作为数据仓库并不常见。然而，在我们的用例中，我们处理的数据量很小，MongoDB 可以轻松处理。即使我们计划在 MongoDB 集合上计算统计数据，它也能在我们的 LLM Twin 数据规模（数百个文档）上运行良好。我们选择 MongoDB 来存储原始数据的主要原因是因为我们非结构化数据的性质：从互联网爬取的文本。通过主要处理非结构化文本，选择一个不强制执行模式的 NoSQL 数据库使我们的开发更加容易和快速。此外，MongoDB 稳定且易于使用。他们的 Python SDK 直观。他们提供了一个开箱即用的 Docker 镜像，以及一个适合概念验证的云免费层，例如 LLM Twin。因此，我们可以在本地和云中自由地使用它。然而，当处理大数据（数百万个文档或更多）时，使用 Snowflake 或 BigQuery 等专用数据仓库将是理想的。

现在我们已经了解了 LLM Twin 数据收集管道的架构，让我们继续其实现。

## 实现 LLM Twin 的数据收集管道

如我们在**第二章**中所述，LLM Twin 项目的每个管道的入口点是 ZenML 管道，该管道可以通过 YAML 文件在运行时进行配置，并通过 ZenML 生态系统运行。因此，让我们首先了解一下 ZenML 的`digital_data_etl`管道。你会注意到，这是我们**第二章**中用作示例的相同管道，用于说明 ZenML。但这次，我们将更深入地探讨其实现，解释数据收集背后的工作原理。在了解管道的工作原理后，我们将探索用于从各个网站收集数据的每个爬虫的实现，以及用于存储和查询数据仓库数据的 MongoDB 文档。

## ZenML 管道和步骤

```py
 below, we can see the implementation of the ZenML digital_data_etl pipeline, which inputs the user’s full name and a list of links that will be crawled under that user (considered the author of the content extracted from those links). Within the function, we call two steps. In the first one, we look up the user in the database based on its full name. Then, we loop through all the links and crawl each independently. The pipeline’s implementation is available in our repository at pipelines/digital_data_etl.py.
```

```py
from zenml import pipeline
from steps.etl import crawl_links, get_or_create_user
@pipeline
def digital_data_etl(user_full_name: str, links: list[str]) -> str:
    user = get_or_create_user(user_full_name)
    last_step = crawl_links(user=user, links=links)
    return last_step.invocation_id 
```

**图 3.3**显示了在 ZenML 仪表板上运行的`digital_data_etl`管道。下一阶段是单独探索`get_or_create_user`和`crawl_links` ZenML 步骤。步骤实现可在我们的存储库`steps/etl`中找到。

![图片](img/B31105_03_03.png)

图 3.3：从 ZenML 仪表板运行的 digital_data_etl 管道示例

我们将从 `get_or_create_user` ZenML 步骤开始。我们首先导入在整个脚本中使用的必要模块和函数。

```py
from loguru import logger
from typing_extensions import Annotated
from zenml import get_step_context, step
from llm_engineering.application import utils
from llm_engineering.domain.documents import UserDocument 
```

接下来，我们定义函数的签名，它接受用户的全名作为输入，并检索现有的用户或（如果不存在）在 MongoDB 数据库中创建一个新的用户：

```py
@step
def get_or_create_user(user_full_name: str) -> Annotated[UserDocument, "user"]: 
```

使用一个实用函数，我们将全名拆分为名和姓。然后，我们尝试从数据库中检索用户或创建一个新的用户（如果不存在）。我们还检索当前的步骤上下文，并将有关用户的元数据添加到输出中，这将反映在 `user` ZenML 输出实体的元数据中：

```py
 logger.info(f"Getting or creating user: {user_full_name}")
    first_name, last_name = utils.split_user_full_name(user_full_name)
    user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)
    step_context = get_step_context()
    step_context.add_output_metadata(output_name="user", metadata=_get_metadata(user_full_name, user))
    return user 
```

此外，我们还定义了一个名为 `_get_metadata()` 的辅助函数，该函数构建一个包含查询参数和检索到的用户信息的字典，这些信息将被添加到用户实体的元数据中：

```py
def _get_metadata(user_full_name: str, user: UserDocument) -> dict:
    return {
        "query": {
            "user_full_name": user_full_name,
        },
        "retrieved": {
            "user_id": str(user.id),
            "first_name": user.first_name,
            "last_name": user.last_name,
        },
    } 
```

我们将进入 `crawl_links` ZenML 步骤，该步骤从提供的链接中收集数据。代码首先导入用于网络爬取的必要模块和库：

```py
from urllib.parse import urlparse
from loguru import logger
from tqdm import tqdm
from typing_extensions import Annotated
from zenml import get_step_context, step
from llm_engineering.application.crawlers.dispatcher import CrawlerDispatcher
from llm_engineering.domain.documents import UserDocument 
```

在导入之后，主函数输入由特定作者编写的链接列表。在这个函数中，初始化并配置了一个爬虫调度器，以处理特定的域名，如 LinkedIn、Medium 和 GitHub：

```py
@step
def crawl_links(user: UserDocument, links: list[str]) -> Annotated[list[str], "crawled_links"]:
    dispatcher = CrawlerDispatcher.build().register_linkedin().register_medium().register_github()
    logger.info(f"Starting to crawl {len(links)} link(s).") 
```

函数初始化变量以存储输出元数据和成功爬取的计数。然后，它遍历每个链接。它尝试爬取并提取每个链接的数据，更新成功爬取的计数并累积有关每个 URL 的元数据：

```py
 metadata = {}
    successfull_crawls = 0
    for link in tqdm(links):
        successfull_crawl, crawled_domain = _crawl_link(dispatcher, link, user)
        successfull_crawls += successfull_crawl
        metadata = _add_to_metadata(metadata, crawled_domain, successfull_crawl) 
```

在处理完所有链接后，该函数将累积的元数据附加到输出实体上：

```py
 step_context = get_step_context()
    step_context.add_output_metadata(output_name="crawled_links", metadata=metadata)
    logger.info(f"Successfully crawled {successfull_crawls} / {len(links)} 
links.")
    return links 
```

代码中包含一个辅助函数，该函数尝试根据链接的域名使用适当的爬虫从每个链接中提取信息。它处理在提取过程中可能发生的任何异常，并返回一个元组，指示爬取的成功与否以及链接的域名：

```py
def _crawl_link(dispatcher: CrawlerDispatcher, link: str, user: UserDocument) -> tuple[bool, str]:
    crawler = dispatcher.get_crawler(link)
    crawler_domain = urlparse(link).netloc
    try:
        crawler.extract(link=link, user=user)
        return (True, crawler_domain)
    except Exception as e:
        logger.error(f"An error occurred while crawling: {e!s}")
        return (False, crawler_domain) 
```

还提供了一个辅助函数来更新元数据字典，以包含每次爬取的结果：

```py
def _add_to_metadata(metadata: dict, domain: str, successfull_crawl: bool) -> dict:
    if domain not in metadata:
        metadata[domain] = {}
    metadata[domain]["successful"] = metadata.get(domain, {}).get("successful", 0) + successfull_crawl
    metadata[domain]["total"] = metadata.get(domain, {}).get("total", 0) + 1
    return metadata 
```

如上述 `_crawl_link()` 函数所示，`CrawlerDispatcher` 类根据每个链接的域名知道要初始化哪个爬虫。然后，逻辑被抽象到爬虫的 `extract()` 方法下。让我们深入探讨 `CrawlerDispatcher` 类，以全面了解其工作原理。

## 调度器：如何实例化正确的爬虫？

我们爬取逻辑的入口点是 `CrawlerDispatcher` 类。如图 3.4 所示，调度器充当提供链接和爬虫之间的中间层。它知道要将哪个爬虫与每个 URL 关联。

`CrawlerDispatcher`类知道如何提取每个链接的域名并初始化从该网站收集数据的适当爬虫。例如，当提供一个指向文章的链接并检测到[`medium.com`](https://medium.com)域名时，它将构建一个用于爬取该特定平台的`MediumCrawler`实例。考虑到这一点，让我们来探讨`CrawlerDispatcher`类的实现。

所有的爬虫逻辑都可以在 GitHub 仓库的`llm_engineering/application/crawlers`中找到。

![](img/B31105_03_04.png)

图 3.4：提供的链接、CrawlerDispatcher 和爬虫之间的关系

我们首先导入处理 URL 和正则表达式的必要 Python 模块，以及导入我们的爬虫类：

```py
import re
from urllib.parse import urlparse
from loguru import logger
from .base import BaseCrawler
from .custom_article import CustomArticleCrawler
from .github import GithubCrawler
from .linkedin import LinkedInCrawler
from .medium import MediumCrawler 
```

`CrawlerDispatcher`类被定义为根据给定的 URL 及其域名管理和调度适当的爬虫实例。其构造函数初始化一个注册表来存储已注册的爬虫。

```py
class CrawlerDispatcher:
    def __init__(self) -> None:
        self._crawlers = {} 
```

由于我们使用构建器创建模式来实例化和配置调度器，我们定义了一个`build()`类方法，该方法返回调度器的一个实例：

```py
 @classmethod
    def build(cls) -> "CrawlerDispatcher":
        dispatcher = cls()
        return dispatcher 
```

调度器包括用于注册特定平台（如 Medium、LinkedIn 和 GitHub）的爬虫的方法。这些方法在底层使用通用的`register()`方法将每个爬虫添加到注册表中。通过返回 self，我们遵循构建器创建模式（更多关于构建器模式：[`refactoring.guru/design-patterns/builder`](https://refactoring.guru/design-patterns/builder)）。在实例化调度器时，我们可以链式调用多个`register_*()`方法，如下所示：`CrawlerDispatcher.build().register_linkedin().register_medium()`。

```py
 def register_medium(self) -> "CrawlerDispatcher":
        self.register("https://medium.com", MediumCrawler)
        return self
    def register_linkedin(self) -> "CrawlerDispatcher":
        self.register("https://linkedin.com", LinkedInCrawler)
        return self
    def register_github(self) -> "CrawlerDispatcher":
        self.register("https://github.com", GithubCrawler)
        return self 
```

通用的`register()`方法将每个域名标准化，以确保在将其添加到调度器的`self._crawlers`注册表作为键之前其格式是一致的。这是一个关键步骤，因为我们将使用字典的键作为域名模式来匹配未来的链接与爬虫：

```py
 def register(self, domain: str, crawler: type[BaseCrawler]) -> None:
        parsed_domain = urlparse(domain)
        domain = parsed_domain.netloc
        self._crawlers[r"https://(www\.)?{}/*".format(re.escape(domain))] = crawler 
```

最后，`get_crawler()`方法通过将给定的 URL 与已注册的域名进行匹配来确定适当的爬虫。如果没有找到匹配项，它将记录一个警告并默认使用`CustomArticleCrawler`。

```py
 def get_crawler(self, url: str) -> BaseCrawler:
        for pattern, crawler in self._crawlers.items():
            if re.match(pattern, url):
                return crawler()
        else:
            logger.warning(f"No crawler found for {url}. Defaulting to CustomArticleCrawler.")
            return CustomArticleCrawler() 
```

理解数据收集管道的工作原理的下一步是逐个分析每个爬虫。

## 爬虫

在探索每个爬虫的实现之前，我们必须介绍它们的基类，该基类为所有爬虫定义了一个统一的接口。如图*3.4*所示，我们可以实现调度器层，因为每个爬虫遵循相同的签名。每个类都实现了`extract()`方法，这使得我们可以利用面向对象技术，如多态，在不了解其具体子类的情况下与抽象对象一起工作。例如，在 ZenML 步骤中的`_crawl_link()`函数中，我们有以下代码：

```py
crawler = dispatcher.get_crawler(link)
crawler.extract(link=link, user=user) 
```

注意我们是如何调用`extract()`方法而不关心我们实例化了哪种具体的爬虫类型。总结来说，使用抽象接口确保了核心的可重用性和扩展的便捷性。

### 基础类

现在，让我们探索`BaseCrawler`接口，它可以在[`github.com/PacktPublishing/LLM-Engineers-Handbook/blob/main/llm_engineering/application/crawlers/base.py`](https://github.com/PacktPublishing/LLM-Engineers-Handbook/blob/main/llm_engineering/application/crawlers/base.py)仓库中找到。

```py
from abc import ABC, abstractmethod
class BaseCrawler(ABC):
    model: type[NoSQLBaseDocument]
    @abstractmethod
    def extract(self, link: str, **kwargs) -> None: ... 
```

如上所述，接口定义了一个`extract()`方法，它接受一个链接作为输入。它还在类级别定义了一个模型属性，代表用于将提取的数据保存到 MongoDB 数据仓库的数据类别文档类型。这样做允许我们使用不同的数据类别自定义每个子类，同时保留类级别的相同属性。当我们深入研究文档实体时，我们将很快探索`NoSQLBaseDocument`类。

我们还通过`BaseSeleniumCrawler`类扩展了`BaseCrawler`类，该类实现了使用 Selenium 爬取各种网站的可重用功能，例如 Medium 或 LinkedIn。**Selenium**是一个自动化网页浏览器的工具。它用于以编程方式与网页交互（如登录 LinkedIn，浏览个人资料等）。

Selenium 可以以编程方式控制各种浏览器，如 Chrome、Firefox 或 Brave。对于这些特定平台，我们需要使用 Selenium 以编程方式操作浏览器以登录并滚动新闻源或文章，然后才能提取整个 HTML。对于其他网站，我们不需要经过登录步骤或可以直接加载整个页面，我们可以使用比 Selenium 更直接的方法从特定 URL 提取 HTML。

为了使基于 Selenium 的爬虫能够工作，您必须在您的机器上安装 Chrome（或如 Brave 这样的基于 Chromium 的浏览器）。

代码首先设置了使用 Selenium 和 ChromeDriver 初始化器进行网络爬取所需的必要导入和配置。`chromedriver_autoinstaller`确保安装了正确的 ChromeDriver 版本并将其添加到系统路径中，以保持与已安装的 Google Chrome 浏览器（或其他基于 Chromium 的浏览器）的兼容性。Selenium 将使用 ChromeDriver 与浏览器通信并打开一个无头会话，在那里我们可以通过编程方式操作浏览器以访问各种 URL，点击特定的元素，如按钮，或滚动新闻源。使用`chromedriver_autoinstaller`，我们确保始终安装了与我们的机器 Chrome 浏览器版本匹配的正确 ChromeDriver 版本。

```py
import time
from tempfile import mkdtemp
import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from llm_engineering.domain.documents import NoSQLBaseDocument
# Check if the current version of chromedriver exists
# and if it doesn't exist, download it automatically,
# then add chromedriver to path
chromedriver_autoinstaller.install() 
```

接下来，我们定义了`BaseSeleniumCrawler`类，用于需要使用 Selenium 来收集数据的用例，例如从 Medium 或 LinkedIn 收集数据。

其构造函数初始化各种 Chrome 选项以优化性能、增强安全性和确保无头浏览环境。这些选项禁用了如 GPU 渲染、扩展和通知等不必要的功能，这些功能可能会干扰自动化浏览。这些是在无头模式下爬取时的标准配置：

```py
class BaseSeleniumCrawler(BaseCrawler, ABC):
    def __init__(self, scroll_limit: int = 5) -> None:
        options = webdriver.ChromeOptions()

        options.add_argument("--no-sandbox")
        options.add_argument("--headless=new")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--ignore-certificate-errors")
        options.add_argument(f"--user-data-dir={mkdtemp()}")
        options.add_argument(f"--data-path={mkdtemp()}")
        options.add_argument(f"--disk-cache-dir={mkdtemp()}")
        options.add_argument("--remote-debugging-port=9226") 
```

在配置 Chrome 选项后，代码允许子类通过调用`set_extra_driver_options()`方法设置任何额外的驱动程序选项。然后，它初始化滚动限制并创建一个新的 Chrome 驱动程序实例，带有指定的选项：

```py
 self.set_extra_driver_options(options)
        self.scroll_limit = scroll_limit
        self.driver = webdriver.Chrome(
            options=options,
        ) 
```

`BaseSeleniumCrawler`类包括`set_extra_driver_options()`和`login()`的占位符方法，子类可以覆盖以提供特定功能。这确保了模块化，因为每个平台都有一个不同的登录页面，具有不同的 HTML 结构：

```py
 def set_extra_driver_options(self, options: Options) -> None:
        pass
    def login(self) -> None:
        pass 
```

最后，`scroll_page()`方法实现了一个滚动机制，用于导航到指定滚动限制的页面，例如 LinkedIn。它滚动到页面底部，等待新内容加载，并重复此过程，直到达到页面底部或超过滚动限制。此方法对于内容随用户滚动而出现的动态内容流至关重要：

```py
 def scroll_page(self) -> None:
        """Scroll through the LinkedIn page based on the scroll limit."""
        current_scroll = 0
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height or (self.scroll_limit and current_scroll >= self.scroll_limit):
                break
            last_height = new_height
            current_scroll += 1 
```

我们已经了解了我们爬虫的基类是什么样的。接下来，我们将探讨以下特定爬虫的实现：

+   `GitHubCrawler(BaseCrawler)`

+   `CustomArticleCrawler(BaseCrawler)`

+   `MediumCrawler(BaseSeleniumCrawler)`

    您可以在 GitHub 仓库中找到上述爬虫的实现，网址为[`github.com/PacktPublishing/LLM-Engineers-Handbook/tree/main/llm_engineering/application/crawlers`](https://github.com/PacktPublishing/LLM-Engineers-Handbook/tree/main/llm_engineering/application/crawlers)。

### GitHubCrawler 类

`GithubCrawler`类旨在抓取 GitHub 仓库，扩展了`BaseCrawler`的功能。我们不需要通过浏览器登录 GitHub，因为我们可以利用 Git 的克隆功能。因此，我们不需要利用任何 Selenium 功能。初始化时，它设置了一个要忽略的图案列表，以排除 GitHub 仓库中发现的标准文件和目录，如`.git`、`.toml`、`.lock`和`.png`，确保不必要的文件被排除在抓取过程之外：

```py
class GithubCrawler(BaseCrawler):
    model = RepositoryDocument
    def __init__(self, ignore=(".git", ".toml", ".lock", ".png")) -> None:
        super().__init__()
        self._ignore = ignore 
```

接下来，我们实现`extract()`方法，其中爬虫首先检查仓库是否已经被处理并存储在数据库中。如果存在，它将退出方法以防止存储重复项：

```py
def extract(self, link: str, **kwargs) -> None:
    old_model = self.model.find(link=link)
    if old_model is not None:
        logger.info(f"Repository already exists in the database: {link}")
        return 
```

如果仓库是新的，爬虫会从链接中提取仓库名称。然后，它创建一个临时目录来克隆仓库，以确保在处理完毕后从本地磁盘清理克隆的仓库：

```py
 logger.info(f"Starting scrapping GitHub repository: {link}")
    repo_name = link.rstrip("/").split("/")[-1]
    local_temp = tempfile.mkdtemp() 
```

在 try 块中，爬虫将当前工作目录更改为`temporary`目录，并在不同的进程中执行`git clone`命令：

```py
 try:
        os.chdir(local_temp)
        subprocess.run(["git", "clone", link]) 
```

成功克隆存储库后，爬虫构建了克隆存储库的路径。它初始化一个空字典，用于以标准化的方式聚合文件内容。它遍历目录树，跳过任何匹配忽略模式的目录或文件。对于每个相关文件，它读取内容，删除任何空格，并以文件路径作为键将其存储在字典中：

```py
 repo_path = os.path.join(local_temp, os.listdir(local_temp)[0])  # 
        tree = {}
        for root, _, files in os.walk(repo_path):
            dir = root.replace(repo_path, "").lstrip("/")
            if dir.startswith(self._ignore):
                continue
            for file in files:
                if file.endswith(self._ignore):
                    continue
                file_path = os.path.join(dir, file)
                with open(os.path.join(root, file), "r", errors="ignore") as f:
                    tree[file_path] = f.read().replace(" ", "") 
```

随后，它创建了一个新的`RepositoryDocument`模型实例，并用存储库内容、名称、链接、平台信息和作者详情填充。然后，这个实例被保存到 MongoDB 中：

```py
 user = kwargs["user"]
        instance = self.model(
            content=tree,
            name=repo_name,
            link=link,
            platform="github",
            author_id=user.id,
            author_full_name=user.full_name,
        )
        instance.save() 
```

最后，无论抓取是否成功或发生异常，爬虫都会确保临时目录被删除，以清理在过程中使用的任何资源：

```py
 except Exception:
        raise
    finally:
        shutil.rmtree(local_temp)
    logger.info(f"Finished scrapping GitHub repository: {link}") 
```

### CustomArticleCrawler 类

`CustomArticleCrawler`类以不同的方式从互联网收集数据。它利用`AsyncHtmlLoader`类从链接中读取整个 HTML，并使用`Html2TextTransformer`类从该 HTML 中提取文本。这两个类由`langchain_community` Python 包提供，如下所示，其中我们导入了所有必要的 Python 模块：

```py
from urllib.parse import urlparse
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers.html2text import Html2TextTransformer
from loguru import logger
from llm_engineering.domain.documents import ArticleDocument
from .base import BaseCrawler 
```

接下来，我们定义`CustomArticleCrawler`类，它继承自`BaseCrawler`。和之前一样，我们不需要登录或使用 Selenium 提供的滚动功能。在`extract`方法中，我们首先检查文章是否存在于数据库中，以避免内容重复：

```py
class CustomArticleCrawler(BaseCrawler):
    model = ArticleDocument
    def extract(self, link: str, **kwargs) -> None:
        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"Article already exists in the database: {link}")
            return 
```

如果文章不存在，我们继续抓取它。我们使用`AsyncHtmlLoader`类从提供的链接中加载 HTML。之后，我们使用`Html2TextTransformer`类将其转换为纯文本，该类返回一个文档列表。我们只对第一个文档感兴趣。由于我们将整个逻辑委托给这两个类，我们无法控制内容是如何提取和解析的。这就是为什么我们使用这个类作为没有自定义实现的域的回退系统。这两个类遵循 LangChain 范式，它提供了在大多数场景中表现良好的高级功能。它实现速度快，但定制困难。这也是许多开发者在生产用例中避免使用 LangChain 的原因之一：

```py
 logger.info(f"Starting scrapping article: {link}")
        loader = AsyncHtmlLoader([link])
        docs = loader.load()
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        doc_transformed = docs_transformed[0] 
```

我们从提取的文档中获取页面内容，以及相关的元数据，如`标题`、`副标题`、`内容`和`语言`：

```py
 content = {
            "Title": doc_transformed.metadata.get("title"),
            "Subtitle": doc_transformed.metadata.get("description"),
            "Content": doc_transformed.page_content,
            "language": doc_transformed.metadata.get("language"),
        } 
```

接下来，我们解析 URL 以确定文章是从哪个平台（或域名）抓取的：

```py
 parsed_url = urlparse(link)
        platform = parsed_url.netloc 
```

然后，我们创建一个文章模型的新实例，并用提取的内容填充它。最后，我们将这个实例保存到 MongoDB 数据仓库中：

```py
 user = kwargs["user"]
        instance = self.model(
            content=content,
            link=link,
            platform=platform,
            author_id=user.id,
            author_full_name=user.full_name,
        )
        instance.save()
        logger.info(f"Finished scrapping custom article: {link}") 
```

到目前为止，我们已经看到了如何使用 LangChain 实用函数抓取 GitHub 存储库和随机网站。最后，我们必须探索一个使用 Selenium 来编程操作浏览器的爬虫。因此，我们将继续`MediumCrawler`的实现。

### MediumCrawler 类

代码首先导入必要的库并定义`MediumCrawler`类，该类继承自`BaseSeleniumCrawler`：

```py
from bs4 import BeautifulSoup
from loguru import logger
from llm_engineering.domain.documents import ArticleDocument
from .base import BaseSeleniumCrawler
class MediumCrawler(BaseSeleniumCrawler):
    model = ArticleDocument 
```

在`MediumCrawler`类中，我们利用`set_extra_driver_options()`方法来扩展 Selenium 使用的默认驱动选项：

```py
 def set_extra_driver_options(self, options) -> None:
        options.add_argument(r"--profile-directory=Profile 2") 
```

`extract()`方法实现了核心功能，首先检查文章是否存在于数据库中，以防止重复条目。

如果文章是新的，该方法将继续导航到文章的链接并滚动页面以确保所有内容都已加载：

```py
 def extract(self, link: str, **kwargs) -> None:
        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"Article already exists in the database: {link}")
            return
        logger.info(f"Starting scrapping Medium article: {link}")
        self.driver.get(link)
        self.scroll_page() 
```

在页面完全加载后，该方法使用`BeautifulSoup`解析 HTML 内容并提取文章的标题、副标题和全文。`BeautifulSoup`是一个流行的 Python 库，用于网络爬取和解析 HTML 或 XML 文档。因此，我们使用它从 Selenium 访问的 HTML 中提取所有需要的 HTML 元素。最后，我们将所有内容聚合到一个字典中：

```py
 soup = BeautifulSoup(self.driver.page_source, "html.parser")
        title = soup.find_all("h1", class_="pw-post-title")
        subtitle = soup.find_all("h2", class_="pw-subtitle-paragraph")
        data = {
            "Title": title[0].string if title else None,
            "Subtitle": subtitle[0].string if subtitle else None,
            "Content": soup.get_text(),
        } 
```

最后，该方法关闭 WebDriver 以释放资源。然后，它创建一个新的`ArticleDocument`实例，用通过`kwargs`提供的提取内容和使用户信息填充它，并将其保存到数据库中：

```py
 self.driver.close()
        user = kwargs["user"]
        instance = self.model(
            platform="medium",
            content=data,
            link=link,
            author_id=user.id,
            author_full_name=user.full_name,
        )
        instance.save()
        logger.info(f"Successfully scraped and saved article: {link}") 
```

有了这些，我们就完成了`MediumCrawler`的实现。LinkedIn 爬虫遵循与 Medium 类似的模式，它使用 Selenium 登录并访问用户最新帖子的动态内容。然后，它提取帖子并滚动动态内容以加载下一页，直到达到限制。您可以在我们的仓库中查看完整的实现：[`github.com/PacktPublishing/LLM-Engineers-Handbook/blob/main/llm_engineering/application/crawlers/linkedin.py`](https://github.com/PacktPublishing/LLM-Engineers-Handbook/blob/main/llm_engineering/application/crawlers/linkedin.py)。

随着 LLMs 的兴起，从互联网收集数据已成为许多现实世界 AI 应用的关键步骤。因此，Python 生态系统出现了更多高级工具，例如 Scrapy ([`github.com/scrapy/scrapy`](https://github.com/scrapy/scrapy))，它爬取网站并从其页面中提取结构化数据，以及 Crawl4AI ([`github.com/unclecode/crawl4ai`](https://github.com/unclecode/crawl4ai))，它高度专业于为 LLMs 和 AI 应用爬取数据。

在本节中，我们探讨了实现三种类型的爬虫：一种利用子进程中的`git`可执行文件来克隆 GitHub 仓库的爬虫，一种使用 LangChain 工具提取单个网页的 HTML 的爬虫，以及一种利用 Selenium 在更复杂的场景中导航登录页面、滚动文章以加载整个 HTML 并将其提取为文本格式的爬虫。最后一步是理解我们本章中使用的文档类，如`ArticleDocument`，是如何工作的。

## NoSQL 数据仓库文档

我们不得不实现三个文档类来结构化我们的数据类别。这些类定义了我们需要的特定属性，例如内容、作者和来源链接。将数据结构化在类中而不是字典中是一种最佳实践，因为每个项目预期的属性更加详细，这可以减少运行错误。例如，当我们从一个 Python 字典中访问值时，我们永远无法确定它是否存在或其类型是否正确。通过将我们的数据项包装在类中，我们可以确保每个属性都符合预期。

通过利用 Python 包如 Pydantic，我们获得了开箱即用的类型验证，这确保了数据集的一致性。因此，我们将数据类别建模为以下文档类，这些类我们在代码中直到该点已经使用过：

+   `ArticleDocument` 类

+   `PostDocument` 类

+   `RepositoryDocument` 类

这些不仅仅是简单的 Python 数据类或 Pydantic 模型。它们支持在 MongoDB 数据仓库上执行读写操作。为了将读写功能注入到所有文档类中而不重复任何代码，我们使用了 **对象-文档映射**（ODM）软件模式，该模式基于 **对象关系映射**（ORM）模式。因此，让我们首先探索 ORM，然后转向 ODM，最后深入我们的自定义 ODM 实现和文档类。

### ORM 和 ODM 软件模式

在我们讨论软件模式之前，让我们看看什么是 ORM。它是一种技术，允许你使用面向对象的方法查询和操作数据库中的数据。而不是编写 SQL 或 API 特定的查询，你将所有复杂性封装在一个 ORM 类中，该类知道如何处理所有数据库操作，最常见的是 CRUD 操作。因此，使用 ORM 可以消除手动处理数据库操作的需要，并减少手动编写样板代码的需要。ORM 与 SQL 数据库（如 PostgreSQL 或 MySQL）交互。

大多数现代 Python 应用程序在与数据库交互时使用 ORM。尽管 SQL 在数据世界中仍然是一个流行的选择，但在 Python 后端组件中很少看到原始 SQL 查询。最流行的 Python ORM 是 SQLAlchemy ([`www.sqlalchemy.org/`](https://www.sqlalchemy.org/))。此外，随着 FastAPI 的兴起，SQLModel ([`github.com/fastapi/sqlmodel`](https://github.com/fastapi/sqlmodel)) 成为了一个常见选择，它是一个 SQLAlchemy 的包装器，使得与 FastAPI 的集成更加容易。

例如，使用 SQLAlchemy，我们定义了一个包含 ID 和名称字段的 `User` ORM。`User` ORM 映射到 SQL 数据库中的 `users` 表。因此，当我们创建一个新用户并将其提交到数据库时，它将自动保存到 `users` 表中。对 `User` 类的所有 CRUD 操作也是如此。

```py
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
   Base = declarative_base()
# Define a class that maps to the users table.
   class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
   name = Column(String) 
```

使用 `User` ORM，我们可以快速地从 Python 中直接插入或查询用户，而不需要写一行 SQL。请注意，ORM 通常支持所有 **CRUD** 操作。以下是一个代码片段，展示了如何将 User ORM 的一个实例保存到 SQLite 数据库中：

```py
engine = create_engine("sqlite:///:memory:")
Base.metadata.create_all(engine)
# Create a session used to interact with the database.
Session = sessionmaker(bind=engine)
session = Session()
# Add a new user.
new_user = User(name="Alice")
session.add(new_user)
session.commit() 
```

此外，这就是我们从 `users` SQLite 表中查询用户的方式：

```py
user = session.query(User).first()
if user:
print(f"User ID: {user.id}")
print(f"User name: {user.name}") 
```

在 GitHub 仓库 `code_snippets/03_orm.py` 中找到整个脚本及其运行方法。

ODM 模式与 ORM 非常相似，但它不是与 SQL 数据库和表一起工作，而是与 NoSQL 数据库（如 MongoDB）和非结构化集合一起工作。当我们与 NoSQL 数据库一起工作时，数据结构以集合为中心，这些集合存储类似于 JSON 的文档，而不是表中的行。

总结来说，ODM 简化了与基于文档的 NoSQL 数据库的工作，并将面向对象的代码映射到类似于 JSON 的文档。我们将在 MongoDB 之上实现一个轻量级的 ODM 模块，以完全理解 ODM 的工作原理。

### 实现 ODM 类

本节将探讨如何从头开始实现一个 ODM 类。这是一个很好的练习，可以学习 ODM 的工作原理并提高我们编写模块化和可重用 Python 类的技能。因此，我们将实现一个名为 `NoSQLBaseDocument` 的基础 ODM 类，其他所有文档都将从中继承以与 MongoDB 数据仓库交互。

该类可以在我们的仓库中找到，位于 `llm_engineering/domain/base/nosql.py`。

代码首先导入必要的模块并设置数据库连接。通过 `_database` 变量，我们连接到设置中指定的数据库，默认名为 `twin`：

```py
import uuid
from abc import ABC
from typing import Generic, Type, TypeVar
from loguru import logger
from pydantic import UUID4, BaseModel, Field
from pymongo import errors
from llm_engineering.domain.exceptions import ImproperlyConfigured
from llm_engineering.infrastructure.db.mongo import connection
from llm_engineering.settings import settings
_database = connection.get_database(settings.DATABASE_NAME) 
```

接下来，我们定义一个类型变量 `T`，它绑定到 `NoSQLBaseDocument` 类。该变量利用 Python 的泛型模块，使我们能够泛化类的类型。例如，当我们实现继承自 `NoSQLBaseDocument` 类的 `ArticleDocument` 类时，所有使用 `T` 的实例在分析函数签名时都将被替换为 `ArticleDocument` 类型（更多关于 Python 泛型的信息：[`realpython.com/python312-typing`](https://realpython.com/python312-typing))。

然后，`NoSQLBaseDocument` 类被声明为一个继承自 Pydantic 的 BaseModel、Python 的 Generic（提供前面描述的功能）和 `ABC`（使该类成为抽象类）类的抽象基类。这个类作为基础 ODM 类：

```py
T = TypeVar("T", bound="NoSQLBaseDocument")
class NoSQLBaseDocument(BaseModel, Generic[T], ABC): 
```

在 `NoSQLBaseDocument` 类中，定义了一个 id 字段，其类型为 UUID4，默认工厂生成一个唯一的 UUID。该类还实现了 `__eq__` 和 `__hash__` 方法，以便实例可以被比较，并可以在基于其唯一 `id` 属性的集合或字典键等散列集合中使用：

```py
id: UUID4 = Field(default_factory=uuid.uuid4)
def __eq__(self, value: object) -> bool:
    if not isinstance(value, self.__class__):
        return False
    return self.id == value.id
def __hash__(self) -> int:
    return hash(self.id) 
```

该类提供了在 MongoDB 文档和类实例之间进行转换的方法。`from_mongo()` 类方法将从 MongoDB 检索到的字典转换为类的实例。`to_mongo()` 实例方法将模型实例转换为适合 MongoDB 插入的字典：

```py
@classmethod
def from_mongo(cls: Type[T], data: dict) -> T:
    if not data:
        raise ValueError("Data is empty.")
    id = data.pop("_id")
    return cls(**dict(data, id=id))
def to_mongo(self: T, **kwargs) -> dict:
    exclude_unset = kwargs.pop("exclude_unset", False)
    by_alias = kwargs.pop("by_alias", True)
    parsed = self.model_dump(exclude_unset=exclude_unset, by_alias=by_alias, **kwargs)
    if "_id" not in parsed and "id" in parsed:
        parsed["_id"] = str(parsed.pop("id"))
    for key, value in parsed.items():
        if isinstance(value, uuid.UUID):
            parsed[key] = str(value)
    return parsed 
```

`save()` 方法允许将模型实例插入到 MongoDB 集合中。它检索适当的集合，利用上述 `to_mongo()` 方法将实例转换为 MongoDB 兼容的文档，并尝试将其插入到数据库中，处理可能发生的任何写入错误：

```py
def save(self: T, **kwargs) -> T | None:
    collection = _database[self.get_collection_name()]
    try:
        collection.insert_one(self.to_mongo(**kwargs))
        return self
    except errors.WriteError:
        logger.exception("Failed to insert document.")
        return None 
```

`get_or_create()` 类方法尝试在数据库中找到与提供的过滤器选项匹配的文档。如果找到匹配的文档，则将其转换为类的实例。如果没有找到，则创建一个新的实例，其初始数据为过滤器选项，并将其保存到数据库中：

```py
@classmethod
def get_or_create(cls: Type[T], **filter_options) -> T:
    collection = _database[cls.get_collection_name()]
    try:
        instance = collection.find_one(filter_options)
        if instance:
            return cls.from_mongo(instance)
        new_instance = cls(**filter_options)
        new_instance = new_instance.save()
        return new_instance
    except errors.OperationFailure:
        logger.exception(f"Failed to retrieve document with filter options: {filter_options}")
        raise 
```

`bulk_insert()` 类方法允许一次将多个文档插入到数据库中：

```py
@classmethod
def bulk_insert(cls: Type[T], documents: list[T], **kwargs) -> bool:
    collection = _database[cls.get_collection_name()]
    try:
        collection.insert_many([doc.to_mongo(**kwargs) for doc in documents])
        return True
    except (errors.WriteError, errors.BulkWriteError):
logger.error(f"Failed to insert documents of type {cls.__name__}")
        return False 
```

`find()` 类方法在数据库中搜索单个文档，该文档与给定的过滤器选项匹配：

```py
@classmethod
def find(cls: Type[T], **filter_options) -> T | None:
    collection = _database[cls.get_collection_name()]
    try:
        instance = collection.find_one(filter_options)
        if instance:
            return cls.from_mongo(instance)
        return None
    except errors.OperationFailure:
        logger.error("Failed to retrieve document.")
        return None 
```

类似地，`bulk_find()` 类方法检索与过滤器选项匹配的多个文档。它将每个检索到的 MongoDB 文档转换为模型实例，并将它们收集到一个列表中：

```py
@classmethod
def bulk_find(cls: Type[T], **filter_options) -> list[T]:
    collection = _database[cls.get_collection_name()]
    try:
        instances = collection.find(filter_options)
        return [document for instance in instances if (document := cls.from_mongo(instance)) is not None]
    except errors.OperationFailure:
        logger.error("Failed to retrieve document.")
        return [] 
```

最后，`get_collection_name()` 类方法确定与该类关联的 MongoDB 集合的名称。它期望该类有一个嵌套的 `Settings` 类，其中包含一个指定集合名称的 `name` 属性。如果缺少此配置，将引发一个 `ImproperlyConfigured` 异常，指定子类应定义一个嵌套的 `Settings` 类：

```py
@classmethod
def get_collection_name(cls: Type[T]) -> str:
    if not hasattr(cls, "Settings") or not hasattr(cls.Settings, "name"):
        raise ImproperlyConfigured(
            "Document should define an Settings configuration class with the name of the collection."
        )
    return cls.Settings.name 
```

我们可以使用嵌套的 `Settings` 类配置每个子类，例如定义集合名称，或任何特定于该子类的其他内容。在 Python 生态系统内，有一个基于 MongoDB 的 ODM 实现，称为 `mongoengine`，您可以在 GitHub 上找到它。它遵循与我们的类似但更全面的模式。我们自行实现了它，因为它是一个练习编写模块化和通用代码的绝佳机会，遵循最佳 OOP 原则，这对于实现生产级代码至关重要。

### 数据类别和用户文档类

最后一个拼图是查看从 `NoSQLBaseDocument` 基类继承的子类的实现。这些是定义我们的数据类别的具体类。您在处理爬虫类中的文章、存储库和帖子时已经看到了这些类的使用。

我们首先导入必要的 Python 模块和 ODM 基类：

```py
from abc import ABC
from typing import Optional
from pydantic import UUID4, Field
from .base import NoSQLBaseDocument
from .types import DataCategory 
```

我们定义一个 `enum` 类，其中我们集中所有我们的数据类别类型。这些变量将在本书中配置所有我们的 ODM 类时作为常量使用。

该类可以在 `llm_engineering/domain/types.py` 仓库中找到。

```py
from enum import StrEnum
class DataCategory(StrEnum):
    PROMPT = "prompt"
    QUERIES = "queries"
    INSTRUCT_DATASET_SAMPLES = "instruct_dataset_samples"
    INSTRUCT_DATASET = "instruct_dataset"
    PREFERENCE_DATASET_SAMPLES = "preference_dataset_samples"
    PREFERENCE_DATASET = "preference_dataset"
    POSTS = "posts"
    ARTICLES = "articles"
     REPOSITORIES = "repositories" 
```

`Document`类被引入作为`NoSQLBaseDocument` ODM 类之上的其他文档的抽象基模型。它包括常见的属性，如内容、平台和作者详情，为将继承它的文档提供了一个标准化的结构：

```py
class Document(NoSQLBaseDocument, ABC):
    content: dict
    platform: str
    author_id: UUID4 = Field(alias="author_id")
    author_full_name: str = Field(alias="author_full_name") 
```

最后，通过扩展`Document`类定义特定的文档类型。`RepositoryDocument`、`PostDocument`和`ArticleDocument`类代表不同的数据类别，每个类别都有独特的字段和设置，指定其在数据库中的相应集合名称：

```py
class RepositoryDocument(Document):
    name: str
    link: str
    class Settings:
        name = DataCategory.REPOSITORIES
class PostDocument(Document):
    image: Optional[str] = None
    link: str | None = None
    class Settings:
        name = DataCategory.POSTS
class ArticleDocument(Document):
    link: str
    class Settings:
        name = DataCategory.ARTICLES 
```

最后，我们定义了`UserDocument`类，用于存储和查询 LLM Twin 项目中的所有用户：

```py
class UserDocument(NoSQLBaseDocument):
    first_name: str
    last_name: str
    class Settings:
        name = "users"
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}" 
```

通过实现`NoSQLBaseDocument` ODM 类，我们必须专注于每个文档或域实体的字段和特定功能。所有 CRUD 功能都委派给了父类。此外，通过利用 Pydantic 来定义字段，我们获得了开箱即用的类型验证。例如，当创建`ArticleDocument`类的实例时，如果提供的链接是`None`或不是字符串，它将抛出一个错误，表示数据无效。

通过这样，我们已经完成了数据收集管道的实现，从 ZenML 组件开始，然后研究了爬虫的实现，最后用 ODM 类和数据类别文档结束了整个过程。最后一步是运行数据收集管道并将原始数据导入 MongoDB 数据仓库。

# 将原始数据收集到数据仓库中

ZenML 协调数据收集管道。因此，利用 ZenML，数据收集管道可以手动运行、计划或由特定事件触发。在这里，我们将向您展示如何手动运行它，而我们将讨论在*第十一章*中深入挖掘 MLOps 的其他场景。

我们为每位作者配置了不同的管道运行。我们为 Paul Iusztin 或 Maxime Labonne 的数据提供了 ZenML 配置文件。例如，要调用数据收集管道以收集 Maxime 的数据，您可以运行以下 CLI 命令：

```py
poetry poe run-digital-data-etl-maxime 
```

这将调用以下 ZenML YAML 配置文件中的管道：

```py
parameters:
  user_full_name: Maxime Labonne # [First Name(s)] [Last Name]
  links:
    # Personal Blog
    - https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31.html
    - https://mlabonne.github.io/blog/posts/2024-07-15_The_Rise_of_Agentic_Data_Generation.html
    # Substack
    - https://maximelabonne.substack.com/p/uncensor-any-llm-with-abliteration-d30148b7d43e
    - https://maximelabonne.substack.com/p/create-mixtures-of-experts-with-mergekit-11b318c99562
    - https://maximelabonne.substack.com/p/merge-large-language-models-with-mergekit-2118fb392b54
    … # More Substack links 
```

在前面的*图 3.3*中，我们看到了管道的运行 DAG 和 ZenML 仪表板中的详细信息。同时，*图 3.5*展示了由这个数据收集管道生成的`user`输出结果。您可以检查`user_full_name`查询和从 MongoDB 数据库检索到的`user`，这是我们在这个特定运行中收集的链接。

![img/B31105_03_05.png]

图 3.5：使用 Maxime 的配置文件运行数据收集管道后的用户输出结果示例

此外，在*图 3.6*中，您可以观察到`crawled_links`输出结果，其中列出了我们从哪些域名收集了数据，每个域抓取的链接总数，以及成功收集的链接数量。

我们再次强调这些工件的力量，因为它们跟踪每个管道的结果和元数据，使得单独监控和调试每个管道运行变得极其容易。

![图片](img/B31105_03_06.png)

图 3.6：使用 Maxime 的配置文件运行数据收集管道后，crawled_links 输出工件的示例

现在，我们可以通过运行以下代码在任何地方下载`crawled_links`工件，其中工件`ID`可以在 ZenML 中找到，并且对于每个工件版本都是唯一的：

```py
from zenml.client import Client
artifact = Client().get_artifact_version('8349ce09-0693-4e28-8fa2-20f82c76ddec')
loaded_artifact = artifact.load() 
```

例如，我们可以轻松运行相同的数据收集管道，但使用保罗·尤斯汀的 YAML 配置，如下所示：

```py
parameters:
  user_full_name: Paul Iusztin # [First Name(s)] [Last Name]
  links:
    # Medium
    - https://medium.com/decodingml/an-end-to-end-framework-for-production-ready-llm-systems-by-building-your-llm-twin-2cc6bb01141f
    - https://medium.com/decodingml/a-real-time-retrieval-system-for-rag-on-social-media-data-9cc01d50a2a0
    - https://medium.com/decodingml/sota-python-streaming-pipelines-for-fine-tuning-llms-and-rag-in-real-time-82eb07795b87
    … # More Medium links
    # Substack
    - https://decodingml.substack.com/p/real-time-feature-pipelines-with?r=1ttoeh
    - https://decodingml.substack.com/p/building-ml-systems-the-right-way?r=1ttoeh
    - https://decodingml.substack.com/p/reduce-your-pytorchs-code-latency?r=1ttoeh
    … # More Substack links 
```

要使用保罗的配置运行管道，我们调用以下`poe`命令：

```py
poetry poe run-digital-data-etl-paul 
```

在幕后，这调用以下 CLI 命令，该命令引用保罗的配置文件：

```py
poetry run python -m tools.run --run-etl --no-cache --etl-config-filename digital_data_etl_paul_iusztin.yaml 
```

您可以在`configs/`目录下的存储库中找到所有配置。此外，使用`poe`，我们配置了一个命令，用于调用所有受支持作者的 数据收集管道：

```py
poetry poe run-digital-data-etl 
```

我们可以轻松地使用我们的 ODM 类查询 MongoDB 数据仓库。例如，让我们查询为保罗·尤斯汀收集的所有文章：

```py
from llm_engineering.domain.documents import ArticleDocument, UserDocument
user = UserDocument.get_or_create(first_name="Paul", last_name="Iusztin")
articles = ArticleDocument.bulk_find(author_id=str(user.id))
print(f"User ID: {user.id}")
print(f"User name: {user.first_name} {user.last_name}")
print(f"Number of articles: {len(articles)}")
print("First article link:", articles[0].link) 
```

上述代码的输出是：

```py
User ID: 900fec95-d621-4315-84c6-52e5229e0b96
User name: Paul Iusztin
Number of articles: 50
First article link: https://medium.com/decodingml/an-end-to-end-framework-for-production-ready-llm-systems-by-building-your-llm-twin-2cc6bb01141f 
```

只需两行代码，我们就可以使用项目内定义的任何 ODM 查询和过滤我们的 MongoDB 数据仓库。

此外，为确保您的数据收集管道按预期工作，您可以使用您的**IDE 的 MongoDB 插件**搜索您的 MongoDB 集合，您必须单独安装此插件。例如，您可以使用此插件为 VSCode：[`www.mongodb.com/products/tools/vs-code`](https://www.mongodb.com/products/tools/vs-code)。对于其他 IDE，您可以使用类似的插件或外部 NoSQL 可视化工具。连接到 MongoDB 可视化工具后，您可以使用以下 URI 连接到我们的本地数据库：`mongodb://llm_engineering:llm_engineering@127.0.0.1:27017`。对于云 MongoDB 集群，您必须更改 URI，我们将在*第十一章*中探讨。

就这样，您已经学会了如何使用不同的 ZenML 配置运行数据收集管道，以及如何可视化每次运行的输出工件。我们还探讨了如何查询特定数据类别和作者的数据库仓库。因此，我们已经完成了数据工程章节，可以进入结论部分。

## 故障排除

存储在 MongoDB 数据库中的原始数据是所有后续步骤的核心。因此，如果您由于爬虫问题未能成功运行本章的代码，本节提供了修复潜在问题的解决方案，以便您继续前进。

### Selenium 问题

运行 Selenium 可能会由于浏览器驱动程序（如 `ChromeDriver`）的问题而导致问题，这是一个众所周知的问题。因此，如果使用 Selenium 的爬虫（如 `MediumCrawler`）由于您的 `ChromeDriver` 问题而失败，您可以通过注释掉添加到数据收集 YAML 配置中的 Medium 链接来轻松绕过这个问题。为此，请转到 `configs/` 目录，并找到所有以 `digital_data_etl_*` 开头的 YAML 文件，例如 `digital_data_etl_maxime_labonne.yaml`。打开它们，并注释掉所有与 Medium 相关的 URL，如图 3.7 所示。您可以省略 Substack 或个人博客 URL，因为这些使用的是 `CustomArticleCrawler`，它不依赖于 Selenium。

![图 3.7](img/B31105_03_07.png)

图 3.7：在爬取原始数据时修复 Selenium 问题

### 导入我们的备份数据

如果什么方法都不奏效，你可以尝试将备份的数据填充到 MongoDB 数据库中，这些数据存储在 `data/data_warehouse_raw_data` 目录下。这将允许你在不运行数据收集 ETL 代码的情况下，继续进行微调和推理部分。要导入此目录中的所有数据，请运行：

```py
poetry poe run-import-data-warehouse-from-json 
```

运行上述 CLI 命令后，你将拥有我们在开发代码时使用的数据集的一对一副本。为确保导入成功完成，你的 MongoDB 数据库中应有 88 篇文章和 3 个用户。

# 摘要

在本章中，我们学习了如何为 LLM Twin 用例设计和构建数据收集管道。我们收集了自定义数据来模拟现实世界的情况，而不是依赖于静态数据集，为我们在构建 AI 系统中面临的现实世界挑战做好准备。

首先，我们检查了 LLM Twin 数据收集管道的架构，该架构作为一个 ETL 流程运行。接下来，我们开始深入研究管道实现。我们首先了解如何使用 ZenML 来编排管道。然后，我们研究了爬虫实现。我们学习了三种爬取数据的方法：使用子进程中的 CLI 命令或使用 LangChain 或 Selenium 的实用函数来构建自定义逻辑，该逻辑可以编程方式操作浏览器。最后，我们研究了如何构建我们自己的 ODM 类，我们使用它来定义我们的文档类层次结构，其中包含文章、帖子、存储库等实体。

在本章末尾，我们学习了如何使用不同的 YAML 配置文件运行 ZenML 管道，并在仪表板中探索结果。我们还看到了如何通过 ODM 类与 MongoDB 数据仓库进行交互。

在下一章中，我们将介绍 RAG 特征管道的关键步骤，包括分块和嵌入文档，将这些文档摄入到向量数据库中，并应用预检索优化以提高性能。我们还将使用 Pulumi 以编程方式设置必要的基础设施，并通过部署 RAG 摄入管道到 AWS 来结束本章。

# 参考文献

+   Breuss, M. (2023, July 26). *Beautiful Soup：使用 Python 构建 Web 爬虫*. [`realpython.com/beautiful-soup-web-scraper-python/`](https://realpython.com/beautiful-soup-web-scraper-python/)

+   David, D. (2024, July 8). *2024 年使用 Selenium 进行网络爬取指南*. Bright Data. [`brightdata.com/blog/how-tos/using-selenium-for-web-scraping`](https://brightdata.com/blog/how-tos/using-selenium-for-web-scraping)

+   Hjelle, G. A. (2023, October 21). *Python 3.12 预览：静态类型改进*. [`realpython.com/python312-typing/`](https://realpython.com/python312-typing/)

+   *ORM 快速入门 — SQLAlchemy 2.0 文档*. (n.d.). [`docs.sqlalchemy.org/en/20/orm/quickstart.html`](https://docs.sqlalchemy.org/en/20/orm/quickstart.html)

+   Ramos, L. P. (2023, August 4). *Python 和 MongoDB：连接到 NoSQL 数据库*. [`realpython.com/introduction-to-mongodb-and-python/`](https://realpython.com/introduction-to-mongodb-and-python/)

+   Refactoring.Guru. (2024, January 1). *Builder*. [`refactoring.guru/design-patterns/builder`](https://refactoring.guru/design-patterns/builder)

+   *什么是 ETL？完整指南*. (n.d.). Qlik. [`www.qlik.com/us/etl`](https://www.qlik.com/us/etl)

# 加入我们书籍的 Discord 空间

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：

[`packt.link/llmeng`](https://packt.link/llmeng)

![](img/QR_Code79969828252392890.png)
