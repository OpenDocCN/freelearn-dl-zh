# 前言

从 2012 年 AlexNet 赢得大规模 ImageNet 竞赛，到 2018 年 BERT 预训练语言模型在多个**自然语言处理**（**NLP**）排行榜上名列前茅，现代**深度学习**（**DL**）在广泛的**人工智能**（**AI**）和**机器学习**（**ML**）社区中的革命持续进行。然而，将这些 DL 模型从离线实验转移到生产环境的挑战依然存在。这主要是因为缺乏一个统一的开源框架来支持 DL 完整生命周期的开发。本书将帮助你理解 DL 完整生命周期开发的全貌，并实现可从本地离线实验扩展到分布式环境和在线生产云的 DL 管道，重点是通过实践项目式学习，使用流行的开源 MLflow 框架支持 DL 过程的端到端实现。

本书从 DL 完整生命周期的概述和新兴的**机器学习运维**（**MLOps**）领域开始，提供了 DL 四大支柱（数据、模型、代码和可解释性）以及 MLflow 在这些领域中的作用的清晰图景。在第一章中，基于转移学习的基本 NLP 情感分析模型使用 PyTorch Lightning Flash 构建，并在接下来的章节中进一步开发、调优并部署到生产环境。从此开始，本书将一步步引导你理解 MLflow 实验的概念和使用模式，使用 MLflow 作为统一的框架来跟踪 DL 数据、代码与管道、模型、参数和指标的规模化管理。我们将在分布式执行环境中运行 DL 管道，确保可重现性和溯源追踪，并通过 Ray Tune、Optuna 和 HyperBand 对 DL 模型进行**超参数优化**（**HPO**）。我们还将构建一个多步骤 DL 推理管道，包括预处理和后处理步骤，使用 Ray Serve 和 AWS SageMaker 部署 DL 推理管道到生产环境，最后，提供一个使用**SHapley 加法解释**（**SHAP**）和 MLflow 集成的 DL 解释服务。

在本书的结尾，你将拥有从初始离线实验到最终部署和生产的深度学习（DL）管道构建的基础和实践经验，所有内容都在一个可重现和开源的框架中完成。在此过程中，你还将学习到与 DL 管道相关的独特挑战，以及我们如何通过实际且可扩展的解决方案克服这些挑战，例如使用多核 CPU、**图形处理单元**（**GPU**）、分布式和并行计算框架，以及云计算。

# 本书适合谁阅读

本书是为数据科学家、机器学习工程师和人工智能从业者编写的，旨在帮助他们掌握深度学习从构想到生产的完整生命周期，使用开源 MLflow 框架及相关工具，如 Ray Tune、SHAP 和 Ray Serve。本书中展示的可扩展、可复现且关注溯源的实现，确保您能够成功构建企业级深度学习管道。本书将为构建强大深度学习云应用程序的人员提供支持。

# 本书内容概述

*第一章*，*深度学习生命周期与 MLOps 挑战*，涵盖了深度学习完整生命周期的五个阶段，并使用迁移学习方法进行文本情感分类的第一个深度学习模型。它还定义了 MLOps 的概念，并介绍了其三个基础层次和四大支柱，以及 MLflow 在这些领域的作用。此外，还概述了深度学习在数据、模型、代码和可解释性方面面临的挑战。本章旨在将每个人带入相同的基础水平，并为本书其余部分的范围提供清晰的指导。

*第二章*，*使用 MLflow 开始深度学习之旅*，作为 MLflow 的入门教程和首个实践模块，快速设置基于本地文件系统的 MLflow 追踪服务器，或在 Databricks 上与远程管理的 MLflow 追踪服务器互动，并使用 MLflow 自动日志记录进行首个深度学习实验。本章还通过具体示例讲解了一些基础的 MLflow 概念，如实验、运行、实验和运行之间的元数据及其关系、代码追踪、模型日志记录和模型类型。特别地，我们强调实验应作为一等公民实体，能够弥合深度学习模型的离线与在线生产生命周期之间的差距。本章为 MLflow 的基础知识奠定了基础。

*第三章*，*跟踪模型、参数和指标*，涵盖了使用完整本地 MLflow 跟踪服务器的第一次深入学习模块。它从设置在 Docker Desktop 中运行的本地完整 MLflow 跟踪服务器开始，后端存储使用 MySQL，工件存储使用 MinIO。在实施跟踪之前，本章提供了基于开放源追踪模型词汇规范的开放源追踪框架，并提出了六种可以通过使用 MLflow 实现的溯源问题。接着，它提供了如何使用 MLflow 模型日志 API 和注册 API 来跟踪模型溯源、模型指标和参数的实践示例，无论是否启用自动日志。与其他典型的 MLflow API 教程不同，本章不仅仅提供使用 API 的指导，而是专注于我们如何成功地使用 MLflow 来回答溯源问题。在本章结束时，我们可以回答六个溯源问题中的四个，剩下的两个问题只能在拥有多步骤管道或部署到生产环境时回答，这些内容将在后续章节中涵盖。

*第四章*，*跟踪代码和数据版本管理*，涵盖了关于 MLflow 跟踪的第二个深入学习模块。它分析了在 ML/DL 项目中使用笔记本和管道的现有实践。它推荐使用 VS Code 笔记本，并展示了一个具体的深度学习笔记本示例，该示例可以在启用 MLflow 跟踪的情况下交互式或非交互式运行。它还建议使用 MLflow 的**MLproject**，通过 MLflow 的入口点和管道链实现多步骤的深度学习管道。为深度学习模型的训练和注册创建了一个三步深度学习管道。此外，它还展示了通过 MLflow 中的父子嵌套运行进行的管道级跟踪和单个步骤的跟踪。最后，它展示了如何使用 MLflow 跟踪公共和私有构建的 Python 库以及在**Delta Lake**中进行数据版本管理。

*第五章*，*在不同环境中运行深度学习管道*，涵盖了如何在不同环境中运行深度学习管道。首先介绍了在不同环境中执行深度学习管道的场景和要求。接着展示了如何使用 MLflow 的**命令行界面**（**CLI**）在四种场景中提交运行：本地运行本地代码、在 GitHub 上运行本地代码、在云端远程运行本地代码、以及在云端远程运行 GitHub 上的代码。MLflow 所支持的灵活性和可重现性在执行深度学习管道时，也为需要时的**持续集成/持续部署**（**CI/CD**）自动化提供了构建块。

*第六章*，*大规模超参数调优运行*，介绍了如何使用 MLflow 支持大规模的超参数优化（HPO），并利用最先进的 HPO 框架如 Ray Tune。首先回顾了深度学习流水线超参数的类型和挑战。然后，比对了三个 HPO 框架：Ray Tune、Optuna 和 HyperOpt，并对它们与 MLflow 的集成成熟度及优缺点进行了详细分析。接着，推荐并展示了如何使用 Ray Tune 与 MLflow 结合，对本书中迄今为止所讨论的深度学习模型进行超参数调优。此外，还介绍了如何切换到其他 HPO 搜索和调度算法，如 Optuna 和 HyperBand。这使得我们能够以一种具有成本效益且可扩展的方式，生产符合业务需求的高性能深度学习模型。

*第七章*，*多步骤深度学习推理流水线*，介绍了使用 MLflow 的自定义 Python 模型方法创建多步骤推理流水线的过程。首先概述了生产环境中四种推理工作流模式，在这些模式下，单一的训练模型通常不足以满足业务应用的需求，需要额外的预处理和后处理步骤。接着，提供了一个逐步指南，讲解如何实现一个多步骤推理流水线，该流水线将先前微调过的深度学习情感模型与语言检测、缓存以及额外的模型元数据结合起来。该推理流水线随后会作为一个通用的 MLflow **PyFunc** 模型进行日志记录，可以通过通用的 MLflow PyFunc 加载 API 加载。将推理流水线包装成 MLflow 模型为自动化和在同一 MLflow 框架内一致地管理模型流水线开辟了新天地。

*第八章*，*大规模部署深度学习推理流水线*，介绍了如何将深度学习推理流水线部署到不同的主机环境中以进行生产使用。首先概述了部署和托管环境的全景，包括大规模的批量推理和流式推理。接着，描述了不同的部署机制，如 MLflow 内置的模型服务工具、自定义部署插件以及像 Ray Serve 这样的通用模型服务框架。示例展示了如何使用 MLflow 的 Spark `mlflow-ray-serve` 部署批量推理流水线。接下来，提供了一个完整的逐步指南，讲解如何将深度学习推理流水线部署到托管的 AWS SageMaker 实例中用于生产环境。

*第九章*，*深度学习可解释性基础*，介绍了可解释性的基础概念，并探索了使用两个流行的可解释性工具。本章从概述可解释性的八个维度和**可解释的人工智能**（**XAI**）开始，然后提供了具体的学习示例，探索如何使用 SHAP 和 Transformers-interpret 工具箱进行 NLP 情感分析管道的应用。它强调，在开发深度学习应用时，可解释性应该被提升为一类重要的工件，因为在各类商业应用和领域中，对模型和数据解释的需求和期望正在不断增加。

*第十章*，*使用 MLflow 实现深度学习可解释性*，介绍了如何使用 MLflow 实现深度学习可解释性，并提供**解释即服务**（**EaaS**）。本章从概述 MLflow 当前支持的解释器和解释功能开始。具体来说，MLflow API 与 SHAP 的现有集成不支持大规模的深度学习可解释性。因此，本章提供了使用 MLflow 的工件日志记录 API 和**PyFunc** API 来实现的两种通用方法。文中提供了实现 SHAP 解释的示例，该解释将 SHAP 值以条形图的形式记录到 MLflow 跟踪服务器的工件存储中。SHAP 解释器可以作为 MLflow Python 模型进行日志记录，然后作为 Spark UDF 批处理解释或作为 Web 服务进行在线 EaaS 加载。这为在统一的 MLflow 框架内实现可解释性提供了最大的灵活性。

# 如何充分利用本书

本书中的大多数代码可以使用开源的 MLflow 工具进行实现和执行，少数情况需要 14 天的完整 Databricks 试用（可以在[`databricks.com/try-databricks`](https://databricks.com/try-databricks)注册）和一个 AWS 免费账户（可以在[`aws.amazon.com/free/`](https://aws.amazon.com/free/)注册）。以下列出了本书中涵盖的一些主要软件包：

+   MLflow 1.20.2 及以上版本

+   Python 3.8.10

+   Lightning-flash 0.5.0

+   Transformers 4.9.2

+   SHAP 0.40.0

+   PySpark 3.2.1

+   Ray[tune] 1.9.2

+   Optuna 2.10.0

本书中每一章的完整包依赖列在`requirements.txt`文件或本书 GitHub 仓库中的`conda.yaml`文件中。所有代码已在 macOS 或 Linux 环境下成功测试。如果你是 Microsoft Windows 用户，建议安装**WSL2**以运行本书中提供的 bash 脚本：[`www.windowscentral.com/how-install-wsl2-windows-10`](https://www.windowscentral.com/how-install-wsl2-windows-10)。已知问题是 MLflow CLI 在 Microsoft Windows 命令行中无法正常工作。

从本书的 *第三章**，* *模型、参数和指标的追踪* 开始，你还需要安装 Docker Desktop（[`www.docker.com/products/docker-desktop/`](https://www.docker.com/products/docker-desktop/)）来设置一个完整的本地 MLflow 跟踪服务器，以便执行本书中的代码。第八章*，* *在大规模上部署深度学习推理管道* 需要 AWS SageMaker 进行云部署示例。本书中使用的是 VS Code 版本 1.60 或更高版本 ([`code.visualstudio.com/updates/v1_60`](https://code.visualstudio.com/updates/v1_60))，作为 **集成开发环境** (**IDE**)。本书中的虚拟环境创建和激活使用的是 Miniconda 版本 4.10.3 或更高版本 ([`docs.conda.io/en/latest/miniconda.html`](https://docs.conda.io/en/latest/miniconda.html))。

**如果你使用的是本书的电子版本，建议你自己输入代码或从本书的 GitHub 仓库中获取代码（下一个部分会提供链接）。这样可以帮助你避免因复制粘贴代码而可能出现的错误。**

最后，为了充分利用本书的内容，你应该具有 Python 编程经验，并且对常用的机器学习和数据处理库（如 pandas 和 PySpark）有基本了解。

# 下载示例代码文件

你可以从 GitHub 下载本书的示例代码文件，链接为 [`github.com/PacktPublishing/Practical-Deep-Learning-at-Scale-with-MLFlow`](https://github.com/PacktPublishing/Practical-Deep-Learning-at-Scale-with-MLFlow)。如果代码有更新，它将在 GitHub 仓库中同步更新。

我们还提供了来自我们丰富图书和视频目录的其他代码包，详情请访问 [`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)。赶紧去看看吧！

# 下载彩色图像

我们还提供了 PDF 文件，包含本书中使用的屏幕截图和图表的彩色图像。你可以在这里下载：`static.packt-cdn.com/downloads/9781803241333_ColorImages.pdf`。

# 使用的约定

本书中使用了许多文本约定。

`Code in text`：表示文本中的代码词汇、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 账号。示例如下：“为了学习目的，我们在 GitHub 仓库的 `chapter08` 文件夹下提供了两个示例 `mlruns` 工件和 `huggingface` 缓存文件夹。”

代码块按如下方式书写：

```py
client = boto3.client('sagemaker-runtime') 
```

```py
response = client.invoke_endpoint(
```

```py
        EndpointName=app_name, 
```

```py
        ContentType=content_type,
```

```py
        Accept=accept,
```

```py
        Body=payload
```

```py
        )
```

当我们希望引起你对代码块中特定部分的注意时，相关行或项目将加粗显示：

```py
loaded_model = mlflow.pyfunc.spark_udf(
```

```py
    spark,
```

```py
    model_uri=logged_model, 
```

```py
    result_type=StringType())
```

任何命令行输入或输出都按如下方式书写：

```py
mlflow models serve -m models:/inference_pipeline_model/6
```

**粗体**：表示一个新术语、重要词汇或屏幕上看到的词语。例如，菜单或对话框中的词语以 **粗体** 显示。举个例子：“要执行这个单元格中的代码，你只需点击右上角下拉菜单中的 **Run Cell**。”

提示或重要注意事项

以这种形式出现。

# 联系我们

我们始终欢迎读者的反馈。

**常见反馈**：如果你对本书的任何内容有疑问，请通过电子邮件联系我们：customercare@packtpub.com，并在邮件主题中注明书名。

**勘误**：虽然我们已尽力确保内容的准确性，但错误难免发生。如果你在本书中发现错误，我们将不胜感激，如果你能将其报告给我们。请访问 www.packtpub.com/support/errata 并填写表单。

**盗版**：如果你在互联网上发现我们的作品以任何形式的非法复制，我们将非常感激你能提供该材料的地址或网站名称。请通过 copyright@packt.com 与我们联系，并提供链接。

**如果你有兴趣成为作者**：如果你在某个领域有专业知识，并且有兴趣写书或为书籍做贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com)。

# 分享你的想法

一旦你阅读了 *Practical Deep Learning at Scale with MLflow*，我们希望听到你的反馈！请点击这里，直接访问本书的亚马逊评论页面，并分享你的看法。

你的评论对我们和技术社区都非常重要，它将帮助我们确保提供优质的内容。
