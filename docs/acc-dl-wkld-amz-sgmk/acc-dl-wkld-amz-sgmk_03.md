

# 第三章：管理 SageMaker 开发环境

在前面的章节中，我们学习了 Amazon SageMaker 的基本组件和功能。到现在为止，你已经知道如何在 SageMaker 上构建和部署第一个简单的模型。然而，在许多更复杂的情况下，你需要在将代码部署到 SageMaker 管理的训练或托管集群之前，编写、分析和测试你的深度学习（DL）代码。能够在本地执行这一操作，同时模拟 SageMaker 运行时，将缩短开发周期，并避免为开发提供 SageMaker 资源所带来的不必要成本。

在本章中，我们将探讨如何组织你的开发环境，以便有效地为 SageMaker 开发和测试你的 DL 模型。本章包括选择 IDE 软件进行开发和测试的注意事项，以及如何在本地机器上模拟 SageMaker 运行时环境。我们还将概述用于管理 SageMaker 资源的可用 SDK 和 API。

以下主题将在后续章节中进行讲解：

+   为 SageMaker 选择开发环境

+   在本地调试 SageMaker 代码

阅读完本章后，你将能够根据特定的使用案例要求，设置与 SageMaker 兼容的高效开发环境。

# 技术要求

在本章中，你可以通过代码示例进行演练，从而培养实际技能。完整的代码示例可以在这里找到：[`github.com/PacktPublishing/Accelerate-Deep-Learning-Workloads-with-Amazon-SageMaker/blob/main/chapter3/`](https://github.com/PacktPublishing/Accelerate-Deep-Learning-Workloads-with-Amazon-SageMaker/blob/main/chapter3/)。为了跟随这些代码，你需要具备以下条件：

+   拥有 AWS 账户，并且是具有管理 Amazon SageMaker 资源权限的 IAM 用户。

+   在本地机器上安装 Docker 和 Docker Compose。如果你的开发环境中有 GPU 设备，你还需要安装 `nvidia-docker` ([`github.com/NVIDIA/nvidia-docker`](https://github.com/NVIDIA/nvidia-docker))。

+   安装 Conda ([`docs.conda.io/en/latest/`](https://docs.conda.io/en/latest/))。

# 为 SageMaker 选择开发环境

开发环境和 IDE 的选择通常由个人偏好或公司政策驱动。由于 SageMaker 是一个云平台，它不会限制你使用任何你选择的 IDE。你可以在本地机器或云机器（例如 Amazon EC2）上运行 IDE。SageMaker 还提供了一套 SDK 和软件包，用于模拟 SageMaker 运行时环境，这样你可以在将任何代码部署到云端之前，先在本地模拟环境中测试代码。

随着数据科学，特别是机器学习的进步，一种新的开发运行时环境应运而生——**交互式笔记本**，即 **Jupyter Notebooks** 和 **JupyterLab**（Jupyter Notebooks 的下一代，具有更多开发能力，如代码调试）。虽然它们并未完全取代传统 IDE，但笔记本因其能够探索和可视化数据、开发并与他人共享代码的功能而变得流行。

SageMaker 提供了几种托管的笔记本环境：

+   **SageMaker Studio** 服务 —— 一个专有的无服务器笔记本 IDE，用于机器学习开发

+   **SageMaker 笔记本实例** —— 一种托管的 Jupyter Notebook/JupyterLab 环境

三种选择——传统的 IDE、SageMaker 笔记本实例和 SageMaker Studio——都有各自的优点，并且在特定场景下可能是最优的选择。在接下来的部分，我们将详细审视这些 IDE 选项，并讨论它们在深度学习开发中的优缺点。

## 为 SageMaker 设置本地环境

在本地进行初步开发有许多好处，具体包括以下几点：

+   在本地进行开发时，您不会产生任何运行成本。

+   您可以选择自己喜欢的 IDE，从而提高开发周期的效率。

然而，本地开发运行时也有一定的限制。例如，您无法在不同的硬件设备上测试和分析代码。获取最新的为深度学习工作负载设计的 GPU 设备可能不切实际且成本高昂。因此，在许多情况下，您会使用 CPU 设备进行深度学习代码的初步开发和测试，以解决初期问题，然后在访问目标 GPU 设备的云实例上进行最终的代码分析和调优。

SageMaker 提供了多个 SDK，允许在本地环境与 AWS 云之间进行集成。让我们通过一个实际例子来演示如何配置本地环境以使用远程 SageMaker 资源。

### 配置 Python 环境

我们通过设置并配置一个与 AWS 集成的 Python 环境来开始配置。建议使用 Conda 环境管理软件来隔离您的 SageMaker 本地环境：

1.  您可以通过使用适当的安装方法（取决于您的本地操作系统）在本地机器上安装 Conda。安装完成后，您可以通过在终端窗口中运行以下命令来创建一个新的 Python 环境：

    ```py
    conda create -n sagemaker python=3.9
    ```

请注意，我们在此环境中明确指定使用的 Python 解释器版本。

1.  接下来，我们切换到创建环境并安装 AWS 和 SageMaker SDK：

    ```py
    conda activate sagemaker
    pip install boto3 awscli sagemaker
    ```

让我们回顾一下刚刚安装的 SDK：

+   `awscli` 是一个 AWS CLI 工具包，允许您以编程方式与任何 AWS 服务进行交互。它还提供了一个机制，用于在本地存储和使用 AWS 凭证。

+   `boto3` 是一个 Python SDK，用于管理你的 AWS 资源。它使用 AWS CLI 工具包建立的凭证，通过加密签名任何管理请求，从而在 AWS 中进行身份验证。

+   `sagemaker` – 你应该已经熟悉这个 Python SDK，因为在本书的前几章中，我们使用它与 SageMaker 资源进行交互，如训练作业或推理端点。与 `boto3` 不同，SageMaker SDK 抽象了许多底层资源管理的方面，通常建议在你需要编程管理 SageMaker 工作负载时使用。

1.  在继续之前，我们需要先配置 AWS 凭证。为此，你需要在终端中运行以下命令并提供你的 AWS 访问密钥和秘密密钥：

    ```py
    aws configure
    ```

你可以在这里阅读有关如何设置 AWS 凭证的详细信息：[`docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.xhtml`](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.xhtml)。

### 配置 Jupyter 环境

一旦我们配置好基本的 Python 环境并设置好 AWS 凭证，我们就可以启动 Jupyter 服务器了。在这个例子中，我们将使用 JupyterLab 环境。然而，你也可以根据自己的需求配置 IDE，因为许多 IDE（如 PyCharm 和 Visual Studio Code）都支持通过插件或原生方式使用 Jupyter Notebook。此方法的额外好处是，你可以在同一 IDE 中轻松地在笔记本和训练、推理脚本之间切换：

1.  要安装 JupyterLab 并创建内核，请在终端中运行以下命令：

    ```py
    conda install -c conda-forge jupyterlabpython -m ipykernel install --user --name sagemaker
    ```

1.  接下来，我们在本地机器上启动 JupyterLab 服务器：

    ```py
    jupyter lab
    ```

你的 JupyterLab 服务器现在应该可以通过 `http://localhost:8888` 访问。

### 在 SageMaker 上运行模型训练

在 JupyterLab 实例中，我们运行一些测试以确保能够从本地机器连接和管理 SageMaker 资源：

1.  本书 GitHub 仓库中的 `chapter3` 目录下包含完整的笔记本代码和训练脚本：

    ```py
    import sagemaker, boto3
    from sagemaker import get_execution_role
    session = sagemaker.Session()
    account = boto3.client('sts').get_caller_identity().get('Account')
    role = f"arn:aws:iam::{account}:role/service-role/AmazonSageMaker-ExecutionRole-<YOUR_ROLE_ID>" 
    ```

SageMaker 执行角色

请注意，你需要手动定义执行角色。对于 SageMaker 管理的环境，如 SageMaker Studio 或 SageMaker Notebook 实例，你可以使用 `get_execution_role()` 方法来获取执行角色。

1.  现在，我们可以像之前一样配置并启动 SageMaker 训练：

    ```py
    from sagemaker.pytorch import PyTorch
    import os
    pytorch_estimator = PyTorch(
                            session=session,
                            entry_point=f'{os.getcwd()}/sources/cifar10.py',
                            role=role,
                            instance_type="ml.m4.xlarge",
                            instance_count=1,
                            job_name="test",
                            framework_version="1.9.0",
                            py_version="py38",
                            hyperparameters={
                                "epochs": 1,
                                "batch-size": 16
                                }
                            )
    pytorch_estimator.fit()
    ```

1.  一旦训练作业完成，你可以查看本地的训练结果以及输出的工件存储位置：

    ```py
    pytorch_estimator.latest_training_job.describe()
    ```

如你所见，拥有本地开发环境为你提供了选择首选 IDE 的灵活性，同时避免了为 SageMaker 管理的开发环境付费。与此同时，它要求你仔细管理开发环境，这需要特定的专业知识和投入的努力。另一个潜在挑战是团队成员之间开发环境的同步。

### 使用 SageMaker Notebook 实例

SageMaker 笔记本实例由一个运行在 EC2 实例上的 AWS Jupyter 环境管理。你可以从基于 CPU 和 GPU 的实例列表中选择实例类型。SageMaker 提供了多个预配置的 Jupyter 内核，支持 Python 运行时。它包括预配置的运行时，支持 PyTorch、TensorFlow、MXNet 和其他流行的深度学习和机器学习框架。你还可以自定义现有内核（例如，安装新软件包）或使用 Conda 环境管理创建完全自定义的内核。

由于 Jupyter 环境直接运行在 EC2 实例上，你可以在本地训练或推理时直接观察资源消耗（例如，通过监控 `nvidia-smi` 工具输出）。你还可以执行 Docker 操作，如构建自定义容器并使用 SageMaker 本地模式进行测试，我们将在本章的 *本地调试 SageMaker 代码* 部分中详细讨论。

在某些场景下，使用笔记本实例可能会带来好处，例如以下情况：

+   你需要访问特定类型的硬件来测试和调试模型（例如，寻找最大训练吞吐量，而不遇到 OOM 问题）

+   在将模型部署到远程环境之前，你希望在本地为特定的超参数组合和硬件基准化模型性能

笔记本实例的一个缺点是缺乏灵活性。如果硬件需求发生变化，你无法快速更改实例类型。这可能导致在任务组合中有不同资源需求时产生不必要的成本。

假设你需要在本地预处理训练数据，并在这些数据上调试训练脚本。通常，数据处理是一个 CPU 密集型过程，不需要 GPU 设备。然而，训练深度学习模型则需要 GPU 设备。因此，你必须为任务中最高硬件需求的部分提供一个实例。或者，你需要在任务之间存储工作并重新配置笔记本实例。

SageMaker 在一个名为 SageMaker Studio 笔记本的较新产品中解决了这一弹性不足的问题。我们来详细回顾一下。

## 使用 SageMaker Studio

**SageMaker Studio**是一个基于 Web 的界面，允许你与各种 SageMaker 功能进行交互，从可视化数据探索、模型库和模型训练，到代码开发和端点监控。SageMaker Studio 旨在通过提供一个单一的工作和协作环境，简化和优化机器学习开发的所有步骤。

SageMaker Studio 中有多种功能。我们来回顾一下与深度学习开发相关的两项具体功能：

+   **Studio 笔记本**允许快速访问不同的计算实例和运行时，无需离开 JupyterLab 应用程序

+   **SageMaker JumpStart**是一个预构建的解决方案和模型库，允许你通过几次点击部署你的深度学习解决方案。

接下来，让我们讨论这些功能和使用案例。

### Studio 笔记本

Studio 笔记本提供了一个完全托管的 JupyterLab 环境，可以快速在不同的内核和计算实例之间切换。在切换过程中，你的工作会自动保存在共享文件系统中。共享文件系统具有高可用性，并根据需要无缝扩展。Studio 笔记本配备了一组预构建的内核，类似于笔记本实例，并且可以进一步自定义。你还可以为 Studio 笔记本创建一个完全自定义的内核镜像。

你可以从广泛的 EC2 实例、最新的 CPU 实例以及专用的 GPU 实例中选择用于训练和推理任务的计算实例。Studio 笔记本可以访问两种类型的 EC2 实例：

+   **快速实例**，可以在 2 分钟内完成切换。

+   **常规实例**，启动大约需要 5 分钟。请注意，这是一个大致的时间，可能会受到特定 AWS 区域资源可用性的影响。

协作功能

Studio 笔记本支持共享功能，允许你只需几次点击便能与团队成员共享代码、内核和实例配置。

SageMaker 笔记本内核在 Docker 镜像中运行。因此，存在若干限制：

+   你不能在 Studio 笔记本中构建或运行容器。

+   Studio 笔记本不支持在部署到 SageMaker 之前调试容器的本地模式。

+   AWS 提供了**Image Build CLI**来绕过这个限制，允许用户在使用 Studio 笔记本时构建自定义容器。

在大多数场景下，Studio 笔记本将是运行你自己的 JupyterLab 在 EC2 实例上或使用 SageMaker 笔记本实例的一个方便且具有成本效益的替代方案。然而，你应该留意之前提到的 Studio 笔记本的限制，并评估这些限制是否会影响你的特定用例或使用模式。此外，Studio 笔记本是 SageMaker Studio 平台的一部分，提供了更多的额外功能，如可视化数据探索和处理、可视化模型监控、预构建解决方案、用于管理特征存储、模型构建管道、端点、实验等的 UI 便捷功能。

### SageMaker JumpStart

SageMaker JumpStart 是一个预构建的端到端机器学习（ML）和深度学习（DL）解决方案库，提供可在 SageMaker 上一键部署的示例笔记本和模型。JumpStart 的解决方案和模型库庞大并持续增长。

**JumpStart 解决方案**专为特定行业用例设计，例如交易欺诈检测、文档理解和预测性维护。每个解决方案都包括多个集成组件，部署后可以立即供最终用户使用。请注意，您需要提供自己的数据集来训练 JumpStart 模型。

**JumpStart 模型**提供访问 SOTA 模型库。根据您的模型架构，您可以选择立即将该模型部署到推理、进行微调、从头训练或在自己的数据集上恢复增量训练。JumpStart 允许用户完全自定义用户操作，如定义训练集群的大小和实例类型、训练作业的超参数以及数据的位置。

模型库包括来自 TensorFlow Hub、PyTorch Hub 和 Hugging Face 的 CV 和 NLP 任务模型。

当您的业务问题可以通过使用通用解决方案和专有数据来解决时，SageMaker JumpStart 可以派上用场。JumpStart 还可以作为向 SageMaker 上的深度学习（DL）友好介绍，或者适合那些希望自己尝试深度学习的非技术用户。

在本节中，我们回顾了 SageMaker 可用的开发环境选项。所有三个选项都有其优缺点，具体选择主要取决于个人偏好和用例需求。通常，最好同时拥有本地环境和 SageMaker Studio 笔记本或笔记本实例。这种设置允许您在不支付任何云资源费用的情况下，本地开发、测试和进行初步调试。一旦您的代码在本地工作，您就可以轻松地在云硬件上运行相同的代码。Studio 笔记本尤其有用，因为它们允许您轻松切换不同的 CPU 和 GPU 运行时，而无需离开 Jupyter 笔记本，因此您可以实验训练配置（例如，调整批量大小或梯度累积）。

在下一节中，我们将重点介绍如何在将工作负载迁移到 SageMaker 云资源之前，高效地在本地调试 SageMaker 代码。

# 在本地调试 SageMaker 代码

为了简化本地代码开发和测试，SageMaker 支持 **本地模式**。此模式允许您在 SageMaker 容器中本地运行训练、推理或数据处理。当您希望在配置任何 SageMaker 资源之前先排查脚本问题时，这尤其有帮助。

所有 SageMaker 镜像以及自定义的 SageMaker 兼容镜像都支持本地模式。它作为 `sagemaker` Python SDK 的一部分实现。当您在本地模式下运行作业时，SageMaker SDK 会在后台创建一个包含作业参数的 Docker Compose YAML 文件，并在本地启动相关容器。配置 Docker 运行时环境的复杂性对用户进行了抽象。

本地模式支持 CPU 和 GPU 设备。你可以在本地模式下运行以下类型的 SageMaker 作业：

+   训练作业

+   实时端点

+   处理作业

+   批处理转换作业

### 本地模式的限制

在本地运行 SageMaker 作业时有一些限制：

+   仅支持一个本地端点。

+   不支持 GPU 的分布式本地训练。然而，你可以在 CPU 上运行分布式作业。

+   EFS 和 FSx for Lustre 不支持作为数据源。

+   不支持`Gzip`压缩、管道模式或输入的清单文件。

### 在本地模式下运行训练和推理

让我们在本地模式下训练一个简单的模型，然后将推理端点本地部署。完整的笔记本代码和训练脚本位于书籍仓库的`chapter3`目录：

1.  我们首先安装所有本地模式所需的依赖项：

    ```py
    pip install 'sagemaker[local]' –upgrade
    ```

1.  然后，我们配置 SageMaker 本地运行时。请注意，我们使用`LocalSession`类来让 SageMaker SDK 知道我们希望在本地配置资源：

    ```py
    import boto3
    from sagemaker.local import LocalSession
    sagemaker_local_session = LocalSession()
    sagemaker_local_session.config = {'local': {'local_code': True}}
    account = boto3.client('sts').get_caller_identity().get('Account')
    role = f"arn:aws:iam::{account}:role/service-role/AmazonSageMaker-ExecutionRole-<YOUR_ROLE_ID>" 
    ```

1.  在本笔记本中，我们打算使用来自 SageMaker ECR 仓库的公共 PyTorch 镜像。为此，我们需要存储凭证，以便 Docker 守护进程可以拉取镜像。在笔记本中运行以下命令（你也可以在终端窗口中运行，只需移除`!`）：

    ```py
    ! aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
    ```

1.  现在，我们需要决定是使用 GPU（如果可用）还是 CPU 设备（默认选择）。以下代码片段判断是否有可用的 CUDA 兼容设备（`"local_gpu"`值），如果没有，则默认使用 CPU 设备（`"local"`值）：

    ```py
    import subprocess
    instance_type = "local"
    try:
        if subprocess.call("nvidia-smi") == 0:
            instance_type = "local_gpu"
    except:
        print("GPU device with CUDA is not available")
    print("Instance type = " + instance_type)
    ```

1.  一旦我们定义了使用的本地设备，我们就可以配置并运行 SageMaker 训练作业：

    ```py
    from sagemaker.pytorch import PyTorch
    import os
    # Configure an MXNet Estimator (no training happens yet)
    pytorch_estimator = PyTorch(
                            session=sagemaker_local_session,
                            entry_point=f'{os.getcwd()}/sources/cifar10.py',
                            role=role,
                            instance_type=instance_type,
                            instance_count=1,
                            job_name="test",
                            framework_version="1.9.0",
                            py_version="py38",
                            hyperparameters={
                                "epochs": 1,
                                "batch-size": 16
                                }
                            )
    pytorch_estimator.fit()
    ```

1.  SageMaker Python SDK 会自动执行以下操作：

    +   从公共 ECR 仓库拉取适当的 PyTorch 镜像

    +   生成一个适当的`docker-compose.yml`文件，设置合适的卷挂载点以访问代码和训练数据

    +   使用`train`命令启动一个 Docker 容器

SageMaker 将输出 Docker Compose 命令及训练容器的 STDOUT/STDERR 到 Jupyter 单元格。

容器内调试代码

许多现代 IDE 支持调试在容器内运行的应用程序。例如，你可以在训练代码中设置断点。容器中的代码执行将停止，这样你就可以检查它是否正确执行。请查阅你的 IDE 文档，了解如何进行设置。

训练作业完成后，让我们看看如何将训练好的模型部署到本地实时端点。请注意，默认情况下，我们只训练单个 epoch，因此不要期望很好的结果！

1.  你可以通过在估算器上运行`deploy()`方法，将推理容器本地部署：

    ```py
    pytorch_estimator.deploy(initial_instance_count=1, instance_type=instance_type)
    ```

1.  一旦端点部署完成，SageMaker SDK 会开始将模型服务器的输出发送到 Jupyter 单元格。你也可以在 Docker 客户端 UI 中或通过`docker logs CONTAINER_ID`终端命令观察容器日志。

1.  我们现在可以发送一张测试图像，并观察推理脚本如何处理 Docker 日志中的推理请求：

    ```py
    import requests
    import json 
    payload = trainset[0][0].numpy().tobytes()
    url = 'http://127.0.0.1:8080/invocations'
    content_type = 'application/x-npy'
    accept_type = "application/json"
    headers = {'content-type': content_type, 'accept': accept_type}
    response = requests.post(url, data=payload, headers=headers)
    print(json.loads(response.content)[0])
    ```

在前面的代码块中，我们执行了以下操作：

+   构造推理有效载荷并将其序列化为`bytes`对象

+   构造了`content-type`和`accept-type` HTTP 头，指示推理服务器客户端发送的内容类型以及期望的内容类型

+   向本地 SageMaker 端点发送请求

+   读取响应输出

如果出现任何问题，您可以登录到运行中的推理容器中，检查运行时环境，或使用您的 IDE 功能设置调试会话。

# 总结

在本章中，我们回顾了一些可用的解决方案和最佳实践，讲解了如何为 Amazon SageMaker 组织深度学习代码的开发。根据您的使用案例需求和个人偏好，您可以选择在本地创建 DIY 环境，或使用 SageMaker 的笔记本环境之一——笔记本实例或 Studio 笔记本。您还学习了如何在本地测试 SageMaker 深度学习容器，以加速开发过程并避免额外的测试费用。

在下一章中，我们将重点介绍 SageMaker 的数据管理和数据处理。由于许多深度学习（DL）问题的训练数据集较大，并且需要进行预处理或后处理，因此理解最佳的存储解决方案至关重要。我们还将讨论使用 SageMaker 功能进行数据标注和数据处理的各个方面，以及访问训练数据的最佳实践。
