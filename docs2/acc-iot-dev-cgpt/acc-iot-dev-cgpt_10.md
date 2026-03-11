# 10

# 设置第一个项目的开发环境

在 *第八章* 中，我们介绍了 10 个适合初学者的物联网项目，并提供了 ChatGPT 提示示例。您可能会兴奋地选择一个项目并开始您的创新开发之旅。在 *第九章* 中，我们通过使用 AI 驱动的工具介绍了应用图生成方法。

在本章中，我们的目标是赋予您将创新物联网概念转化为现实所需的实用技能。学习如何使用 VS Code、PlatformIO IDE 和其他编码扩展设置开发环境，将为您提供有效编译和上传软件代码所需的工具。此外，在 PlatformIO 中创建您的第一个项目将提供实际操作经验，加深您对开发过程的理解。这些技能不仅将促进您当前的学习之旅，还将为未来的物联网项目做好准备，增强您将创意愿景变为现实的能力。

在本章中，我们将涵盖以下主题：

+   安装 **Visual Studio Code** (**VS Code**)

+   设置 PlatformIO IDE

+   安装其他编码辅助扩展

+   在 PlatformIO 下创建您的第一个项目

# 技术要求

本章将向您展示如何在 MacBook 上安装 VS Code、PlatformIO 和其他编码辅助扩展。为了获得最佳效果，请确保您的系统配备了运行 macOS Sonoma、版本 14.3.1 的 ARM CPU（即 Apple M1）。

# 安装 Visual Studio Code (VS Code)

**Visual Studio Code**，通常简称为 **VS Code**，是由微软开发的一个轻量级但功能强大的源代码编辑器。它提供了对 JavaScript、TypeScript 和 Node.js 的内置支持，以及丰富的扩展生态系统，支持其他语言，如 C++、C#、Python、PHP 等。此外，它还提供了调试、语法高亮、智能代码补全、代码片段、代码重构和嵌入式 Git 等功能。

在本章中，我们将指导如何在 macOS 上安装 VS Code。您可以从互联网上找到 Windows 和 Linux 的安装指南。

为了启动我们项目的设置，让我们看看以下流程并在 macOS 上安装 VS Code：

1.  首先，您需要通过点击 **下载** **Mac Universal** 从 [`code.visualstudio.com/`](https://code.visualstudio.com/) 下载 VS Code 软件。

![图 10.1 – 下载 VS Code](img/B22002_10_01.jpg)

图 10.1 – 下载 VS Code

1.  在系统下载目录中点击下载的 **Visual Studio Code** 软件。

![图 10.2 – 在下载中找到 VS Code](img/B22002_10_02.jpg)

图 10.2 – 在下载中找到 VS Code

1.  您现在将看到 **欢迎** 页面。从这里，您可以根据您的偏好选择一个主题。

![图 10.3 – VS Code 欢迎页面](img/B22002_10_03.jpg)

图 10.3 – VS Code 欢迎页面

1.  在左侧侧边栏中，找到 *扩展* 图标（如图 10.4 中箭头所示）。我们将从 *扩展* 部分安装 PlatformIO 和其他编码辅助工具。

![图 10.4 – VS Code 欢迎页面中的扩展](img/B22002_10_04.jpg)

图 10.4 – VS Code 欢迎页面中的扩展

您现在已完成了 VS Code 的初始安装。下一步是在 VS Code 上安装 PlatformIO IDE 扩展。

# 设置 PlatformIO IDE

一个开源的 **集成开发环境**（**IDE**），**PlatformIO** 兼容跨平台和跨架构。它支持超过 30 个嵌入式平台，具有多平台构建系统，包含众多库，并支持超过 800 个开源硬件板。它还作为 VS Code 中的强大扩展。在以下步骤中，我们将学习如何安装 PlatformIO：

1.  从上一节的 *步骤 4* 继续操作，点击 **扩展**。您将在左上角找到搜索窗口，如图 10.5 所示。

![图 10.5 – 扩展搜索窗口](img/B22002_10_05.jpg)

图 10.5 – 扩展搜索窗口

1.  在搜索窗口中输入 `Platformio` 以定位它。

![图 10.6 – 在搜索窗口中搜索“Platformio”](img/B22002_10_06.jpg)

图 10.6 – 在搜索窗口中搜索“Platformio”

1.  在 **PlatformIO IDE** 下点击 **安装** 以安装此扩展。

![图 10.7 – 选择并安装“PlatformIO IDE”](img/B22002_10_07.jpg)

图 10.7 – 选择并安装“PlatformIO IDE”

1.  安装后，您将看到如下所示的 **PlatformIO** 的 **欢迎** 页面：

![图 10.8 – PlatformIO IDE 欢迎页面](img/B22002_10_08.jpg)

图 10.8 – PlatformIO IDE 欢迎页面

1.  要首次启动 PlatformIO，点击类似 *蚂蚁头* 的图标。此操作将启动 PlatformIO 初始化过程。此过程包括安装 PlatformIO 核心和 Python 以及 *clang* 等其他所需软件包。请根据需要安装它们，并请注意这可能需要几分钟。

![图 10.9 – 安装后首次启动 PlatformIO](img/B22002_10_09.jpg)

图 10.9 – 安装后首次启动 PlatformIO

1.  如果之前在您的 MacBook 上尚未安装 *clang*，请通过点击 **安装** 允许其安装。

![图 10.10 – Clang 工具安装](img/B22002_10_10.jpg)

图 10.10 – Clang 工具安装

1.  在下一屏幕上继续安装过程，如图 10.11 所示。

![图 10.11 – PlatformIO 核心初始化过程](img/B22002_10_11.jpg)

图 10.11 – PlatformIO 核心初始化过程

1.  安装后点击 **立即重新加载**。

![图 10.12 – 初始化完成并重新加载 PlatformIO](img/B22002_10_12.jpg)

图 10.12 – 初始化完成并重新加载 PlatformIO

1.  重新加载后，您将看到如图 *10**.12* 所示的 PlatformIO 界面。继续点击**TERMINAL**，您将被引导到 PlatformIO 终端窗口。

![图 10.13 – 打开终端窗口](img/B22002_10_13.jpg)

图 10.13 – 打开终端窗口

1.  现在我们将安装开源软件包管理器。例如 Ubuntu 和 Debian 这样的 Linux 系统使用`/bin/bash -c “$(curl -``fsSL` [`raw.githubusercontent.com/Homebrew/install/HEAD/install.sh`](https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)`)”`。

![图 10.14 – 复制 Homebrew 安装链接](img/B22002_10_14.jpg)

图 10.14 – 复制 Homebrew 安装链接

1.  切换到 VS Code 窗口，将安装链接粘贴到**TERMINAL**命令行中，并按*Enter*键。

![图 10.15 – 在 PlatformIO 的终端窗口中粘贴 Homebrew 安装链接](img/B22002_10_15.jpg)

图 10.15 – 在 PlatformIO 的终端窗口中粘贴 Homebrew 安装链接

1.  可能会要求您输入 root 用户密码，这是您 MacBook 的登录密码。

![图 10.16 – 输入您的 root 用户密码](img/B22002_10_16.jpg)

图 10.16 – 输入您的 root 用户密码

1.  将开始 Homebrew 的安装过程。然后，在需要时按*RETURN*或*ENTER*键继续。

![图 10.17 – 在 PlatformIO 上开始 Homebrew 安装](img/B22002_10_17.jpg)

图 10.17 – 在 PlatformIO 上开始 Homebrew 安装

1.  完成 Homebrew 的安装过程后，按照指示将 Homebrew 添加到您的`PATH`中：

![图 10.18 – 将 Homebrew 添加到您的 MacOS 的 PATH 中](img/B22002_10_18.jpg)

图 10.18 – 将 Homebrew 添加到您的 MacOS 的 PATH 中

1.  复制、粘贴并运行第一个命令。

![图 10.19 – 执行第一个命令](img/B22002_10_19.jpg)

图 10.19 – 执行第一个命令

1.  然后复制、粘贴并运行第二个命令。

![图 10.20 – 执行第二个命令](img/B22002_10_20.jpg)

图 10.20 – 执行第二个命令

1.  在终端窗口中，使用`brew install platformio`安装 PlatformIO CLI（命令行）。

![图 10.21 – 使用 Homebrew 安装 PlatformIO CLI](img/B22002_10_21.jpg)

图 10.21 – 使用 Homebrew 安装 PlatformIO CLI

1.  PlatformIO CLI 安装完成后，重新启动 VS Code 以在终端窗口中使用 PlatformIO 命令行。

![图 10.22 – 通过 CLI 安装 PlatformIO 源](img/B22002_10_22.jpg)

图 10.22 – 通过 CLI 安装 PlatformIO 源

1.  重启后，通过 CLI 在终端窗口中使用`pio platform install espressif32`安装最新的稳定版 PlatformIO 包。

![图 10.23 – 安装 PlatformIO 最新包](img/B22002_10_23.jpg)

图 10.23 – 安装 PlatformIO 最新包

1.  现在的安装过程应该如以下图所示。

![图 10.24 – PlatformIO 包安装过程](img/B22002_10_24.jpg)

图 10.24 – PlatformIO 包安装过程

安装过程完成后，您需要重新启动 VS Code。恭喜！您已成功设置开发环境，并准备好开始您的第一个项目！

安装后，您可以使用终端窗口中的`pio system info`命令检查您的 PlatformIO 系统信息。

![图 10.25 – 检查 PlatformIO 包信息](img/B22002_10_25.jpg)

图 10.25 – 检查 PlatformIO 包信息

我们现在已经成功完成了 PlatformIO IDE 扩展在 Visual Studio Code 中的安装，包括其 CLI 和最新包。这个设置构成了我们编码环境的基础结构。

为了优化我们的开发工作流程和体验，考虑安装额外的扩展是至关重要的。这些扩展可以显著提高您的编码、故障排除和测试过程。

# 安装其他编码辅助扩展

VS Code 中有许多编码辅助工具作为扩展可用，如 Prettier、缩进彩虹和更好的注释。它们可以帮助使代码更容易阅读和调试，并增强您的编码体验：

+   **Prettier**：这是一个流行的代码格式化工具，支持多种语言，并且与 VS Code 集成良好。Prettier 会根据一组预定义的样式指南自动格式化您的代码。这有助于保持代码看起来整洁和一致，使其更容易阅读和维护。您可以选择在保存文件时自动格式化代码，或者手动运行它。此工具在团队项目中特别有用，以确保每个人都遵守相同的编码风格，从而减少差异并提高协作。

+   **缩进彩虹**：这是一个视觉工具，通过以渐变方式着色缩进级别，使缩进更易于阅读。每个缩进级别都有独特的颜色，有助于一眼区分代码的各个作用域和代码块。这在缩进在语言中起关键作用的语言中特别有用，例如 Python，或者在任何复杂的嵌套代码结构中，使代码的逻辑流程更容易跟踪。颜色和作为缩进级别的空格数是可定制的，允许用户根据他们的喜好和编码标准调整外观。

+   **更好的注释**：此扩展增强了代码中注释的可读性和功能性。它允许您对注释进行分类和彩色编码，使它们更加显眼和组织。例如，您可以区分信息性注释、问题、TODO 项、突出显示或警报，每个都有不同的颜色或格式。

# 在 PlatformIO 下创建您的第一个项目

在设置开发环境后，让我们来了解一下在 PlatformIO 中创建第一个项目的流程：

1.  点击左侧栏中的 PlatformIO 图标。

![图 10.26 – 在 VS Code 中启动 PlatformIO IDE](img/B22002_10_26.jpg)

图 10.26 – 在 VS Code 中启动 PlatformIO IDE

1.  在 **PIO 主页** 下点击 **打开**。

![图 10.27 – PlatformIO 欢迎页面](img/B22002_10_27.jpg)

图 10.27 – PlatformIO 欢迎页面

1.  在 **快速访问** 下点击 **新建项目**。

![图 10.28 – 创建新项目](img/B22002_10_28.jpg)

图 10.28 – 创建新项目

1.  你将看到如下所示的 **项目向导**。在 **名称** 字段中为项目命名。

![图 10.29 – 给项目命名](img/B22002_10_29.jpg)

图 10.29 – 给项目命名

1.  点击 **板** 下拉菜单以查找你的 ESP32 板型。

![图 10.30 – 查找硬件类型](img/B22002_10_30.jpg)

图 10.30 – 查找硬件类型

例如，如果你想使用 ESP32-C3，你可以输入 `esp32-c3`，然后你会找到你期望的板型。点击板型名称来选择它。

![图 10.31 – 选择你将要使用的硬件](img/B22002_10_31.jpg)

图 10.31 – 选择你将要使用的硬件

1.  在 **框架** 下拉菜单中，确保选择 **Arduino 框架**，然后点击 **完成**。

![图 10.32 – 在框架下拉菜单中选择“Arduino 框架”](img/B22002_10_32.jpg)

图 10.32 – 在框架下拉菜单中选择“Arduino 框架”

1.  然后 PlatformIO 开始自动配置你的项目，如下图所示。

![图 10.33 – 正在创建新项目](img/B22002_10_33.jpg)

图 10.33 – 正在创建新项目

1.  点击 **是，我信任作者** 以继续。

![图 10.34 – 选择“是”以信任作者](img/B22002_10_34.jpg)

图 10.34 – 选择“是”以信任作者

1.  现在你的第一个项目已经正确创建，如下图所示。点击左侧栏中的 **src**，你会看到 **main.cpp**，这是你编写 C++ 代码片段的地方。

![图 10.35 – 浏览 main.cpp 模板](img/B22002_10_35.jpg)

图 10.35 – 浏览 main.cpp 模板

到目前为止，所有必要的发展环境设置都应该已经正确配置。你应该能够在 PlatformIO IDE 中成功创建你的第一个物联网项目。

# 摘要

在本章中，你已经设置了你的开发环境，并准备好创建你的第一个物联网项目。现在你已经配置了 VS Code 和 PlatformIO IDE，你拥有了开始这段创新之旅所需的所有工具和知识。在接下来的章节中，我们将指导你完成第一个项目的实际实施，帮助你有效地应用所学知识。
