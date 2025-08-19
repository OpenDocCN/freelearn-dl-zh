# 第二章：为深度学习配置 R 环境

在本书中，我们将主要使用以下库进行深度学习：**H2O**、**MXNet**和**Keras**。我们还将专门使用**限制玻尔兹曼机**（**RBM**）包来处理 RBM 和**深度信念网络**（**DBN**）。此外，我们将在本书结尾时使用`ReinforcementLearning`包。

在本章中，我们将安装所有之前列出的包。每个包都可以用于在 R 中训练深度学习模型。然而，每个包都有其特定的优点和缺点。我们将探讨每个包的底层架构，这将帮助我们理解它们是如何执行代码的。除了`RBM`和`ReinforcementLearning`这两个包未使用 R 本地编写外，其它包都为 R 程序员提供了深度学习功能。这对于我们来说有重要的影响，从确保我们具备安装包所需的所有依赖开始。

本章将涵盖以下主要主题：

+   安装软件包

+   准备示例数据集

+   探索 Keras

+   探索 H2O

+   强化学习和 RBM

+   深度学习库对比

# 技术要求

你可以在[`github.com/PacktPublishing/Hands-on-Deep-Learning-with-R`](https://github.com/PacktPublishing/Hands-on-Deep-Learning-with-R)找到本章中使用的代码文件。

# 安装软件包

一些软件包可以直接从 CRAN 或 GitHub 安装，而`H2O`和`MXNet`则稍微复杂一些。我们将从最简单安装的包开始，然后转向那些更复杂的包。

# 安装 ReinforcementLearning

你可以通过使用`install.packages`来安装`ReinforcementLearning`，因为这个包有 CRAN 版本，使用以下代码行即可：

```py
install.packages("ReinforcementLearning")
```

# 安装 RBM

`RBM`包仅在 GitHub 上提供，不在 CRAN 上发布，因此其安装方式稍有不同。首先，如果你还没有安装`devtools`包，需要先安装它。接着，使用`devtools`包中的`install_github()`函数代替`install.packages`来安装`RBM`包，代码如下：

```py
install.packages("devtools")
library(devtools)
install_github("TimoMatzen/RBM")
```

# 安装 Keras

安装 Keras 的方式与我们安装`RBM`的方式类似，唯一的区别是稍微微妙但非常重要的。下载并安装包后，你需要运行`install_keras()`来完成安装。根据 Keras 的文档，如果你希望手动安装 Keras，则不需要调用`install_keras()`函数。

如果你选择这种方式，R 包将自动找到你已安装的版本。在本书中，我们将使用`install_keras()`来完成安装，代码如下：

```py
devtools::install_github("rstudio/keras")
library(keras)

install_keras()
```

如果你更倾向于安装 GPU 版本，只需在调用函数时做出如下更改：

```py
## for the gpu version :
install_keras(gpu=TRUE)
```

运行`install_keras()`将默认在虚拟环境中安装 Keras 和 TensorFlow，除了在 Windows 机器上——在写这本书时——此功能尚不支持，在这种情况下将使用`conda`环境，并且需要预先在 Windows 机器上安装 Anaconda。默认情况下，将安装 TensorFlow 的 CPU 版本以及最新版本的 Keras；可以添加一个可选参数来安装 GPU 版本，如前面的代码所示。对于本书，我们将接受默认值并运行`install_keras`。

如果你的机器上有多个版本的 Python，可能会遇到一些问题。如果你想使用的 Python 实例没有声明，R 将尝试通过先在常见位置查找，如`usr/bin`和`usr/local/bin`，来找到 Python。

使用 Keras 时，你可能希望指向 TensorFlow 虚拟环境中的 Python 实例。默认情况下，虚拟环境将命名为`r-tensorflow`。你可以使用 `reticulate` 包中的`use_python()`函数告诉 R 你想使用的 Python 版本。在函数内部，只需注明虚拟环境中 Python 实例的路径。在我的机器上，它如下所示：

```py
use_python('/Users/pawlus/.virtualenvs/r-tensorflow/bin/python') 
```

在你的机器上应该类似于这个样子。

一旦 R 找到正确的 Python 实例路径，接下来我们将在本章中介绍的代码应该可以正常运行。然而，如果没有引用正确的 Python 版本，你将遇到错误，代码将无法运行。

# 安装 H2O

对于 H2O，我们将使用 H2O 网站上的安装说明。通过这种方法，我们将首先搜索并移除任何先前安装的 H2O。接下来，安装 `RCurl` 和 `jsonlite`，然后从包含最新发布版本的 AWS S3 存储桶中安装 H2O。这个过程通过在获取软件包文件时简单地修改存储库位置来完成，默认情况下它是 CRAN 服务器。我们通过运行以下代码安装 H2O：

```py
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

pkgs <- c("RCurl","jsonlite")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}

install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/latest_stable_R")))
```

# 安装 MXNet

安装 MXNet 有多种方法。以下代码是设置 MXNet CPU 版本的最简单安装说明：

```py
cran <- getOption("repos")
cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
options(repos = cran)
install.packages("mxnet")
```

对于 GPU 支持，使用以下安装代码：

```py
  cran <- getOption("repos")
  cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/GPU/cu92"
  options(repos = cran)
  install.packages("mxnet")
```

使用 MXNet 需要 OpenCV 和 OpenBLAS。如果你需要安装这些，可以通过以下选项之一进行安装。

对于 macOS X，可以使用 Homebrew 来安装这些库：

1.  如果尚未安装 Homebrew，安装说明可以在[`brew.sh/`](https://brew.sh/)找到。

1.  如果你已经安装了 Homebrew，打开终端窗口并使用以下命令安装库：

```py
brew install opencv
brew install openblas
```

1.  最后，如此处所示，创建一个符号链接以确保使用的是最新版本的 OpenBLAS：

```py
ln -sf /usr/local/opt/openblas/lib/libopenblas.dylib /usr/local/opt/openblas/lib/libopenblasp-r0.3.1.dylib
```

对于 Windows，过程稍微复杂一些，因此本书中不会详细说明：

+   要安装 OpenCV，请按照[`docs.opencv.org/3.4.3/d3/d52/tutorial_windows_install.html`](https://docs.opencv.org/3.4.3/d3/d52/tutorial_windows_install.html)中提供的说明进行操作。

+   要安装 OpenBLAS，请按照[`github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio`](https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio)中提供的说明进行操作。

在安装了 OpenCV 和 OpenBLAS 后，前面的几行代码应该可以正常工作，以下载和安装 MXNet 包。不过，如果在尝试加载库时遇到错误，可能需要构建 MXNet 包并创建 R 包。完成这些操作的说明非常清晰且详细，但它们太长，无法包含在本书中：

+   **对于 macOS X**：[`mxnet.incubator.apache.org/versions/master/install/osx_setup.html`](https://mxnet.incubator.apache.org/versions/master/install/osx_setup.html)

+   **对于 Windows**：[`mxnet.incubator.apache.org/versions/master/install/windows_setup.html`](https://mxnet.incubator.apache.org/versions/master/install/windows_setup.html)

如果按照构建 MXNet 库和 R 绑定所需的步骤操作后，下载和安装包时仍然出现问题，这可能是由于多种可能的原因，很多问题已经有文档说明。不幸的是，尝试解决所有可能的安装场景和问题超出了本书的范围。不过，为了学习目的，你可以通过 Kaggle 网站上的内核使用 MXNet，那里可以使用 MXNet。

# 准备示例数据集

对于 Keras、H2O 和 MXNet，我们将使用成人普查数据集，该数据集使用美国普查数据来预测某人的年收入是否超过 50,000 美元。我们将在这里进行 Keras 和 MXNet 示例的数据准备，这样就不需要在两个示例中重复相同的代码：

1.  在以下代码中，我们将加载数据并标注这两个数据集，以便将它们合并：

```py
library(tidyverse)
library(caret)

train <- read.csv("adult_processed_train.csv")
train <- train %>% dplyr::mutate(dataset = "train")
test <- read.csv("adult_processed_test.csv")
test <- test %>% dplyr::mutate(dataset = "test")
```

运行上述代码后，我们的库已经加载并准备好使用。我们还加载了`train`和`test`数据集，现在可以在`Environment`面板中看到它们。

1.  接下来，我们将合并这些数据集，以便可以同时对所有数据进行一些更改。为了简化这些示例，我们将使用`complete.cases`函数来删除包含`NA`的行。我们还将删除字符项周围的空格，以便像`Male`和`Male `这样的项都被视为相同的项。让我们来看一下以下代码：

```py
all <- rbind(train,test)

all <- all[complete.cases(all),]

all <- all %>%
  mutate_if(~is.factor(.),~trimws(.))
```

1.  接下来，我们将在 `train` 数据集上执行一些额外的预处理步骤。首先，我们使用 `filter()` 函数从名为 `all` 的合并数据框中提取 `train` 数据。然后，我们将提取 `target` 列作为向量，并移除 `target` 和 `label` 列。我们通过以下代码来隔离 `train` 数据和 `train` 目标变量：

```py
train <- all %>% filter(dataset == "train")
train_target <- as.numeric(factor(train$target))
train <- train %>% select(-target, -dataset)
```

1.  现在，我们将分离数值列和字符列，以便对字符列进行编码，准备一个完全数值化的矩阵。我们通过以下代码分离数值列和字符列：

```py
train_chars <- train %>%
  select_if(is.character)

train_ints <- train %>%
  select_if(is.integer)
```

1.  接下来，我们将使用 `caret` 中的 `dummyVars()` 函数，将列中的字符值转换为单独的列，并通过将 `1` 分配给行来指示某个字符字符串是否存在。如果字符字符串不存在，那么该列在这一行中将包含 `0`。我们通过运行以下代码执行这一步骤：

```py
ohe <- caret::dummyVars(" ~ .", data = train_chars)
train_ohe <- data.frame(predict(ohe, newdata = train_chars))
```

1.  数据转换后，我们将使用以下代码将两个数据集重新合并：

```py
train <- cbind(train_ints,train_ohe)
```

1.  接下来，我们将通过运行以下代码对 `test` 数据集执行相同的步骤：

```py
test <- all %>% filter(dataset == "test")
test_target <- as.numeric(factor(test$target))
test <- test %>% select(-target, -dataset)

test_chars <- test %>%
  select_if(is.character)

test_ints <- test %>%
  select_if(is.integer)

ohe <- caret::dummyVars(" ~ .", data = test_chars)
test_ohe <- data.frame(predict(ohe, newdata = test_chars))

test <- cbind(test_ints,test_ohe)
```

1.  当我们创建目标向量时，它将因子值转换为 `1` 和 `2`。但是，我们希望将其转换为 `1` 和 `0`，因此我们将从向量中减去 `1`，如以下代码所示：

```py
train_target <- train_target-1
test_target <- test_target-1
```

1.  最后一步是清理 `train` 数据集中的一列，因为它在 `test` 数据集中不存在。我们通过运行以下代码来移除这一列：

```py
train <- train %>% select(-native.countryHoland.Netherlands)
```

现在我们已经加载并准备好了这个数据集，可以在下一步中使用它来展示一些初步示例，演示我们已安装的所有包。此时，我们的目标是查看语法，确保代码能够运行且库已正确安装。在后续章节中，我们将深入探讨每个包的详细内容。

# 探索 Keras

Keras 由 Francois Chollet 创建并维护。Keras 声称自己是为人类设计的，因此常见的用例非常简单，语法清晰易懂。Keras 可与多种低级深度学习语言配合使用，在本书中，Keras 将作为接口，帮助我们利用多个流行的深度学习后端，包括 TensorFlow。

# 可用函数

Keras 提供了对多种深度学习方法的支持，包括以下内容：

+   **循环神经网络** (**RNNs**)

+   **长短期记忆** (**LSTM**) 网络

+   **卷积神经网络** (**CNNs**)

+   **多层感知机** (**MLPs**)

+   **变量自编码器**

这并不是一个详尽无遗的列表，针对其他方法的支持也可以提供。然而，这些是本书后续章节中将涉及的内容。

# 一个 Keras 示例

在此示例中，我们将训练一个多层感知器，使用我们刚刚准备的成人普查数据集。此示例包含在内，以介绍该包的语法，并展示无需过多代码即可完成的基础练习：

如果系统中安装了多个版本的 Python，可能会出现问题。使用`reticulate`包和`use_python()`函数来定义您希望使用的 Python 实例的路径；例如，`use_python(usr/local/bin/python3)`。您也可以在`.Rprofile`文件中使用`RETICULATE_PYTHON`来设置 R 应该使用的 Python 实例路径。

1.  首先，我们将加载`tensorflow`和`keras`库，如下所示：

```py
library(tensorflow)
library(keras)
```

1.  接下来，我们将把数据集转换为矩阵，如下所示：

```py
train <- as.matrix(train)
test <- as.matrix(test)
```

1.  现在，我们可以创建一个顺序模型，该模型将按顺序通过每一层。我们将有一层，然后我们将编译结果。我们通过运行以下代码来定义我们的模型：

```py
model <- keras_model_sequential()

model %>%
  layer_dense(units=35, activation = 'relu')

model %>% keras::compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics='accuracy')
```

1.  在前一步中，我们定义了我们的模型，现在，在以下代码中，我们将把此模型拟合到我们的训练数据集：

```py
history <- model%>%
  fit(train, 
      train_target,
      epoch=10,
     batch=16,
      validation_split = 0.15)
```

1.  最后，我们可以通过将模型结果与`test`目标值进行比较来评估我们的模型。我们通过运行以下代码来评估模型性能：

```py
model%>%
  keras::evaluate(test,test_target)
```

这是`keras`的一般语法。如我们所示，它与管道操作兼容，并且具有对 R 程序员来说熟悉的语法。接下来，我们将查看一个使用 MXNet 包的例子。

# 探索 MXNet

MXNet 是由 Apache 软件基金会设计的深度学习库。它支持命令式编程和符号编程。通过将带有依赖关系的函数序列化，同时并行运行没有依赖关系的函数，它被设计成具有较高的速度。它支持 CPU 和 GPU 处理器。

# 可用函数

MXNet 提供了运行多种深度学习方法的手段，包括以下内容：

+   卷积神经网络（CNN）

+   循环神经网络（RNN）

+   生成对抗网络（GAN）

+   长短时记忆网络（LSTM）

+   自动编码器

+   受限玻尔兹曼机/深度玻尔兹曼网络（RBM/DBN）

+   强化学习

# 开始使用 MXNet

对于 MXNet，我们将使用相同的准备好的成人普查数据集。我们还将使用多层感知器作为我们的模型。如果您熟悉使用其他常见机器学习包来拟合模型，使用 MXNet 拟合模型将非常熟悉：

1.  首先，我们将使用以下代码行加载 MXNet 包：

```py
library(mxnet)
```

1.  然后，我们将定义我们的多层感知器。为了可重复性，我们设置了一个`seed`值。之后，训练数据将转换为数据矩阵，并作为参数传递给模型，同时传递训练目标值，如下所示：

```py
mx.set.seed(0)

model <- mx.mlp(data.matrix(train), train_target, hidden_node=10, out_node=2, out_activation="softmax",
                num.round=10, array.batch.size=20, learning.rate=0.05, momentum=0.8,
                eval.metric=mx.metric.accuracy)
```

1.  接下来，我们将通过将模型应用于`test`数据的矩阵版本来进行预测，如以下代码所示：

```py
preds = predict(model, data.matrix(test))
```

1.  然后，我们可以使用混淆矩阵来评估性能，将调整后的目标类别放在*y*-轴上，预测结果放在*x*-轴上，如下所示：

```py
pred.label = max.col(t(preds))-1
table(pred.label, test_target)
```

对于有 R 语言机器学习编程经验的人来说，MXNet 的语法应该看起来很熟悉。训练模型的函数接受描述性数据和目标数据，并且捕捉多个选项的值，就像使用 RandomForest 或 XGBoost 一样。

这些选项略有不同，我们将在后面的章节中介绍如何为这些参数赋值。然而，语法与其他 R 语言机器学习库的语法非常相似。接下来，我们将编写代码，使用 H2O 训练一个最小化的模型。

# 探索 H2O

H2O 的出现时间比 Keras 和 MXNet 更久，并且仍然被广泛使用。它利用 Java 和 MapReduce 内存压缩来处理大数据集。H2O 被用于许多机器学习任务，并且也支持深度学习。特别地，H2O 原生支持前馈人工神经网络（多层感知机）。H2O 还执行自动数据准备和缺失值处理。加载数据时需要使用一种特殊的数据类型：`H2OFrame`。

# 可用的函数

H2O 原生只支持前馈神经网络。与其他主要的深度学习包相比，这为该库带来了明显的限制。然而，这仍然是一个非常常见的深度学习实现方法。此外，H2O 允许将大对象存储在 H2O 集群的内存之外。由于这些原因，H2O 仍然是学习深度学习时值得了解的一个有价值的库。

# 一个 H2O 示例

在这个示例中，我们将再次使用成人人口普查数据集来预测收入。与我们的 Keras 示例一样，这个示例将保持极简，并且我们只会涵盖足够的内容来展示与 H2O 交互的语法，以及与其他包不同的设计细节：

1.  使用 H2O 时的第一个主要区别是，我们必须显式地初始化我们的 H2O 会话，这将生成一个 Java 虚拟机实例，并将其与 R 连接。这可以通过以下代码行来实现：

```py
# load H2O package
library(h2o)

# start H2O
h2o::h2o.init()
```

1.  加载数据以供 H2O 使用时，需要将数据转换为 `H2OFrame`。`H2OFrame` 与数据框非常相似，主要区别在于对象的存储位置。数据框存储在内存中，而 `H2OFrame` 存储在 H2O 集群中。对于非常大的数据集来说，这个特性可能是一个优势。在下面的示例中，我们将通过两步过程将数据转换为适当的格式。首先，我们按常规方式读取 `csv` 文件加载数据。然后，我们将数据框转换为 `H2OFrame`。我们使用以下代码将数据转换为适当的格式：

```py
## load data 
train <- read.csv("adult_processed_train.csv")
test <- read.csv("adult_processed_test.csv")

# load data on H2o
train <- as.h2o(train)
test <- as.h2o(test)
```

1.  对于这个例子，我们将执行一些插补作为唯一的预处理步骤。在此步骤中，我们将替换所有缺失值，对于数值数据我们使用`mean`，对于因子数据我们使用`mode`。在 H2O 中，设置`column = 0`将函数应用于整个数据框。值得注意的是，该函数是作用于数据的；然而，不需要将结果分配给新对象，因为插补结果会直接反映在作为函数参数传递的数据中。还值得强调的是，在 H2O 中，我们可以将一个向量传递给方法参数，它将在此情况下用于每个变量，首先检查是否可以使用第一种方法，如果不行，则切换到第二种方法。通过运行以下代码行可以完成数据的预处理：

```py
## pre-process
h2o.impute(train, column = 0, method = c("mean", "mode"))
h2o.impute(test, column = 0, method = c("mean", "mode"))
```

1.  此外，在此步骤中，我们将定义`dependent`和`independent`变量。`dependent`变量存储在`target`列中，而所有其余列包含`independent`变量，这些变量将在该任务中用于预测`target`变量：

```py
#set dependent and independent variables
target <- "target"
predictors <- colnames(train)[1:14]
```

1.  在所有准备步骤完成后，我们现在可以创建一个最小模型。H2O 的`deeplearning`函数将创建一个前馈人工神经网络。在这个例子中，只包含运行模型所需的最小内容。然而，该函数可以接受 80 到 90 个参数，我们将在后面的章节中介绍这些参数。以下代码中，我们为模型提供一个名称，识别训练数据，通过复制模型中涉及的伪随机数设置种子以确保可重复性，定义`dependent`和`independent`变量，并指定模型应运行的次数以及每轮数据的切分方式：

```py
#train the model - without hidden layer
model <- h2o.deeplearning(model_id = "h2o_dl_example"
                          ,training_frame = train
                          ,seed = 321
                          ,y = target
                          ,x = predictors
                          ,epochs = 10
                          ,nfolds = 5)
```

1.  运行模型后，可以使用以下代码行在外部折叠样本上评估性能：

```py
h2o.performance(model, xval = TRUE)
```

1.  最后，当我们的模型完成时，集群必须像初始化时那样显式关闭。以下函数将关闭当前的`h2o`实例：

```py
h2o::h2o.shutdown()
```

我们可以在这个例子中观察到以下几点：

+   H2O 的语法与其他机器学习库有很大不同。

+   首先，我们需要初始化 Java 虚拟机，并且我们需要将数据存储在该包的特殊数据容器中。

+   此外，我们可以看到，通过在数据对象上运行函数而不将更改分配回对象，插补会发生。

+   我们可以看到，我们还需要包括所有自变量的列名，这与其他模型略有不同。

+   所有这些都是在说明，使用 H2O 时它可能会显得有些不熟悉。它在算法的可用性方面也有限。但它能够处理更大的数据集，这一点是该包的明显优势。

现在我们已经了解了全面的深度学习包，我们将重点关注使用 R 编写的，执行特定建模任务或一组有限任务的包。

# 探索 ReinforcementLearning 和 RBM

`ReinforcementLearning`和`RBM`包与之前介绍的库有两个重要的不同之处：首先，它们是专门化的包，仅包含单一深度学习任务的函数，而不是试图支持各种深度学习选项；其次，它们完全用 R 语言编写，没有额外的语言依赖。这是一个优点，因为之前的库的复杂性意味着，当包外部发生变化时，包可能会崩溃。对于这些库的支持页面，充满了安装常见问题和故障排除的示例，以及一些包突然停止工作或被弃用的案例。在这些情况下，我们鼓励你继续在 CRAN 和其他网站上寻找解决方案，因为 R 社区以其动态发展和强大的支持闻名。

# 强化学习示例

在这个示例中，我们将创建一个强化学习的样本环境。强化学习的概念将在后续章节中更详细地探讨。在这个示例中，我们将生成一系列状态和动作，以及采取这些动作的奖励，即，采取行动是否导致了期望的结果或负面后果。然后，我们将定义我们的代理如何响应或从这些动作中学习。一旦所有这些内容都被定义，我们将运行程序，代理将通过环境进行学习并解决任务。我们将通过运行以下代码定义并执行一个最小的强化学习示例：

```py
library(ReinforcementLearning)

data <- sampleGridSequence(N = 1000)

control <- list(alpha = 0.1, gamma = 0.1, epsilon = 0.1)

model <- ReinforcementLearning(data, s = "State", a = "Action", r = "Reward", s_new = "NextState", control = control)

print(model)
```

在这个示例中，我们可以看到以下内容：

+   语法非常熟悉，类似于我们可能使用的许多其他 R 包。此外，我们还可以看到，我们可以使用极少的代码完成一个简单的强化学习任务。

+   在该包的 GitHub 代码库中，所有函数都是用 R 语言编写的，这为探索问题发生的可能原因提供了便利。如果出现问题，也能够减少对更复杂包中其他语言的依赖，这一点较为重要。

# 一个 RBM 示例

这里是一个使用`RBM`包的简单示例。RBM 模型可以通过`MXNet`库创建。然而，我们在本书中包含这个包，是为了说明何时使用`MXNet`训练`RBM`模型最为合适，以及何时独立实现算法可能会更合适。

在以下示例中，我们将`train` Fashion MNIST 数据集分配给一个对象，在此数据上创建一个 RBM 模型，并使用模型的结果进行预测。关于 RBM 算法如何实现这一结果以及建议的应用将在后续章节中详细探讨。我们将通过运行以下代码，看到我们如何使用熟悉的语法简单地训练此模型并用于预测：

```py
library(RBM)

data(Fashion)

train <- Fashion$trainX
train_label <- Fashion$trainY

rbmModel <- RBM(x = t(train), y = train_label, n.iter = 500, n.hidden = 200, size.minibatch = 10)

test <- Fashion$testX
test_label <- Fashion$testY

PredictRBM(test = t(test), labels = test_label, model = rbmModel)
```

与`ReinforcementLearning`包类似，以下内容适用：

+   RBM 完全用 R 编写，因此浏览代码库是更好地理解这种特定技术如何工作的一个绝佳方法。

+   此外，正如之前提到的，如果你只需要使用 RBM 训练模型，那么使用一个独立的包是避免加载过多不必要的函数的好方法，这种情况在使用像 MXNet 这样的库时会出现。

+   综合包和独立包在深度学习工作流中各有其位置，因此本书将重点介绍它们各自的优缺点。

# 比较深度学习库

在比较本章中强调的三种综合机器学习库（Keras、H2O 和 MXNet）时，有三个主要的区别：外部语言依赖性、函数和语法（易用性和认知负担）。我们将逐一介绍这些主要区别。

三个软件包之间的第一个主要区别是它们的外部语言依赖性。如前所述，这些软件包都不是用 R 编写的。这意味着你需要在机器上安装额外的语言才能使这些软件包正常工作。这也意味着你不能轻松查看源文档来了解某个函数是如何工作的，或者为什么会收到特定的错误（当然，前提是你知道某种语言）。这些软件包是使用以下语言编写的：Keras 用 Python，H2O 用 Java，MXNet 用 C#。

下一个主要区别涉及每个软件包中可以实现的模型类型。你可以使用这三个软件包训练前馈模型，例如多层感知器，其中所有隐藏层都是全连接层。Keras 和 MXNet 允许你训练包括不同类型隐藏层的深度学习模型，并且支持层间反馈循环。这些模型包括 RNN、LSTM 和 CNN。MXNet 还支持其他算法，包括 GAN、RBM/DBN 和强化学习。

最后一个主要区别涉及模型的语法。有些语法非常熟悉，这使得它们更容易学习和使用。当代码类似于你用于其他用途的 R 代码时，你需要记住的内容会更少。为此，Keras 采用了一种非常熟悉的语法。它是模块化的，这意味着每个函数在整体模型中执行一个离散的步骤，所有函数都可以链式连接起来。

这与 `tidyverse` 函数如何连接在一起进行数据准备非常相似。MXNet 的语法类似于其他机器学习库，其中数据集和目标变量向量被传递给函数以训练模型，同时还有许多额外的参数来控制模型的创建。H2O 的语法则与常见的 R 编程习惯最为不同。它要求在任何建模之前初始化一个集群。数据也必须存储在特定的数据对象中，一些函数通过对该对象调用函数来操作数据对象，而不需要将结果分配给一个新对象，这与典型的 R 编程不同。

除了这些差异，Keras 还提供了使用 TensorFlow 的方法，而 H2O 允许将较大的对象存储在内存之外。如前所述，MXNet 提供了最强大的深度学习算法库。正如我们所看到的，每个包都有其优点，在本书中我们将深入探讨它们，并在此过程中指出最合适的使用案例和应用。

# 总结

完成本章后，你应该已经安装了本书中将使用的所有库。此外，你还应该熟悉每个库的语法，并且已经看到过如何使用它们中的每个来训练模型的初步示例。我们还探讨了深度学习库之间的一些差异，指出了它们的优点以及局限性。这三大主要库（Keras、MXNet 和 H2O）在工业和学术界广泛应用于深度学习，理解这些库将使你能够解决多个深度学习问题。现在，我们准备深入探索这些库。不过，在此之前，我们将回顾一下神经网络——所有深度学习的基础构建块。

在接下来的章节中，你将学习人工神经网络，它构成了所有深度学习的基本构建模块。在下一章中我们还不会使用这些深度学习库；然而，神经网络如何编码的基础知识将对我们后续的学习至关重要。在下一章中，我们所涵盖的内容将继续推进，并在我们编写深度学习模型示例时非常有用。所有深度学习的案例都是基本神经网络的变体，我们将在下一章中学习如何创建这些神经网络。
