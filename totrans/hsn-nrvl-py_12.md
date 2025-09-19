# 协同进化和SAFE方法

在本章中，我们介绍了协同进化的概念，并解释了如何使用它来协同进化求解器和优化求解器进化的目标函数。然后，我们讨论了**解决方案和适应度进化**（**SAFE**）方法，并简要概述了不同的协同进化策略。您将学习如何使用基于神经进化的方法进行协同进化。您还将获得修改迷宫求解实验的实践经验。

本章将涵盖以下主题：

+   协同进化基础和常见协同进化策略

+   SAFE方法基础

+   修改后的迷宫求解实验

+   关于实验结果的讨论

# 技术要求

为了执行本章中描述的实验，应满足以下技术要求：

+   Windows 8/10，macOS 10.13或更高版本，或现代Linux

+   Anaconda Distribution版本2019.03或更高版本

本章的代码可以在[https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python/tree/master/Chapter9](https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python/tree/master/Chapter9)找到。

# 常见的协同进化策略

生物系统的自然进化不能与协同进化的概念分开考虑。协同进化是导致当前生物圈状态的中央进化驱动力之一，其中包括我们周围可感知的有机体的多样性。

我们可以将协同进化定义为多个不同生物系谱同时进化的互利策略。一个物种的进化不可能在没有其他物种的情况下进行。在进化过程中，协同进化的物种相互互动，这些物种间的关系塑造了它们的进化策略。

存在三种主要的协同进化类型：

+   **互利共生**是指两种或更多物种共存并相互受益。

+   **竞争性协同进化**：

    +   **捕食**是指一个生物体杀死另一个生物体并消耗其资源。

    +   **寄生**是指一个生物体利用另一个生物体的资源，但不会杀死它。

+   **共生**是指一种物种的成员从另一种物种中受益，而不对另一种物种造成伤害或利益。

研究人员已经探索了每种协同进化策略，它们作为神经进化过程的指导原则各有优缺点。然而，最近有一组研究人员探索了共生策略作为神经进化的指导原则，并取得了有希望的结果。他们创建了SAFE算法，我们将在本章中讨论。

关于SAFE算法的更多细节，请参阅原始出版物[https://doi.org/10.1007/978-3-030-16670-0_10](https://doi.org/10.1007/978-3-030-16670-0_10)。

现在我们已经涵盖了常见的协同进化类型，让我们详细讨论SAFE方法。

# SAFE方法

如其名所示，SAFE方法涉及解决方案和适应度函数的协同进化，这引导了解决方案搜索优化。SAFE方法围绕两个种群之间的*共生*协同进化策略构建：

+   那些进化以解决当前问题的潜在解决方案种群

+   那些进化以引导解决方案种群进化的目标函数候选种群

在这本书中，我们已经讨论了几种可以用来指导潜在解决方案进化过程的搜索优化策略。这些策略是基于目标函数的适应度优化和新颖搜索优化。前一种优化策略在适应度函数景观简单的情况下非常完美，我们可以将优化搜索集中在最终目标上。在这种情况下，我们可以使用基于目标的度量标准，它评估在每个进化时代，我们的当前解决方案与目标有多接近。

新颖搜索优化策略是不同的。在这个策略中，我们并不关心候选解与最终目标的接近程度，而是主要关注候选解所采取的路径。新颖搜索方法背后的核心思想是逐步探索垫脚石，最终引导到目的地。这种优化策略非常适合我们面临的是一个复杂度高的适应度函数景观，其中有许多误导性的死胡同和局部最优解的情况。

因此，SAFE方法背后的主要思想是利用这里提到的两种搜索优化方法的优势。接下来，我们将讨论修改后的迷宫实验，该实验使用这里提到的两种搜索优化方法来指导神经进化过程。

# 修改后的迷宫实验

我们已经在本书中讨论了如何将基于目标的搜索优化或新颖搜索优化方法应用于解决迷宫的问题。在本章中，我们介绍了一种修改后的迷宫解决实验，我们尝试使用SAFE算法结合这两种搜索优化方法。

我们介绍了两个种群之间的协同进化：一个是迷宫解决代理种群，另一个是目标函数候选种群。遵循SAFE方法，我们在实验中采用了一种共生协同进化策略。让我们首先讨论迷宫解决代理。

# 迷宫解决代理

迷宫解决代理配备了一套传感器，使其能够感知迷宫环境，并在每一步知道迷宫出口的方向。传感器的配置如下所示：

![图片](img/2fa62210-89a3-4c3e-83eb-2ea33a48e645.png)

迷宫解决代理的传感器配置

在前面的图中，暗箭头定义了测距传感器的作用范围，允许代理感知障碍物并找到给定方向上的障碍物距离。围绕机器人身体的四个区域是扇形雷达，它们在每个时间步检测迷宫出口的方向。机器人身体内部的浅箭头确定机器人面向的方向。

此外，机器人有两个执行器：一个用于改变其角速度（旋转），另一个用于改变其线性速度。

我们使用与[第5章](22365f85-3003-4b67-8e1e-cc89fa5e259b.xhtml)，*自主迷宫导航*中相同的机器人配置。因此，您应该参考该章节以获取更多详细信息。现在我们已经涵盖了迷宫求解代理，让我们来看看迷宫环境。

# 迷宫环境

迷宫被定义为由外部墙壁包围的区域。在迷宫内部，多个内部墙壁创建了多个局部适应度最优的死胡同，这使得以目标为导向的优化搜索不太有效。此外，由于局部适应度最优值，基于目标的搜索代理可能会陷入特定的死胡同，完全停止进化过程。死胡同在以下图中显示：

![图片](img/d638be6b-cb23-4f8c-85b4-2cceb6bb9397.png)

迷宫中的局部最优区域

在前面的图中，求解代理的起始位置用左下角的实心圆圈标记，迷宫出口用左上角的实心圆圈标记。欺骗性的局部适应度最优值以实心扇区形式显示在代理的起始位置。

迷宫环境通过配置文件定义，我们已经实现了模拟器来模拟求解代理在迷宫中的遍历。我们已在[第5章](22365f85-3003-4b67-8e1e-cc89fa5e259b.xhtml)，*自主迷宫导航*中讨论了迷宫模拟器环境的实现，您可以参考该章节以获取详细信息。

在本章中，我们讨论了引入到原始实验中的修改，以实现SAFE优化策略。最关键的区别在于适应度函数的定义，我们将在下一节中讨论。

您可以在源代码中查看迷宫模拟器环境的完整实现细节，网址为[https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python/blob/master/Chapter9/maze_environment.py](https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python/blob/master/Chapter9/maze_environment.py)。

# 适应度函数定义

SAFE方法涉及解决方案候选者和目标函数候选者的共同进化，也就是说，我们有两种共同进化的种群。因此，我们需要定义两个适应度函数：一个用于解决方案候选者（迷宫求解器），另一个用于目标函数候选者。在本节中，我们讨论了这两种变体。

# 迷宫求解器的适应度函数

在进化的每一代中，每个解决方案个体（迷宫求解器）都会与所有目标函数候选者进行评估。我们使用在评估迷宫求解器与每个目标函数候选者时获得的最高适应度得分作为解决方案的适应度得分。

迷宫求解器的适应度函数是两个指标的总和——迷宫出口的距离（基于目标的得分）和求解器最终位置的新颖性（新颖性得分）。这些得分通过从目标函数候选者的特定个体中获得的系数对进行算术组合。

以下公式给出了这些得分的组合作为适应度得分：

![](img/99dd6cf2-6a2e-4556-8ed8-5425e150abf8.png)

![](img/3503325b-6d31-41e2-b0f0-6a834628f697.png) 是通过评估解决方案候选者 ![](img/123669da-9593-4d08-8241-5ff4cfac875e.png) 对目标函数 ![](img/e453f8b3-5790-49f1-a05f-0c51d6a62f68.png) 的评估获得的适应度值。所使用的系数对 ![](img/53c7b36c-1a65-4a9f-8595-ca7a756e87a3.png) 是特定目标函数候选者的输出。这对系数决定了迷宫出口 (![](img/624adf98-48ae-4495-99e3-5530a8419e25.png)) 和解决方案的行为新颖性 (![](img/d029abbb-a6c9-40ab-9c34-a925f61ccac2.png)) 如何影响迷宫求解器在轨迹末尾的最终适应度得分。

迷宫出口的距离 (![](img/da936b97-ef1d-4664-b1c0-5b345b1d4584.png)) 被确定为迷宫求解器的最终坐标和迷宫出口坐标之间的欧几里得距离。这如下公式所示：

![](img/6f31a6a7-a7d2-4f25-8104-1dde6650c8d4.png)

![](img/3e37f415-f49c-4e25-b2b7-bc6da6ba6cb5.png) 和 ![](img/3867c95c-8ab2-45a4-a478-67c68f75eef7.png) 是迷宫求解器的最终坐标，而 ![](img/2f68814d-b56a-4612-beda-2b10ad0268fd.png) 和 ![](img/969ad1ef-bfbf-4e2d-a978-e474b87b9d20.png) 是迷宫出口的坐标。

每个迷宫求解器的创新得分，![](img/04896aaa-ddc1-4b2a-a092-523c6d720d2f.png)，由其在迷宫中的最终位置（点 ![](img/e73f9cc2-2135-4af1-a181-cf88e46c68e6.png)）决定。它被计算为从这个点到最近的k个邻居点的平均距离，这些邻居点的位置是其他迷宫求解器的最终位置。

以下公式给出了行为空间中点 *x* 处的创新得分值：

![](img/8a0e787f-c22c-48ee-9cf9-c45937899fcc.png)

![](img/ee826d80-17a8-43ca-b909-3da2823ca5e6.png) 是 ![](img/ba9a13e6-7f4d-4fa9-90a9-b46b55b0734b.png) 的第i个最近邻，![](img/669712d0-2d48-42c2-9474-6e2c62206052.png) 是 ![](img/1bae89af-0a10-4b72-ad2d-576c0ae140a1.png) 和 ![](img/a9776135-ca81-445d-b2fd-a900c50f2d65.png) 之间的距离。

两点之间的距离是新颖度度量，衡量当前解决方案（![](img/7178d7f4-99f7-4bb3-bbc1-5b18e3af1c14.png)）与由不同迷宫求解者产生的另一个（![](img/21fafe7a-1559-41b7-b414-fcfa2f9bdb4a.png)）之间的差异。新颖度度量是两点之间的欧几里得距离：

![](img/217974f3-7c7b-4cbc-8fa3-8f326324d2af.png)

![](img/5d45a339-7baa-4ae5-9d60-ab120d7b2472.png) 和 ![](img/4b5ba532-39df-4eba-b232-393c6597ed1b.png) 分别是坐标向量中持有 ![](img/8f5c16cc-d11d-4bc6-b4f4-c89d32720229.png) 和 ![](img/32cb7134-f8c8-4fd1-8305-87feb5b67701.png) 点坐标的位置 ![](img/19ab567e-e72f-438e-b07c-799a073fbe44.png) 的值。

接下来，我们将讨论如何定义目标函数候选者优化的适应度函数。

# 目标函数候选者的适应度函数

SAFE方法基于一种互利共生协同进化方法，这意味着在进化过程中，其中一个协同进化的种群既不受益也不受损。在我们的实验中，互利共生的种群是目标函数候选者的种群。对于这个种群，我们需要定义一个与迷宫求解者种群性能无关的适应度函数。

这样的函数候选者是一个使用新颖度评分作为要优化的适应度评分的适应度函数。计算每个目标函数候选者新颖度评分的公式与迷宫求解者给出的相同。唯一的区别是，在目标函数候选者的案例中，我们使用每个个体的输出值向量来计算新颖度评分。之后，我们使用新颖度评分值作为个体的适应度评分。

这种新颖度评分估计方法是改进的**新颖度搜索**（**NS**）方法的一部分，我们将在下一节中讨论。

# 改进的Novelty Search

我们在[第6章](62301923-b398-43da-b773-c8b1fe383f1d.xhtml)，《新颖度搜索优化方法》中介绍了NS方法。在当前实验中，我们使用NS方法的一个略微修改版本，我们将在下一节中讨论。

我们将在本次实验中提出的对NS方法的修改与维护新颖度点存档的新方法有关。新颖度点持有迷宫求解者在轨迹末尾在迷宫中的位置，并将其与新颖度评分相结合。

在NS方法的更传统版本中，新颖存档的大小是动态的，如果新颖度得分超过某个阈值（新颖度阈值），则允许添加特定的创新点。此外，新颖度阈值可以在运行时进行调整，考虑到在进化过程中新新颖点的发现速度。这些调整使我们能够控制存档的最大大小（在一定程度上）。然而，我们需要从一个初始新颖度阈值值开始，这个选择并不明显。

修改后的NS方法引入了固定大小的创新存档来解决选择正确新颖度阈值值的问题。新的新颖点被添加到存档中，直到它填满。之后，只有当新颖度得分超过存档当前最小得分时，才会将新颖点添加到存档中，通过用具有最小得分的当前点替换它。这样，我们可以保持新颖存档的固定大小，并在其中仅存储在进化过程中发现的最有价值的新颖点。

修改后的新颖存档实现的源代码可以在[https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python/blob/master/Chapter9/novelty_archive.py](https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python/blob/master/Chapter9/novelty_archive.py)找到。

接下来，让我们讨论实现中最有趣的部分。

# `_add_novelty_item` 函数

此功能允许在保持其大小的同时向存档中添加新的新颖点。其实现如下：

```py
        if len(self.novel_items) >= MAXNoveltyArchiveSize:
            # check if this item has higher novelty than  
            # last item in the archive (minimal novelty)
            if item > self.novel_items[-1]:
                # replace it
                self.novel_items[-1] = item
        else:
            # just add new item
            self.novel_items.append(item)

        # sort items array in descending order by novelty score
        self.novel_items.sort(reverse=True)
```

代码首先检查新颖存档的大小是否尚未超过，在这种情况下直接将新的新颖点附加到其中。否则，一个新的新颖点将替换存档中的最后一个项目，即具有最小新颖度得分的项目。我们可以确信存档中的最后一个项目具有最小的新颖度得分，因为在我们将新项目添加到存档后，我们按新颖度得分值降序排序。

# `evaluate_novelty_score` 函数

此函数提供了一种机制来评估新颖项目相对于已收集在新颖存档中的所有项目以及当前种群中发现的全部新颖项目的创新度得分。我们按照以下步骤计算新颖度得分，作为到 *k=15* 个最近邻的平均距离：

1.  我们需要收集提供的创新项目与新颖存档中所有项目之间的距离：

```py
        distances = []
        for n in self.novel_items:
            if n.genomeId != item.genomeId:
                distances.append(self.novelty_metric(n, item))
            else:
                print("Novelty Item is already in archive: %d" % 
                       n.genomeId)
```

1.  之后，我们将提供的创新项目与当前种群中的所有项目之间的距离添加到其中：

```py
        for p_item in n_items_list:
            if p_item.genomeId != item.genomeId:
                distances.append(self.novelty_metric(p_item, item))
```

1.  最后，我们可以估计平均k-最近邻值：

```py
        distances = sorted(distances) 
        item.novelty = sum(distances[:KNN])/KNN
```

我们将列表按距离升序排序，以确保最近的项首先出现在列表中。然后，我们计算列表中前*k=15*项的总和，并将其除以总和值的数量。因此，我们得到到*k-最近邻*的平均距离值。

修改后的NS优化方法是迷宫求解者种群和目标函数候选者种群适应度评分评估的核心。我们在实验运行器的实现中广泛使用它，我们将在下一节中讨论。

# 修改后的迷宫实验实现

实验运行器的实现基于MultiNEAT Python库，我们在本书的几个实验中使用了该库。每个协同进化种群的进化由基本NEAT算法控制，该算法在[第3章](7acd0cf5-c389-4e55-93d7-9438fcaa1390.xhtml)，*使用NEAT进行XOR求解器优化*，[第4章](34913ccd-6aac-412a-8f54-70d1900cef41.xhtml)，*杆平衡实验*，和[第5章](22365f85-3003-4b67-8e1e-cc89fa5e259b.xhtml)，*自主迷宫导航*中进行了讨论。

然而，在本节中，我们展示了如何使用NEAT算法来维持两个独立物种种群（迷宫求解器和目标函数候选者）的协同进化。

接下来，我们讨论修改后的迷宫实验运行器的关键部分。

更多细节，请参阅[https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python/blob/master/Chapter9/maze_experiment_safe.py](https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python/blob/master/Chapter9/maze_experiment_safe.py)的源代码。

# 协同进化种群的创建

在这个实验中，我们需要创建两个具有不同初始基因型配置的协同进化的物种种群，以满足产生物种的表型需求。

迷宫求解器的表型有11个输入节点来接收来自传感器的信号，以及两个输出节点来产生控制信号。同时，目标函数候选者的表型有一个输入节点接收固定值（`0.5`），该值被转换为两个输出值，用作迷宫求解器的适应度函数系数。

我们首先讨论如何创建目标函数候选者种群。

# 目标函数候选者种群的创建

编码目标函数候选者表型的基因型必须产生具有至少一个输入节点和两个输出节点的表型配置，正如之前所讨论的。我们在`create_objective_fun`函数中实现种群创建如下：

```py
    params = create_objective_fun_params()
    # Genome has one input (0.5) and two outputs (a and b)
    genome = NEAT.Genome(0, 1, 1, 2, False, 
        NEAT.ActivationFunction.TANH, # hidden layer activation
        NEAT.ActivationFunction.UNSIGNED_SIGMOID, # output layer activation
        1, params, 0)
    pop = NEAT.Population(genome, params, True, 1.0, seed)
    pop.RNG.Seed(seed)

    obj_archive = archive.NoveltyArchive(
                             metric=maze.maze_novelty_metric_euclidean)
    obj_fun = ObjectiveFun(archive=obj_archive, 
                             genome=genome, population=pop)
```

在此代码中，我们创建了一个具有一个输入节点、两个输出节点和一个隐藏节点的NEAT基因型。隐藏节点被预先种入初始基因组中以增强进化过程中的预定义非线性。隐藏层的激活函数类型被选为双曲正切，以支持负输出值。这一特性对于我们的任务至关重要。目标函数候选者产生的系数之一为负值可以表明迷宫求解代理适应性函数的特定组成部分具有负面影响，这会发出进化需要尝试其他路径的信号。

最后，我们创建`ObjectiveFun`对象来维护目标函数候选者的进化群体。

接下来，我们将讨论迷宫求解代理群体的创建方法。

# 创建迷宫求解代理的群体

迷宫求解代理需要从11个传感器获取输入并生成两个控制信号，这些信号影响机器人的角速度和线速度。因此，编码迷宫求解代理表型的基因组必须产生包含11个输入节点和两个输出节点的表型配置。您可以通过查看`create_robot`函数来了解迷宫求解代理初始基因组群体的创建过程：

```py
    params = create_robot_params()
    # Genome has 11 inputs and two outputs
    genome = NEAT.Genome(0, 11, 0, 2, False, 
                        NEAT.ActivationFunction.UNSIGNED_SIGMOID, 
                        NEAT.ActivationFunction.UNSIGNED_SIGMOID, 
                        0, params, 0)
    pop = NEAT.Population(genome, params, True, 1.0, seed)
    pop.RNG.Seed(seed)

    robot_archive = archive.NoveltyArchive(metric=maze.maze_novelty_metric)
    robot = Robot(maze_env=maze_env, archive=robot_archive, genome=genome, 
                  population=pop)
```

在代码中，我们从`create_robot_params`函数中获取适当的NEAT超参数。之后，我们使用它们来创建具有相应数量输入和输出节点的初始NEAT基因型。最后，我们创建一个`Robot`对象，它封装了与迷宫求解代理群体相关的所有数据，以及迷宫模拟环境。

现在，当我们创建了两个协同进化的群体后，我们需要实现两个群体中个体的适应性分数评估。我们将在下一节中讨论适应性分数评估的实现细节。

# 协同进化群体的适应性评估

已经定义了两个协同进化的群体后，我们需要创建函数来评估每个群体中个体的适应性分数。正如我们之前提到的，迷宫求解代理群体中个体的适应性分数取决于目标函数候选者群体产生的输出。同时，每个目标函数候选者的适应性分数完全由该个体的新颖性分数决定。

因此，我们有两种不同的方法来评估适应性分数，我们需要实现两个不同的函数。以下我们将讨论这两种实现方法。

# 目标函数候选者的适应性评估

目标函数候选者群体中每个个体的适应性分数由其新颖性分数决定，该分数的计算方法我们之前已经讨论过。适应性分数评估的实现被分为两个函数：`evaluate_obj_functions`和`evaluate_individ_obj_function`。

接下来，我们将讨论这两个函数的实现。

# `evaluate_obj_functions` 函数实现

此函数接受 `ObjectiveFun` 对象，该对象包含目标函数候选者的种群，并使用它通过以下步骤来估计种群中每个个体的适应度分数：

1.  首先，我们遍历种群中的所有基因组，并为每个基因组收集新颖性点：

```py
    obj_func_genomes = NEAT.GetGenomeList(obj_function.population)
    for genome in obj_func_genomes:
        n_item = evaluate_individ_obj_function(genome=genome, 
                                            generation=generation)
        n_items_list.append(n_item)
        obj_func_coeffs.append(n_item.data)
```

在代码中，从 `evaluate_individ_obj_function` 函数获得的新颖性点被追加到种群新颖性点列表中。此外，我们将新颖性点数据追加到系数对列表中。该系数对列表将用于估计个体迷宫求解器的适应度分数。

1.  接下来，我们遍历种群基因组的列表，并使用上一步收集到的新颖性点来评估每个基因组的 novelty 分数：

```py
    max_fitness = 0
    for i, genome in enumerate(obj_func_genomes):
        fitness = obj_function.archive.evaluate_novelty_score(
               item=n_items_list[i],n_items_list=n_items_list)
        genome.SetFitness(fitness)
        max_fitness = max(max_fitness, fitness)
```

使用新颖性点估计的新颖性分数已经收集在新颖性存档中和为当前种群创建的新颖性点列表中。之后，我们将估计的新颖性分数设置为相应基因组的适应度分数。此外，我们找到适应度分数的最大值，并返回它，以及系数对列表。

# `evaluate_individ_obj_function` 函数实现

此函数接受目标函数候选者的个体 NEAT 基因组，并返回新颖性点评估结果。我们按以下方式实现它：

```py
    n_item = archive.NoveltyItem(generation=generation, genomeId=genome_id)
    # run the simulation
    multi_net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(multi_net)
    depth = 2
    try:
        genome.CalculateDepth()
        depth = genome.GetDepth()
    except:
        pass
    obj_net = ANN(multi_net, depth=depth)

    # set inputs and get outputs ([a, b])
    output = obj_net.activate([0.5])

    # store coefficients
    n_item.data.append(output[0])
    n_item.data.append(output[1])
```

我们从创建一个 `NoveltyItem` 对象开始，以保存给定基因组的 novelty 点数据。之后，我们构建一个表型人工神经网络（ANN）并用输入 `0.5` 激活它。最后，我们使用 ANN 的输出创建 novelty 点。

在下一节中，我们将讨论迷宫求解种群中个体的适应度分数评估。

# 迷宫求解代理的适应度评估

我们将迷宫求解种群中每个个体的适应度分数估计为一个由两个组成部分组成的复合体：新颖性分数和轨迹结束时到达迷宫出口的距离。每个组成部分的影响由目标函数候选者种群中个体产生的系数对控制。

适应度分数评估分为三个函数，我们将在下面讨论。

# `evaluate_solutions` 函数实现

`evaluate_solutions` 函数接收 `Robot` 对象作为输入参数，该对象维护迷宫求解代理的种群和迷宫环境模拟器。它还接收在评估目标函数候选者种群期间生成的系数对列表。

我们使用函数的输入参数来评估种群中的每个基因组，并估计其适应度函数。在这里，我们讨论基本实现细节：

1.  首先，我们将种群中的每个个体与迷宫模拟器进行评估，并找到轨迹末尾到迷宫出口的距离：

```py
    robot_genomes = NEAT.GetGenomeList(robot.population)
    for genome in robot_genomes:
        found, distance, n_item = evaluate_individual_solution(
            genome=genome, generation=generation, robot=robot)
        # store returned values
        distances.append(distance)
        n_items_list.append(n_item)
```

1.  接下来，我们遍历种群中的所有基因，并估计每个个体的新颖度得分。同时，我们使用之前收集的相应的到迷宫出口的距离，并将其与计算出的新颖度得分结合起来，以评估基因的适应度：

```py
    for i, n_item in enumerate(n_items_list):
        novelty = robot.archive.evaluate_novelty_score(item=n_item, 
                                         n_items_list=n_items_list)
        # The sanity check
        assert robot_genomes[i].GetID() == n_item.genomeId

        # calculate fitness
        fitness, coeffs = evaluate_solution_fitness(distances[i], 
                                        novelty, obj_func_coeffs)
        robot_genomes[i].SetFitness(fitness)
```

在代码的前半部分，我们使用`robot.archive.evaluate_novelty_score`函数来估计种群中每个个体的新颖度得分。后半部分调用`evaluate_solution_fitness`函数，使用新颖度得分和到迷宫出口的距离来估计每个个体的适应度得分。

1.  最后，我们收集关于种群中最佳迷宫求解器基因的性能评估统计数据：

```py
        if not solution_found:
            # find the best genome in population
            if max_fitness < fitness:
                max_fitness = fitness
                best_robot_genome = robot_genomes[i]
                best_coeffs = coeffs
                best_distance = distances[i]
                best_novelty = novelty
        elif best_robot_genome.GetID() == n_item.genomeId:
            # store fitness of winner solution
            max_fitness = fitness
            best_coeffs = coeffs
            best_distance = distances[i]
            best_novelty = novelty
```

最后，函数返回在种群评估过程中收集的所有统计数据。

此后，我们讨论如何评估个体迷宫求解器基因相对于迷宫环境模拟器。

# `evaluate_individual_solution`函数的实现

这是评估特定迷宫求解器相对于迷宫环境模拟器性能的函数。其实现如下：

1.  首先，我们创建迷宫求解器的表型人工神经网络（ANN），并将其用作控制器来引导机器人穿越迷宫：

```py
    n_item = archive.NoveltyItem(generation=generation, 
                                 genomeId=genome_id)
    # run the simulation
    maze_env = copy.deepcopy(robot.orig_maze_environment)
    multi_net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(multi_net)
    depth = 8
    try:
        genome.CalculateDepth()
        depth = genome.GetDepth()
    except:
        pass
    control_net = ANN(multi_net, depth=depth)
    distance = maze.maze_simulation_evaluate(
        env=maze_env, net=control_net, 
        time_steps=SOLVER_TIME_STEPS, n_item=n_item)
```

在代码中，我们创建一个`NoveltyItem`对象来保存创新点，该创新点由机器人在迷宫中的最终位置定义。之后，我们创建表型ANN并运行迷宫模拟器，将其用作控制ANN进行给定数量的时间步（400）。模拟完成后，我们接收迷宫求解器最终位置与迷宫出口之间的距离。

1.  接下来，我们将模拟统计信息保存到我们在实验结束时分析的`AgentRecord`对象中：

```py
    record = agent.AgenRecord(generation=generation, 
                              agent_id=genome_id)
    record.distance = distance
    record.x = maze_env.agent.location.x
    record.y = maze_env.agent.location.y
    record.hit_exit = maze_env.exit_found
    record.species_id = robot.get_species_id(genome)
    robot.record_store.add_record(record)
```

之后，该函数返回一个包含以下值的元组：一个标志，指示我们是否找到了解决方案，机器人轨迹末尾到迷宫出口的距离，以及封装有关发现的创新点信息的`NoveltyItem`对象。

在下一节中，我们讨论迷宫求解器适应度函数的实现。

# `evaluate_solution_fitness`函数的实现

此函数是实现我们之前讨论过的迷宫求解器适应度函数。该函数接收到迷宫出口的距离、新颖度得分以及当前目标函数候选者生成器生成的系数对列表。然后，它使用接收到的输入参数来计算适应度得分，如下所示：

```py
    normalized_novelty = novelty
    if novelty >= 1.00:
        normalized_novelty = math.log(novelty)
    norm_distance = math.log(distance)

    max_fitness = 0
    best_coeffs = [-1, -1]
    for coeff in obj_func_coeffs:
        fitness = coeff[0] / norm_distance + coeff[1] * normalized_novelty
        if fitness > max_fitness:
            max_fitness = fitness
            best_coeffs[0] = coeff[0]
            best_coeffs[1] = coeff[1]
```

首先，我们需要使用自然对数对距离和新颖度得分值进行归一化。这种归一化将保证距离和新颖度得分值始终处于相同的尺度。确保这些值处于相同的尺度是必要的，因为系数对始终在范围 `[0,1]` 内。因此，如果距离和新颖度得分值具有不同的尺度，一对系数将无法在计算适应度分数时影响每个值的显著性。

代码遍历系数对的列表，并对每一对系数，通过结合距离和新颖度得分值来计算适应度分数。

迷宫求解器的最终适应度分数是所有找到的适应度分数中的最大值。然后，该值和相应的系数对由函数返回。

# 修改后的迷宫实验运行器

现在，当我们已经实现了创建共同进化种群和评估这些种群中个体适应度的所有必要程序后，我们就可以开始实现实验运行器循环了。

完整的细节可以在`maze_experiment_safe.py`文件中的`run_experiment`函数中找到，该文件位于[https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python/blob/master/Chapter9/maze_experiment_safe.py](https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python/blob/master/Chapter9/maze_experiment_safe.py)。

在这里，我们讨论实现的关键细节：

1.  我们从创建共同进化的物种对应种群开始：

```py
    robot = create_robot(maze_env, seed=seed)
    obj_func = create_objective_fun(seed)
```

1.  接下来，我们开始进化循环，并如下评估两个种群：

```py
    for generation in range(n_generations):
        # evaluate objective function population
        obj_func_coeffs, max_obj_func_fitness = \
                    evaluate_obj_functions(obj_func, generation)
        # evaluate robots population
        robot_genome, solution_found, robot_fitness, distances, \
        obj_coeffs, best_distance, best_novelty = \
          evaluate_solutions(robot=robot, 
          obj_func_coeffs=obj_func_coeffs, generation=generation)
```

1.  在评估种群之后，我们将当前进化代的结果保存为统计数据：

```py
        stats.post_evaluate(max_fitness=robot_fitness, 
                            errors=distances)
        # store the best genome
        best_fitness = robot.population.GetBestFitnessEver()
        if solution_found or best_fitness < robot_fitness:
            best_robot_genome_ser = pickle.dumps(robot_genome)
            best_robot_id = robot_genome.GetID()
            best_obj_func_coeffs = obj_coeffs
            best_solution_novelty = best_novelty
```

1.  在进化循环结束时，如果当前代未找到解决方案，我们向两个种群发出信号，使其进入下一个时代：

```py
        if solution_found:
            print('Solution found at generation: %d, best fitness: %f, species count: %d' % (generation, robot_fitness, len(pop.Species)))
            break
        # advance to the next generation
        robot.population.Epoch()
        obj_func.population.Epoch()
```

1.  在进化循环完成对指定代数的迭代后，我们可视化收集到的迷宫记录：

```py
        if args is None:
            visualize.draw_maze_records(maze_env, 
                       robot.record_store.records, 
                       view=show_results)
        else:
            visualize.draw_maze_records(maze_env, 
                      robot.record_store.records, 
                      view=show_results, width=args.width, 
                      height=args.height,
                      filename=os.path.join(trial_out_dir, 
                                     'maze_records.svg'))
```

这里提到的迷宫记录包含在进化过程中收集的迷宫模拟器中每个迷宫求解器基因组的评估统计数据，作为`AgentRecord`对象。在可视化中，我们使用迷宫绘制每个评估的迷宫求解器的最终位置。

1.  接下来，我们使用在进化过程中找到的最佳求解器基因组创建的控制ANN进行迷宫求解模拟。迷宫求解器在模拟过程中的轨迹可以如下可视化：

```py
        multi_net = NEAT.NeuralNetwork()
        best_robot_genome.BuildPhenotype(multi_net)

        control_net = ANN(multi_net, depth=depth)
        path_points = []
        distance = maze.maze_simulation_evaluate(
                                    env=maze_env, 
                                    net=control_net, 
                                    time_steps=SOLVER_TIME_STEPS,
                                    path_points=path_points)
        print("Best solution distance to maze exit: %.2f, novelty: %.2f" % (distance, best_solution_novelty))
        visualize.draw_agent_path(robot.orig_maze_environment, 
                          path_points, best_robot_genome,
                          view=show_results, width=args.width, 
                          height=args.height, 
                          filename=os.path.join(trial_out_dir,
                                      'best_solver_path.svg'))
```

首先，代码从最佳求解器基因组创建一个表型人工神经网络（ANN）。然后，它使用创建的表型ANN作为迷宫求解器控制器运行迷宫模拟器。我们随后绘制迷宫求解器的收集轨迹点。

1.  最后，我们以下列方式绘制每代的平均适应度分数图：

```py
        visualize.plot_stats(stats, ylog=False, view=show_results, 
           filename=os.path.join(trial_out_dir,'avg_fitness.svg'))
```

这里提到的所有可视化内容也都以SVG文件的形式保存在本地文件系统中，以后可用于结果分析。

在下一节中，我们将讨论如何运行修改后的迷宫实验以及实验结果。

# 修改后的迷宫实验

我们几乎准备好使用修改后的迷宫实验开始协同进化实验。然而，在那之前，我们需要讨论每个协同进化种群的超参数选择。

# 迷宫求解器种群的超参数

对于这个实验，我们选择使用MultiNEAT Python库，该库使用`Parameters` Python类来维护所有支持的超参数列表。迷宫求解器种群的超参数初始化在`create_robot_params`函数中定义。接下来，我们将讨论关键超参数及其选择特定值的原因：

1.  我们决定从一开始就有一个中等大小的种群，以提供足够的种群多样性：

```py
    params.PopulationSize = 250
```

1.  我们对在进化过程中产生紧凑的基因组拓扑结构以及限制种群中物种数量感兴趣。因此，我们在进化过程中定义了非常小的添加新节点和连接的概率：

```py
    params.MutateAddNeuronProb = 0.03
    params.MutateAddLinkProb = 0.05
```

1.  新颖度得分奖励在迷宫中找到独特位置。实现这一目标的一种方法是在表型中增强数值动力学。因此，我们增加了连接权重的范围：

```py
    params.MaxWeight = 30.0
    params.MinWeight = -30.0
```

1.  为了支持进化过程，我们选择通过定义传递到下一代基因组的比例来引入精英主义：

```py
    params.Elitism = 0.1
```

精英主义值决定了大约十分之一的个体将被带到下一代。

# 目标函数候选种群的超参数

我们在`create_objective_fun_params`函数中为客观函数候选种群的进化创建超参数。在这里，我们讨论最重要的超参数：

1.  我们决定从一个小的种群开始，以减少计算成本。此外，预期目标函数候选的基因型不会非常复杂。因此，一个小种群应该足够：

```py
    params.PopulationSize = 100
```

1.  与迷宫求解器类似，我们感兴趣的是产生紧凑的基因组。因此，添加新节点和连接的概率保持非常小：

```py
    params.MutateAddNeuronProb = 0.03
    params.MutateAddLinkProb = 0.05
```

我们不期望目标函数候选种群中的基因组拓扑结构复杂。因此，大多数超参数都设置为默认值。

# 工作环境设置

在这个实验中，我们使用MultiNEAT Python库。因此，我们需要创建一个合适的Python环境，其中包括这个库和其他依赖项。您可以使用以下命令在Anaconda的帮助下设置Python环境：

```py
$ conda create --name maze_co python=3.5
$ conda activate maze_co
$ conda install -c conda-forge multineat 
$ conda install matplotlib
$ conda install graphviz
$ conda install python-graphviz
```

这些命令创建了一个使用Python 3.5的`maze_co`虚拟环境，并将所有必要的依赖项安装到其中。

# 运行修改后的迷宫实验

现在，我们已经准备好在新创建的虚拟环境中运行实验。你可以通过克隆相应的Git仓库并使用以下命令运行脚本来开始实验：

```py
$ git clone https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python.git
$ cd Hands-on-Neuroevolution-with-Python/Chapter9
$ python maze_experiment_safe.py -t 1 -g 150 -m medium
```

不要忘记使用`conda activate maze_co`命令激活适当的虚拟环境。

前面的命令启动了一个实验的试验，使用中等复杂性的迷宫配置进行150代的进化。大约在100代进化后，神经进化过程发现了一个成功的解决方案，你应该能够在控制台看到以下输出：

```py
****** Generation: 105 ******

Maze solved in 338 steps

Solution found at generation: 105, best fitness: 3.549289, species count: 7

==================================
Record store file: out/maze_medium_safe/5/data.pickle
Random seed: 1571021768
Best solution fitness: 3.901621, genome ID: 26458
Best objective func coefficients: [0.7935419704765059, 0.9882050653334634]
------------------------------
Maze solved in 338 steps
Best solution distance to maze exit: 3.56, novelty: 19.29
------------------------
Trial elapsed time: 4275.705 sec
==================================
```

从这里展示的输出中，你可以看到在第105代找到了一个成功的迷宫求解器，并且能够在400步中解决迷宫。有趣的是，注意到由最佳目标函数候选者产生的系数对给迷宫求解器适应度函数的新颖度分数组件赋予了略微更多的重视。

看一下每代最佳适应度分数的图表也是很有趣的：

![图片](img/812c2596-9516-4ca5-a854-6f90a391fa61.png)

每代的适应度分数

在前面的图表中，你可以看到最佳适应度分数在进化的早期代数达到最大值。这是由于高新颖度分数值，在进化的开始阶段更容易获得，因为有许多迷宫区域尚未被探索。另一个需要注意的重要点是，平均距离到迷宫出口在大多数进化代数中几乎保持在同一水平。因此，我们可以假设正确的解决方案不是通过逐步改进，而是通过冠军基因的质量飞跃找到的。这一结论也得到下一个图表的支持，其中我们按物种渲染收集到的迷宫记录：

![图片](img/c3c0e7fd-8a02-43b7-a9f3-df8e4e59c386.png)

记录了最终迷宫求解器的位置

前面的图表分为两部分：上面部分是具有目标适应度分数（基于迷宫出口的距离）大于**0.8**的物种，下面部分是其他物种。你可以看到只有一种物种产生了能够到达迷宫出口附近区域的迷宫求解器基因组。此外，你还可以看到，该物种的基因组通过探索比所有其他物种加起来还要多的迷宫区域，表现出非常探索性的行为。

最后，我们讨论了成功迷宫求解器在迷宫中的路径，如下面的图表所示：

![图片](img/70fc92ee-1779-4fef-b8a6-4ea935d83d97.png)

成功迷宫求解器在迷宫中的路径

成功迷宫求解器的路径对于给定的迷宫配置来说是近最优的。

这个实验也展示了初始条件在找到成功解决方案中的重要性。初始条件是由我们在运行实验之前选择的随机种子值定义的。

# 练习

1.  我们已经将难以解决的迷宫配置纳入了实验源代码中，可以在[https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python/blob/master/Chapter9/hard_maze.txt](https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python/blob/master/Chapter9/hard_maze.txt)找到。你可以通过以下命令尝试解决这个困难的迷宫配置：`python maze_experiment_safe.py -g 120 -t 5 -m hard --width 200 --height 200`。

1.  我们发现使用`1571021768`作为随机种子值是一个成功的解决方案。尝试找到另一个产生成功解决方案的随机种子值。找到它需要多少代？

# 摘要

在本章中，我们讨论了两种物种群体的协同进化。你学习了如何通过共生协同进化来产生一群成功的迷宫解决者。我们向你介绍了一种令人兴奋的方法，即结合基于目标的分数和新颖性分数，使用目标函数候选群体产生的系数来实现迷宫解决者的适应度函数。此外，你还了解了改进后的新颖性搜索方法，以及它与我们在[第6章](62301923-b398-43da-b773-c8b1fe383f1d.xhtml)“新颖性搜索优化方法”中讨论的原方法有何不同。

利用本章获得的知识，你将能够将共生协同进化方法应用于你的工作或研究任务，这些任务没有明确的适应度函数定义。

在下一章中，你将了解深度神经进化方法以及如何使用它来进化能够玩经典Atari游戏的智能体。
