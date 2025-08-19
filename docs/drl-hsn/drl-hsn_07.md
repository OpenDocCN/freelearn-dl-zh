

# 第七章：高级 RL 库

在第六章中，我们实现了由 DeepMind 于 2015 年发布的深度 Q 网络（DQN）模型[Mni+15]。这篇论文通过证明尽管普遍认为不可能，但在 RL 中使用非线性近似器是可行的，极大地影响了 RL 领域。这一概念验证激发了深度 Q 学习领域和深度 RL 一般领域的极大兴趣。

在本章中，我们将通过讨论高级 RL 库迈向实际应用的 RL，这将使你能够从更高级的模块构建代码，专注于你正在实现方法的细节，避免重复实现相同的逻辑。本章大部分内容将介绍 PyTorch AgentNet（PTAN）库，它将在本书的其余部分中使用，以防止代码重复，因此会进行详细介绍。

我们将涵盖以下内容：

+   使用高级库的动机，而不是从头重新实现所有内容

+   PTAN 库，包括最重要部分的覆盖，代码示例将加以说明

+   在 CartPole 上实现的 DQN，使用 PTAN 库

+   你可能考虑的其他 RL 库

# 为什么选择 RL 库？

我们在第六章中实现的基本 DQN 并不长且复杂——大约 200 行训练代码，加上 50 行环境包装代码。当你开始熟悉 RL 方法时，自己实现所有内容非常有用，可以帮助你理解事物是如何运作的。然而，随着你在该领域的深入，你会越来越频繁地意识到自己在一遍又一遍地编写相同的代码。

这种重复来自 RL 方法的通用性。正如我们在第一章中讨论的，RL 非常灵活，许多现实问题都可以归结为环境-代理互动模型。RL 方法对观察和动作的具体内容不做过多假设，因此为 CartPole 环境编写的代码可以适用于 Atari 游戏（可能需要做一些小的调整）。

一遍又一遍地编写相同的代码效率不高，因为每次可能都会引入 bug，这将花费你大量的调试和理解时间。此外，经过精心设计并在多个项目中使用的代码通常在性能、单元测试、可读性和文档方面具有更高的质量。

相较于其他更加成熟的领域，RL 的实际应用在计算机科学标准下相对较年轻，因此你可能没有那么丰富的选择。例如，在 Web 开发中，即便你只使用 Python，你也有数百个非常优秀的库可供选择：Django 用于重型、功能齐全的网站；Flask 用于轻量级的 Web 服务器网关接口（WSGI）应用；还有许多其他大小不一的库。

强化学习（RL）不像 Web 框架那样成熟，但你仍然可以从几个简化强化学习实践者生活的项目中进行选择。此外，你始终可以像我几年前那样编写一套自己的工具。我创建的工具是一个名为 PTAN 的库，正如前面所提到的，它将在本书的后续部分中用于演示实例。

# PTAN 库

该库可以在 GitHub 上找到：[`github.com/Shmuma/ptan`](https://github.com/Shmuma/ptan)。所有后续示例都是使用 PTAN 0.8 版本实现的，可以通过以下命令在虚拟环境中安装：

```py
$ pip install ptan==0.8
```

PTAN 的最初目标是简化我的强化学习实验，它试图在两种极端之间找到平衡：

+   导入库后，只需写几行包含大量参数的代码，就可以训练提供的某个方法，例如 DQN（一个非常生动的例子是 OpenAI Baselines 和 Stable Baselines3 项目）。这种方法非常不灵活。当你按照库的预期使用时，它能很好地工作。但如果你想做一些复杂的操作，很快你就会发现自己在破解库并与其施加的约束作斗争，而不是解决你想解决的问题。

+   从头开始实现所有方法的逻辑。第二种极端方式提供了过多的自由度，需要一遍又一遍地实现重放缓冲区和轨迹处理，这既容易出错，又乏味且低效。

PTAN 尝试在这两种极端之间找到平衡，提供高质量的构建块来简化你的强化学习代码，同时保持灵活性，不限制你的创造力。

从高层次来看，PTAN 提供了以下实体：

+   Agent：一个知道如何将一批观察转化为一批待执行动作的类。它可以包含一个可选的状态，以便在一个回合内跟踪连续动作之间的信息。（我们将在第十五章的深度确定性策略梯度（DDPG）方法中使用这种方法，其中包括用于探索的 Ornstein–Uhlenbeck 随机过程。）该库为最常见的强化学习案例提供了多个代理，但如果没有预定义的类能满足你的需求，你始终可以编写自己的 BaseAgent 子类。

+   ActionSelector：一小段逻辑，知道如何从网络的某些输出中选择动作。它与 Agent 类协同工作。

+   ExperienceSource 及其子类：Agent 实例和 Gym 环境对象可以提供有关代理在回合中轨迹的信息。它的最简单形式是一次性提供一个（a, r, s′）过渡，但它的功能不仅限于此。

+   ExperienceSourceBuffer 及其子类：具有各种特征的重放缓冲区。它们包括一个简单的重放缓冲区和两个版本的优先级重放缓冲区。

+   各种实用工具类：例如，TargetNet 和用于时间序列预处理的包装器（用于在 TensorBoard 中跟踪训练进度）。

+   PyTorch Ignite 助手：可以用来将 PTAN 集成到 Ignite 框架中。

+   Gym 环境的包装器：例如，针对 Atari 游戏的包装器（与我们在第六章中描述的包装器非常相似）。

基本上就是这样。在接下来的章节中，我们将详细了解这些实体。

## 动作选择器

在 PTAN 术语中，动作选择器是一个帮助从网络输出到具体动作值的对象。最常见的情况包括：

+   贪婪（或 argmax）：Q 值方法常用的，当网络为一组动作预测 Q 值时，所需的动作是具有最大 Q(s,a)的动作。

+   基于策略：网络输出概率分布（以 logits 或归一化分布的形式），需要从该分布中采样一个动作。你在第四章讨论交叉熵方法时已经看过这个。

动作选择器由代理使用，通常不需要自定义（但你有这个选项）。库提供的具体类包括：

+   ArgmaxActionSelector：在传入张量的第二个维度上应用 argmax。它假设矩阵的第一维是 batch 维度。

+   ProbabilityActionSelector：从离散动作集的概率分布中采样。

+   EpsilonGreedyActionSelector：具有 epsilon 参数，指定随机动作被执行的概率。它还持有另一个 ActionSelector 实例，当我们不采样随机动作时使用它。

所有类假设会将 NumPy 数组传递给它们。此章节的完整示例可以在 Chapter07/01_actions.py 中找到。这里，我将向你展示如何使用这些类：

```py
>>> import numpy as np 
>>> import ptan 
>>> q_vals = np.array([[1, 2, 3], [1, -1, 0]]) 
>>> q_vals 
array([[ 1,  2,  3], 
      [ 1, -1,  0]]) 
>>> selector = ptan.actions.ArgmaxActionSelector() 
>>> selector(q_vals) 
array([2, 0])
```

正如你所看到的，选择器返回具有最大值的动作的索引。

下一个动作选择器是 EpisilonGreedyActionSelector，它“包装”另一个动作选择器，并根据 epsilon 参数，使用包装的动作选择器或采取随机动作。这个动作选择器在训练过程中用于为代理的动作引入随机性。如果 epsilon 是 0.0，则不会采取随机动作：

```py
>>> selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=0.0, selector=ptan.actions.ArgmaxActionSelector()) 
>>> selector(q_vals) 
array([2, 0])
```

如果我们将 epsilon 更改为 1，动作将变为随机：

```py
>>> selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0) 
>>> selector(q_vals) 
array([0, 1])
```

你还可以通过为动作选择器的属性赋值来更改 epsilon 的值：

```py
>>> selector.epsilon 
1.0 
>>> selector.epsilon = 0.0 
>>> selector(q_vals) 
array([2, 0])
```

使用 ProbabilityActionSelector 的方法是一样的，但输入需要是一个归一化的概率分布：

```py
>>> selector = ptan.actions.ProbabilityActionSelector() 
>>> for _ in range(10): 
...    acts = selector(np.array([ 
...        [0.1, 0.8, 0.1], 
...        [0.0, 0.0, 1.0], 
...        [0.5, 0.5, 0.0] 
...    ])) 
...    print(acts) 
... 
[0 2 1] 
[1 2 1] 
[1 2 1] 
[0 2 1] 
[2 2 0] 
[0 2 0] 
[1 2 1] 
[1 2 0] 
[1 2 1] 
[1 2 0]
```

在前面的示例中，我们从三个分布中进行采样（因为我们在传入的矩阵中有三行）：

+   第一个由向量[0.1, 0.8, 0.1]定义；因此，索引为 1 的动作以 80%的概率被选中

+   向量[0.0, 0.0, 1.0]总是给我们动作 2 的索引

+   分布[0.5, 0.5, 0.0]以 50%的几率产生动作 0 和动作 1

## 代理

智能体实体提供了一种统一的方式，将来自环境的观察与我们想要执行的动作连接起来。到目前为止，你只看到了一个简单的无状态 DQN 智能体，该智能体使用神经网络（NN）从当前观察中获取动作值，并根据这些值贪婪地做出决策。我们使用 epsilon-贪婪策略来探索环境，但这并没有显著改变局面。

在强化学习（RL）领域，这可能会更加复杂。例如，我们的智能体可能不是预测动作的值，而是预测动作上的概率分布。这样的智能体被称为策略智能体，我们将在本书的第三部分讨论这些方法。

在某些情况下，智能体可能需要在观察之间保持状态。例如，通常情况下，单一的观察（甚至是最近的 k 个观察）不足以做出关于动作的决策，我们希望在智能体中保留一些记忆，以捕捉必要的信息。强化学习中有一个子领域试图通过部分可观察马尔可夫决策过程（POMDP）来解决这个问题，我们在第六章中简要提到过，但在本书中没有广泛覆盖。

智能体的第三种变体在连续控制问题中非常常见，这将在本书的第四部分讨论。目前，只需说在这种情况下，动作不再是离散的，而是连续的值，智能体需要根据观察来预测这些值。

为了捕捉所有这些变体并使代码具有灵活性，PTAN 中的智能体是通过一个可扩展的类层次结构实现的，ptan.agent.BaseAgent 抽象类位于顶部。从高层来看，智能体需要接受一批观察（以 NumPy 数组或 NumPy 数组列表的形式），并返回它想要执行的动作批次。批次的使用可以使处理更加高效，因为在图形处理单元（GPU）中一次性处理多个观察通常比逐个处理更快。

抽象基类没有定义输入和输出的类型，这使得它非常灵活且易于扩展。例如，在连续域中，我们的动作不再是离散动作的索引，而是浮动值。在任何情况下，智能体可以被视为一种知道如何将观察转换为动作的实体，如何做到这一点由智能体决定。通常，对于观察和动作类型没有假设，但智能体的具体实现则更具限制性。PTAN 提供了两种将观察转换为动作的常见方法：DQNAgent 和 PolicyAgent。我们将在后续章节中探讨这些方法。

然而，在实际问题中，通常需要定制的智能体。这些是一些原因：

+   神经网络的架构很复杂——它的动作空间是连续和离散的混合，并且它有多模态的观察（例如文本和像素），或者类似的东西。

+   你想使用非标准的探索策略，例如 Ornstein–Uhlenbeck 过程（在连续控制领域中是一种非常流行的探索策略）。

+   你有一个 POMDP 环境，智能体的决策不仅仅由观察定义，还由某些内部状态（这对于 Ornstein–Uhlenbeck 探索也是如此）决定。

所有这些情况都可以通过子类化 BaseAgent 类轻松支持，书中的后续部分将给出几个这样的重定义示例。

现在，让我们看看库中提供的标准智能体：DQNAgent 和 PolicyAgent。完整示例在 Chapter07/02_agents.py 中。

### DQNAgent

这个类适用于 Q-learning，当动作空间不是很大的时候，涵盖了 Atari 游戏和许多经典问题。这个表示方式不是普适的，后面书中会介绍如何处理这种情况。DQNAgent 接收一批观察数据作为输入（作为 NumPy 数组），将网络应用到这些数据上以获得 Q 值，然后使用提供的 ActionSelector 将 Q 值转换为动作的索引。

让我们考虑一个简单的例子。为了简化起见，我们的网络始终为输入批次产生相同的输出。

首先，我们定义神经网络类，它应该将观察转换为动作。在我们的例子中，它根本不使用神经网络，并始终产生相同的输出：

```py
class DQNNet(nn.Module): 
    def __init__(self, actions: int): 
        super(DQNNet, self).__init__() 
        self.actions = actions 

    def forward(self, x): 
        # we always produce diagonal tensor of shape 
        # (batch_size, actions) 
        return torch.eye(x.size()[0], self.actions)
```

一旦我们定义了模型类，就可以将其用作 DQN 模型：

```py
>>> net = DQNNet(actions=3) 
>>> net(torch.zeros(2, 10)) 
tensor([[1., 0., 0.], 
       [0., 1., 0.]])
```

我们从简单的 argmax 策略开始（该策略返回值最大的动作），因此智能体将始终返回与网络输出中对应的动作：

```py
>>> selector = ptan.actions.ArgmaxActionSelector() 
>>> agent = ptan.agent.DQNAgent(model=net, action_selector=selector) 
>>> agent(torch.zeros(2, 5)) 
(array([0, 1]), [None, None])
```

在输入中，给定了一批两条观察数据，每条包含五个值；在输出中，智能体返回了一个包含两个对象的元组：

+   一个数组，表示我们批次中要执行的动作。在我们的例子中，对于第一批样本是动作 0，第二批样本是动作 1。

+   一个包含智能体内部状态的列表。对于有状态的智能体，这个列表很有用，而在我们的例子中，它是一个包含 None 的列表。由于我们的智能体是无状态的，可以忽略它。

现在，让我们使智能体具备 epsilon-greedy 探索策略。为此，我们只需要传递一个不同的动作选择器：

```py
>>> selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0) 
>>> agent = ptan.agent.DQNAgent(model=net, action_selector=selector) 
>>> agent(torch.zeros(10, 5))[0] 
array([2, 0, 0, 0, 1, 2, 1, 2, 2, 1])
```

由于 epsilon 为 1.0，所有动作都会是随机的，与网络的输出无关。但是我们可以在训练过程中动态地更改 epsilon 的值，这在逐步降低 epsilon 时非常方便：

```py
>>> selector.epsilon = 0.5 
>>> agent(torch.zeros(10, 5))[0] 
array([0, 1, 2, 2, 0, 0, 1, 2, 0, 2]) 
>>> selector.epsilon = 0.1 
>>> agent(torch.zeros(10, 5))[0] 
array([0, 1, 2, 0, 0, 0, 0, 0, 0, 0])
```

### PolicyAgent

PolicyAgent 期望网络为离散的动作集生成一个策略分布。策略分布可以是 logits（未归一化的）或者归一化的分布。实际上，你应该始终使用 logits 以提高训练过程的数值稳定性。

让我们重新实现之前的例子，但这次网络将产生一个概率值。

我们首先定义以下类：

```py
class PolicyNet(nn.Module): 
    def __init__(self, actions: int): 
        super(PolicyNet, self).__init__() 
        self.actions = actions 

    def forward(self, x): 
        # Now we produce the tensor with first two actions 
        # having the same logit scores 
        shape = (x.size()[0], self.actions) 
        res = torch.zeros(shape, dtype=torch.float32) 
        res[:, 0] = 1 
        res[:, 1] = 1 
        return res
```

上述类可用于获取一批观察的动作对数：

```py
>>> net = PolicyNet(actions=5) 
>>> net(torch.zeros(6, 10)) 
tensor([[1., 1., 0., 0., 0.], 
       [1., 1., 0., 0., 0.], 
       [1., 1., 0., 0., 0.], 
       [1., 1., 0., 0., 0.], 
       [1., 1., 0., 0., 0.], 
       [1., 1., 0., 0., 0.]])
```

现在，我们可以将 PolicyAgent 与 ProbabilityActionSelector 结合使用。由于后者期望的是归一化的概率，我们需要要求 PolicyAgent 对网络的输出应用 softmax：

```py
>>> selector = ptan.actions.ProbabilityActionSelector() 
>>> agent = ptan.agent.PolicyAgent(model=net, action_selector=selector, apply_softmax=True) 
>>> agent(torch.zeros(6, 5))[0] 
array([2, 1, 2, 0, 2, 3])
```

请注意，softmax 操作会对零对数值产生非零概率，因此我们的代理仍然可以选择具有零对数值的动作。

```py
>>> torch.nn.functional.softmax(torch.tensor([1., 1., 0., 0., 0.])) 
tensor([0.3222, 0.3222, 0.1185, 0.1185, 0.1185])
```

## 经验源

前一节中描述的代理抽象使我们能够以通用的方式实现环境通信。这些通信通过应用代理的动作到 Gym 环境中，形成轨迹。

从高层次来看，经验源类获取代理实例和环境，并为你提供来自轨迹的逐步数据。这些类的功能包括：

+   支持同时与多个环境进行通信。这使得在一次处理一批观察时能够有效利用 GPU。

+   轨迹可以预处理并以方便的形式呈现以供进一步训练。例如，有一种实现子轨迹回滚并累积奖励的方式。对于 DQN 和 n 步 DQN 来说，这种预处理非常方便，因为我们不关心子轨迹中的个别中间步骤，所以可以省略它们。这样可以节省内存，并减少我们需要编写的代码量。

+   支持来自 Gymnasium 的向量化环境（AsyncVectorEnv 和 SyncVectorEnv 类）。我们将在第十七章讨论这个话题。

因此，经验源类充当了一个“魔法黑箱”，隐藏了环境交互和轨迹处理的复杂性，让库用户不必处理这些问题。但整体的 PTAN 哲学是灵活和可扩展的，所以如果你愿意，你可以子类化现有的类或根据需要实现自己的版本。

系统提供了三个类：

+   ExperienceSource：通过使用代理和环境集，它生成包含所有中间步骤的 n 步子轨迹。

+   ExperienceSourceFirstLast：这与 ExperienceSource 相同，但它仅保留第一步和最后一步的子轨迹，并在两者之间进行适当的奖励累积。对于 n 步 DQN 或优势演员-评论家（A2C）回滚，这可以节省大量内存。

+   ExperienceSourceRollouts：这遵循 Mnih 在关于 Atari 游戏的论文中描述的异步优势演员-评论家（A3C）回滚方案（我们将在第十二章讨论这个话题）。

所有的类都被编写得既高效地使用中央处理单元（CPU），也高效地使用内存，这对于玩具问题来说并不重要，但在下一章当我们进入 Atari 游戏时，涉及到需要存储和处理大量数据的问题，这一点就显得非常重要。

### 玩具环境

为了演示，我们将实现一个非常简单的 Gym 环境，具有一个小而可预测的观察状态，来展示`ExperienceSource`类如何工作。这个环境的观察值是整数，从 0 到 4，动作也是整数，奖励等于给定的动作。环境产生的所有回合总是有 10 个步骤：

```py
class ToyEnv(gym.Env): 
    def __init__(self): 
        super(ToyEnv, self).__init__() 
        self.observation_space = gym.spaces.Discrete(n=5) 
        self.action_space = gym.spaces.Discrete(n=3) 
        self.step_index = 0 

    def reset(self): 
        self.step_index = 0 
        return self.step_index, {} 

    def step(self, action: int): 
        is_done = self.step_index == 10 
        if is_done: 
            return self.step_index % self.observation_space.n, 0.0, is_done, False, {} 
        self.step_index += 1 
        return self.step_index % self.observation_space.n, float(action), \ 
            self.step_index == 10, False, {}
```

除了这个环境，我们还将使用一个代理，它会根据观察结果始终生成固定的动作：

```py
class DullAgent(ptan.agent.BaseAgent): 
    def __init__(self, action: int): 
        self.action = action 

    def __call__(self, observations: tt.List[int], state: tt.Optional[list] = None) -> \ 
            tt.Tuple[tt.List[int], tt.Optional[list]]: 
        return [self.action for _ in observations], state
```

这两个类都定义在`Chapter07/lib.py`模块中。现在我们已经定义了代理，接下来我们讨论它产生的数据。

### `ExperienceSource`类

我们将讨论的第一个类是`ptan.experience.ExperienceSource`，它生成给定长度的代理轨迹片段。实现会自动处理回合结束的情况（即环境中的`step()`方法返回`is_done=True`），并重置环境。构造函数接受多个参数：

+   将要使用的 Gym 环境。或者，也可以是环境列表。

+   代理实例。

+   `steps_count=2`：要生成的子轨迹的长度。

该类实例提供标准的 Python 迭代器接口，因此你可以直接迭代它以获取子轨迹：

```py
>>> from lib import * 
>>> env = ToyEnv() 
>>> agent = DullAgent(action=1) 
>>> exp_source = ptan.experience.ExperienceSource(env=env, agent=agent, steps_count=2) 
>>> for idx, exp in zip(range(3), exp_source): 
...    print(exp) 
... 
(Experience(state=0, action=1, reward=1.0, done_trunc=False), Experience(state=1, action=1, reward=1.0, done_trunc=False)) 
(Experience(state=1, action=1, reward=1.0, done_trunc=False), Experience(state=2, action=1, reward=1.0, done_trunc=False)) 
(Experience(state=2, action=1, reward=1.0, done_trunc=False), Experience(state=3, action=1, reward=1.0, done_trunc=False))
```

在每次迭代中，`ExperienceSource`返回代理在与环境交互时的一段轨迹。它看起来可能很简单，但我们的示例背后有几件事在发生：

1.  调用了环境中的`reset()`以获取初始状态。

1.  代理被要求从返回的状态中选择要执行的动作。

1.  调用`step()`方法以获得奖励和下一个状态。

1.  这个下一个状态被传递给代理，以供其执行下一个动作。

1.  返回了从一个状态到下一个状态的转移信息。

1.  如果环境返回回合结束标志，我们就会输出剩余的轨迹并重置环境以重新开始。

1.  在对经验源的迭代过程中，过程继续（从第 3 步开始）。

如果代理改变了它生成动作的方式（我们可以通过更新网络权重、减少 epsilon 或其他方法来实现），它将立即影响我们获得的经验轨迹。

`ExperienceSource`实例返回的元组的长度等于或小于构造时传入的`step_count`参数。在我们的例子中，我们要求的是两个步骤的子轨迹，因此元组的长度为 2 或 1（在回合结束时）。元组中的每个对象都是`ptan.experience.Experience`类的实例，这是一个包含以下字段的数据类：

+   `state`：我们在采取行动前观察到的状态

+   `action`：我们完成的动作

+   `reward`：我们从环境中获得的即时奖励

+   `done_trunc`：回合是否结束或被截断

如果回合结束，子轨迹将会更短，且底层环境会自动重置，因此我们无需担心这个问题，可以继续迭代：

```py
>>> for idx, exp in zip(range(15), exp_source): 
...    print(exp) 
... 
(Experience(state=0, action=1, reward=1.0, done_trunc=False), Experience(state=1, action=1, reward=1.0, done_trunc=False)) 
....... 
(Experience(state=3, action=1, reward=1.0, done_trunc=False), Experience(state=4, action=1, reward=1.0, done_trunc=True)) 
(Experience(state=4, action=1, reward=1.0, done_trunc=True),) 
(Experience(state=0, action=1, reward=1.0, done_trunc=False), Experience(state=1, action=1, reward=1.0, done_trunc=False)) 
(Experience(state=1, action=1, reward=1.0, done_trunc=False), Experience(state=2, action=1, reward=1.0, done_trunc=False))
```

我们可以向 ExperienceSource 请求任意长度的子轨迹：

```py
>>> exp_source = ptan.experience.ExperienceSource(env=env, agent=agent, steps_count=4) 
>>> next(iter(exp_source)) 
(Experience(state=0, action=1, reward=1.0, done_trunc=False), Experience(state=1, action=1, reward=1.0, done_trunc=False), Experience(state=2, action=1, reward=1.0, done_trunc=False), Experience(state=3, action=1, reward=1.0, done_trunc=False))
```

我们可以传递多个 gym.Env 实例。在这种情况下，它们将按轮流方式使用：

```py
>>> exp_source = ptan.experience.ExperienceSource(env=[ToyEnv(), ToyEnv()], agent=agent, steps_count=4) 
>>> for idx, exp in zip(range(5), exp_source): 
...    print(exp) 
... 
(Experience(state=0, action=1, reward=1.0, done_trunc=False), Experience(state=1, action=1, reward=1.0, done_trunc=False), Experience(state=2, action=1, reward=1.0, done_trunc=False), Experience(state=3, action=1, reward=1.0, done_trunc=False)) 
(Experience(state=0, action=1, reward=1.0, done_trunc=False), Experience(state=1, action=1, reward=1.0, done_trunc=False), Experience(state=2, action=1, reward=1.0, done_trunc=False), Experience(state=3, action=1, reward=1.0, done_trunc=False)) 
(Experience(state=1, action=1, reward=1.0, done_trunc=False), Experience(state=2, action=1, reward=1.0, done_trunc=False), Experience(state=3, action=1, reward=1.0, done_trunc=False), Experience(state=4, action=1, reward=1.0, done_trunc=False)) 
(Experience(state=1, action=1, reward=1.0, done_trunc=False), Experience(state=2, action=1, reward=1.0, done_trunc=False), Experience(state=3, action=1, reward=1.0, done_trunc=False), Experience(state=4, action=1, reward=1.0, done_trunc=False)) 
(Experience(state=2, action=1, reward=1.0, done_trunc=False), Experience(state=3, action=1, reward=1.0, done_trunc=False), Experience(state=4, action=1, reward=1.0, done_trunc=False), Experience(state=0, action=1, reward=1.0, done_trunc=False))
```

请注意，当你将多个环境传递给 ExperienceSource 时，它们必须是独立的实例，而不是单一环境实例，否则你的观察将变得混乱。

### ExperienceSourceFirstLast 类

ExperienceSource 类为我们提供了指定长度的完整子轨迹，作为 (s, a, r) 对象的列表。下一个状态 s′ 会在下一个元组中返回，这有时不太方便。例如，在 DQN 训练中，我们希望一次性获得 (s, a, r, s′) 元组，以便在训练过程中进行一步 Bellman 近似。此外，DQN 的一些扩展，如 n 步 DQN，可能希望将更长的观察序列合并为 (first-state, action, total-reward-for-n-steps, state-after-step-n)。

为了以通用的方式支持这一点，已经实现了一个 ExperienceSource 的简单子类：ExperienceSourceFirstLast。它在构造函数中接受几乎相同的参数，但返回不同的数据：

```py
>>> exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=1.0, steps_count=1) 
>>> for idx, exp in zip(range(11), exp_source): 
...    print(exp) 
... 
ExperienceFirstLast(state=0, action=1, reward=1.0, last_state=1) 
ExperienceFirstLast(state=1, action=1, reward=1.0, last_state=2) 
ExperienceFirstLast(state=2, action=1, reward=1.0, last_state=3) 
ExperienceFirstLast(state=3, action=1, reward=1.0, last_state=4) 
ExperienceFirstLast(state=4, action=1, reward=1.0, last_state=0) 
ExperienceFirstLast(state=0, action=1, reward=1.0, last_state=1) 
ExperienceFirstLast(state=1, action=1, reward=1.0, last_state=2) 
ExperienceFirstLast(state=2, action=1, reward=1.0, last_state=3) 
ExperienceFirstLast(state=3, action=1, reward=1.0, last_state=4) 
ExperienceFirstLast(state=4, action=1, reward=1.0, last_state=None) 
ExperienceFirstLast(state=0, action=1, reward=1.0, last_state=1)
```

现在，它不再返回元组，而是每次迭代返回一个单一的对象，这个对象也是一个数据类，包含以下字段：

+   state: 我们用来决定采取什么动作的状态。

+   action: 我们在这一步骤采取的动作。

+   reward: 对于 steps_count（在我们的案例中，steps_count=1，因此它等于即时奖励）的部分累计奖励。

+   last_state: 执行动作后得到的状态。如果我们的回合结束，这里是 None。

这些数据对于 DQN 训练更为方便，因为我们可以直接应用 Bellman 近似。

让我们检查一下使用更多步数时的结果：

```py
>>> exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=1.0, steps_count=2) 
>>> for idx, exp in zip(range(11), exp_source): 
...    print(exp) 
... 
ExperienceFirstLast(state=0, action=1, reward=2.0, last_state=2) 
ExperienceFirstLast(state=1, action=1, reward=2.0, last_state=3) 
ExperienceFirstLast(state=2, action=1, reward=2.0, last_state=4) 
ExperienceFirstLast(state=3, action=1, reward=2.0, last_state=0) 
ExperienceFirstLast(state=4, action=1, reward=2.0, last_state=1) 
ExperienceFirstLast(state=0, action=1, reward=2.0, last_state=2) 
ExperienceFirstLast(state=1, action=1, reward=2.0, last_state=3) 
ExperienceFirstLast(state=2, action=1, reward=2.0, last_state=4) 
ExperienceFirstLast(state=3, action=1, reward=2.0, last_state=None) 
ExperienceFirstLast(state=4, action=1, reward=1.0, last_state=None) 
ExperienceFirstLast(state=0, action=1, reward=2.0, last_state=2)
```

所以，现在我们在每次迭代中合并了两步，并计算即时奖励（这就是为什么大多数样本的 reward=2.0）。回合结束时更有趣的样本：

```py
ExperienceFirstLast(state=3, action=1, reward=2.0, last_state=None) 
ExperienceFirstLast(state=4, action=1, reward=1.0, last_state=None)
```

当回合结束时，我们在这些样本中将 last_state=None，但此外，我们会计算回合尾部的奖励。如果你自己处理所有的轨迹，这些细节非常容易出错。

## 经验回放缓存

在 DQN 中，我们很少处理即时经验样本，因为它们高度相关，这会导致训练的不稳定。通常，我们有一个大的回放缓存，它充满了经验片段。然后从缓存中进行采样（随机或带优先级权重），以获取训练批次。回放缓存通常有一个最大容量，因此当回放缓存达到上限时，旧样本会被推送出去。

这里有几个实现技巧，当你需要处理大型问题时，这些技巧非常重要：

+   如何高效地从大缓存中采样

+   如何从缓存中推送旧样本

+   在优先级缓存的情况下，如何以最有效的方式维护和处理优先级

如果你想处理 Atari 游戏，保持 10-100M 样本，其中每个样本都是游戏中的一张图片，这一切就变成了一项相当复杂的任务。一个小错误可能导致 10-100 倍的内存增加，并且会严重拖慢训练过程。

PTAN 提供了几种重放缓冲区的变体，它们可以与 ExperienceSource 和 Agent 架构轻松集成。通常，您需要做的是请求缓冲区从源中提取一个新样本并采样训练批次。提供的类包括：

+   ExperienceReplayBuffer：一个简单的、大小预定义的重放缓冲区，采用均匀采样。

+   PrioReplayBufferNaive：一种简单但效率不高的优先级重放缓冲区实现。采样复杂度为 O(n)，对于大缓冲区来说可能成为一个问题。这个版本相比优化后的类，代码更简单。对于中等大小的缓冲区，性能仍然可以接受，因此我们会在一些示例中使用它。

+   PrioritizedReplayBuffer：使用线段树进行采样，这使得代码变得晦涩，但采样复杂度为 O(log(n))。

以下展示了如何使用重放缓冲区：

```py
>>> exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=1.0, steps_count=1) 
>>> buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=100) 
>>> len(buffer) 
0 
>>> buffer.populate(1) 
>>> len(buffer) 
1
```

所有重放缓冲区提供以下接口：

+   一个 Python 迭代器接口，用于遍历缓冲区中的所有样本。

+   populate(N) 方法用于从经验源中获取 N 个样本并将它们放入缓冲区。

+   方法 sample(N) 用于获取 N 个经验对象的批次。

因此，DQN 的正常训练循环看起来像是以下步骤的无限重复：

1.  调用 buffer.populate(1) 从环境中获取一个新样本。

1.  调用 batch = buffer.sample(BATCH_SIZE) 从缓冲区中获取批次。

1.  计算所采样批次的损失。

1.  反向传播。

1.  重复直到收敛（希望如此）。

其余的过程自动完成——重置环境、处理子轨迹、维护缓冲区大小等：

```py
>>> for step in range(6): 
...    buffer.populate(1) 
...    if len(buffer) < 5: 
...        continue 
...    batch = buffer.sample(4) 
...    print(f"Train time, {len(batch)} batch samples") 
...    for s in batch: 
...        print(s) 
... 
Train time, 4 batch samples 
ExperienceFirstLast(state=1, action=1, reward=1.0, last_state=2) 
ExperienceFirstLast(state=2, action=1, reward=1.0, last_state=3) 
ExperienceFirstLast(state=2, action=1, reward=1.0, last_state=3) 
ExperienceFirstLast(state=0, action=1, reward=1.0, last_state=1) 
Train time, 4 batch samples 
ExperienceFirstLast(state=0, action=1, reward=1.0, last_state=1) 
ExperienceFirstLast(state=4, action=1, reward=1.0, last_state=0) 
ExperienceFirstLast(state=3, action=1, reward=1.0, last_state=4) 
ExperienceFirstLast(state=3, action=1, reward=1.0, last_state=4)
```

## TargetNet 类

我们在前一章中提到过自举问题，当用于下一个状态评估的网络受到我们训练过程的影响时，这一问题会出现。通过将当前训练的网络与用于预测下一个状态 Q 值的网络分离，解决了这个问题。

TargetNet 是一个小巧但有用的类，允许我们同步两个相同架构的神经网络。此类支持两种同步模式：

+   sync()：源网络的权重被复制到目标网络中。

+   alpha_sync()：源网络的权重通过一个 alpha 权重（介于 0 和 1 之间）融合到目标网络中。

第一个模式是执行离散动作空间问题（如 Atari 和 CartPole）中的目标网络同步的标准方法，正如我们在第六章中所做的那样。后者模式用于连续控制问题，这将在本书第四部分中描述。在此类问题中，两个网络参数之间的过渡应是平滑的，因此使用了 alpha 混合，公式为 w[i] = w[i]α + si，其中 w[i] 是目标网络的第 i 个参数，s[i] 是源网络的权重。以下是如何在代码中使用 TargetNet 的一个小示例。假设我们有以下网络：

```py
class DQNNet(nn.Module): 
    def __init__(self): 
        super(DQNNet, self).__init__() 
        self.ff = nn.Linear(5, 3) 
    def forward(self, x): 
        return self.ff(x)
```

目标网络可以通过以下方式创建：

```py
>>> net = DQNNet() 
>>> net 
DQNNet( 
  (ff): Linear(in_features=5, out_features=3, bias=True) 
) 
>>> tgt_net = ptan.agent.TargetNet(net)
```

目标网络包含两个字段：model，它是对原始网络的引用；target_model，它是原始网络的深拷贝。如果我们检查这两个网络的权重，它们将是相同的：

```py
>>> net.ff.weight 
Parameter containing: 
tensor([[ 0.2039,  0.1487,  0.4420, -0.0210, -0.2726], 
       [-0.2020, -0.0787,  0.2852, -0.1565,  0.4012], 
       [-0.0569, -0.4184, -0.3658,  0.4212,  0.3647]], requires_grad=True) 
>>> tgt_net.target_model.ff.weight 
Parameter containing: 
tensor([[ 0.2039,  0.1487,  0.4420, -0.0210, -0.2726], 
       [-0.2020, -0.0787,  0.2852, -0.1565,  0.4012], 
       [-0.0569, -0.4184, -0.3658,  0.4212,  0.3647]], requires_grad=True)
```

它们相互独立，然而，仅仅有相同的架构：

```py
>>> net.ff.weight.data += 1.0 
>>> net.ff.weight 
Parameter containing: 
tensor([[1.2039, 1.1487, 1.4420, 0.9790, 0.7274], 
       [0.7980, 0.9213, 1.2852, 0.8435, 1.4012], 
       [0.9431, 0.5816, 0.6342, 1.4212, 1.3647]], requires_grad=True) 
>>> tgt_net.target_model.ff.weight 
Parameter containing: 
tensor([[ 0.2039,  0.1487,  0.4420, -0.0210, -0.2726], 
       [-0.2020, -0.0787,  0.2852, -0.1565,  0.4012], 
       [-0.0569, -0.4184, -0.3658,  0.4212,  0.3647]], requires_grad=True)
```

要再次同步它们，可以使用 sync() 方法：

```py
>>> tgt_net.sync() 
>>> tgt_net.target_model.ff.weight 
Parameter containing: 
tensor([[1.2039, 1.1487, 1.4420, 0.9790, 0.7274], 
       [0.7980, 0.9213, 1.2852, 0.8435, 1.4012], 
       [0.9431, 0.5816, 0.6342, 1.4212, 1.3647]], requires_grad=True)
```

对于混合同步，你可以使用 alpha_sync() 方法：

```py
>>> net.ff.weight.data += 1.0 
>>> net.ff.weight 
Parameter containing: 
tensor([[2.2039, 2.1487, 2.4420, 1.9790, 1.7274], 
       [1.7980, 1.9213, 2.2852, 1.8435, 2.4012], 
       [1.9431, 1.5816, 1.6342, 2.4212, 2.3647]], requires_grad=True) 
>>> tgt_net.target_model.ff.weight 
Parameter containing: 
tensor([[1.2039, 1.1487, 1.4420, 0.9790, 0.7274], 
       [0.7980, 0.9213, 1.2852, 0.8435, 1.4012], 
       [0.9431, 0.5816, 0.6342, 1.4212, 1.3647]], requires_grad=True) 
>>> tgt_net.alpha_sync(0.1) 
>>> tgt_net.target_model.ff.weight 
Parameter containing: 
tensor([[2.1039, 2.0487, 2.3420, 1.8790, 1.6274], 
       [1.6980, 1.8213, 2.1852, 1.7435, 2.3012], 
       [1.8431, 1.4816, 1.5342, 2.3212, 2.2647]], requires_grad=True)
```

## Ignite 辅助工具

PyTorch Ignite 在第三章中简要讨论过，之后在本书的其余部分将用于减少训练循环代码的量。PTAN 提供了几个小的辅助工具，以简化与 Ignite 的集成，这些工具位于 ptan.ignite 包中：

+   EndOfEpisodeHandler：附加到 ignite.Engine，触发 EPISODE_COMPLETED 事件，并在事件中跟踪奖励和步骤数，记录到引擎的指标中。它还可以在最后几集的平均奖励达到预定义边界时触发事件，预定用于在达到某个目标奖励时停止训练。

+   EpisodeFPSHandler：跟踪代理与环境之间执行的交互次数，并计算每秒帧数的性能指标。它还跟踪从训练开始到现在经过的秒数。

+   PeriodicEvents：每 10、100 或 1,000 次训练迭代时触发相应事件。它有助于减少写入 TensorBoard 的数据量。

在下一章中将详细说明如何使用这些类，当时我们将用它们重新实现第六章中的 DQN 训练，然后检查几个 DQN 扩展和调整，以提高基础 DQN 的收敛性。

# PTAN CartPole 解算器

现在我们来看看 PTAN 类（目前没有 Ignite），并尝试将所有内容结合起来解决我们的第一个环境：CartPole。完整的代码位于 Chapter07/06_cartpole.py。这里只展示与我们刚刚讨论的材料相关的重要部分代码。

首先，我们创建神经网络（之前用于 CartPole 的简单两层前馈神经网络）并将其目标设为 NN epsilon-greedy 动作选择器和 DQNAgent。接着，创建经验源和回放缓冲区：

```py
 net = Net(obs_size, HIDDEN_SIZE, n_actions) 
    tgt_net = ptan.agent.TargetNet(net) 
    selector = ptan.actions.ArgmaxActionSelector() 
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1, selector=selector) 
    agent = ptan.agent.DQNAgent(net, selector) 
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA) 
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
```

通过这几行代码，我们已经完成了数据管道的创建。

现在，我们只需要在缓冲区上调用`populate()`并从中采样训练批次：

```py
 while True: 
        step += 1 
        buffer.populate(1) 

        for reward, steps in exp_source.pop_rewards_steps(): 
            episode += 1 
            print(f"{step}: episode {episode} done, reward={reward:.2f}, " 
                  f"epsilon={selector.epsilon:.2f}") 
            solved = reward > 150 
        if solved: 
            print("Whee!") 
            break 
        if len(buffer) < 2*BATCH_SIZE: 
            continue 
        batch = buffer.sample(BATCH_SIZE)
```

在每次训练循环的开始，我们要求缓冲区从经验源中获取一个样本，然后检查是否有已完成的 episode。`pop_rewards_steps()`方法在`ExperienceSource`类中返回一个元组列表，其中包含自上次调用该方法以来完成的 episodes 信息。

在训练循环后期，我们将一批`ExperienceFirstLast`对象转换为适合 DQN 训练的张量：

```py
 batch = buffer.sample(BATCH_SIZE) 
        states_v, actions_v, tgt_q_v = unpack_batch(batch, tgt_net.target_model, GAMMA) 
        optimizer.zero_grad() 
        q_v = net(states_v) 
        q_v = q_v.gather(1, actions_v.unsqueeze(-1)).squeeze(-1) 
        loss_v = F.mse_loss(q_v, tgt_q_v) 
        loss_v.backward() 
        optimizer.step() 
        selector.epsilon *= EPS_DECAY 

        if step % TGT_NET_SYNC == 0: 
            tgt_net.sync()
```

我们计算损失并进行一次反向传播步骤。最后，我们在我们的动作选择器中衰减 epsilon（使用的超参数使得 epsilon 在第 500 步训练时衰减为零），并要求目标网络每 10 次训练迭代同步一次。

`unpack_batch`方法是我们实现的最后一部分：

```py
@torch.no_grad() 
def unpack_batch(batch: tt.List[ExperienceFirstLast], net: Net, gamma: float): 
    states = [] 
    actions = [] 
    rewards = [] 
    done_masks = [] 
    last_states = [] 
    for exp in batch: 
        states.append(exp.state) 
        actions.append(exp.action) 
        rewards.append(exp.reward) 
        done_masks.append(exp.last_state is None) 
        if exp.last_state is None: 
            last_states.append(exp.state) 
        else: 
            last_states.append(exp.last_state) 

    states_v = torch.as_tensor(np.stack(states)) 
    actions_v = torch.tensor(actions) 
    rewards_v = torch.tensor(rewards) 
    last_states_v = torch.as_tensor(np.stack(last_states)) 
    last_state_q_v = net(last_states_v) 
    best_last_q_v = torch.max(last_state_q_v, dim=1)[0] 
    best_last_q_v[done_masks] = 0.0 
    return states_v, actions_v, best_last_q_v * gamma + rewards_v
```

它接受一批采样的`ExperienceFirstLast`对象，并将它们转换为三个张量：状态、动作和目标 Q 值。代码应该在 2000 到 3000 次训练迭代中收敛：

```py
Chapter07$ python 06_cartpole.py 
26: episode 1 done, reward=25.00, epsilon=1.00 
52: episode 2 done, reward=26.00, epsilon=0.82 
67: episode 3 done, reward=15.00, epsilon=0.70 
80: episode 4 done, reward=13.00, epsilon=0.62 
112: episode 5 done, reward=32.00, epsilon=0.45 
123: episode 6 done, reward=11.00, epsilon=0.40 
139: episode 7 done, reward=16.00, epsilon=0.34 
148: episode 8 done, reward=9.00, epsilon=0.31 
156: episode 9 done, reward=8.00, epsilon=0.29 
... 
2481: episode 113 done, reward=58.00, epsilon=0.00 
2544: episode 114 done, reward=63.00, epsilon=0.00 
2594: episode 115 done, reward=50.00, epsilon=0.00 
2786: episode 116 done, reward=192.00, epsilon=0.00 
Whee!
```

# 其他的 RL 库

正如我们之前讨论的，市面上有几种专门用于 RL 的库。几年前，TensorFlow 比 PyTorch 更流行，但如今，PyTorch 在该领域占据主导地位，并且最近 JAX 的使用趋势正在上升，因为它提供了更好的性能。以下是我推荐的一些你可能想要考虑在项目中使用的库：

+   stable-baselines3：我们在讨论 Atari 包装器时提到了这个库。它是 OpenAI Stable Baselines 库的一个分支，主要目的是提供一个经过优化且可复现的 RL 算法集，你可以用它来验证你的方法（[`github.com/DLR-RM/stable-baselines3`](https://github.com/DLR-RM/stable-baselines3)）。

+   TorchRL：PyTorch 的 RL 扩展。这个库相对较新——它的第一个版本发布于 2022 年底——但提供了丰富的 RL 帮助类。它的设计理念与 PTAN 非常接近——一个以 Python 为主的灵活类集合，你可以将它们组合和扩展来构建你的系统——所以我强烈推荐你学习这个库。在本书的剩余部分，我们将使用这个库的类。很可能，本书下一版的示例（除非我们迎来了“人工智能奇点”，书籍变得像粘土板一样过时）将不再基于 PTAN，而是基于 TorchRL，它维护得更好。文档：[`pytorch.org/rl/`](https://pytorch.org/rl/)，源代码：[`github.com/pytorch/rl`](https://github.com/pytorch/rl)。

+   Spinning Up：这是 OpenAI 的另一个库，但目标不同：提供关于最先进方法的有价值且简洁的教育材料。这个库已经有几年没有更新了（最后的提交是在 2020 年），但仍然提供了关于这些方法的非常有价值的材料。文档：[`spinningup.openai.com/`](https://spinningup.openai.com/)。代码：[`github.com/openai/spinningup`](https://github.com/openai/spinningup)。

+   Keras-RL：由 Matthias Plappert 于 2016 年启动，包含基本的深度强化学习方法。正如名称所示，该库是使用 Keras 实现的，Keras 是一个高层次的 TensorFlow 封装器（[`github.com/keras-rl/keras-rl`](https://github.com/keras-rl/keras-rl)）。不幸的是，最后一次提交是在 2019 年，因此该项目已被废弃。

+   Dopamine：谷歌于 2018 年发布的库。它是 TensorFlow 特定的，这对于谷歌发布的库来说并不令人惊讶（[`github.com/google/dopamine`](https://github.com/google/dopamine)）。

+   Ray：一个用于分布式执行机器学习代码的库。它包含作为库一部分的强化学习工具（[`github.com/ray-project/ray`](https://github.com/ray-project/ray)）。

+   TF-Agents：谷歌于 2018 年发布的另一个库（[`github.com/tensorflow/agents`](https://github.com/tensorflow/agents)）。

+   ReAgent：来自 Facebook Research 的库。它内部使用 PyTorch，并采用声明式配置风格（当你创建 JSON 文件来描述问题时），这限制了可扩展性。但当然，由于它是开源的，你总是可以扩展功能（[`github.com/facebookresearch/ReAgent`](https://github.com/facebookresearch/ReAgent)）。最近，ReAgent 已经被归档，并由同一团队的 Pearl 库所替代：[`github.com/facebookresearch/Pearl/`](https://github.com/facebookresearch/Pearl/)。

# 总结

在这一章中，我们讨论了更高层次的强化学习库、它们的动机和要求。接着，我们深入了解了 PTAN 库，它将在本书的其余部分中用于简化示例代码。专注于方法的细节而非实现，这对于你在本书后续章节学习强化学习时会非常有帮助。

在下一章，我们将通过探索研究人员和实践者自经典 DQN 方法引入以来，为了提高方法的稳定性和性能所发现的扩展，重新回到 DQN 方法。
