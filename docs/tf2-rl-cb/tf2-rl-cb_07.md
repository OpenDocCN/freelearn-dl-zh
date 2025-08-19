# 第七章：*第七章*：将深度 RL 代理部署到云端

云计算已成为基于 AI 的产品和解决方案的*事实上的*部署平台。深度学习模型在云端运行变得越来越普遍。然而，由于各种原因，将基于强化学习的代理部署到云端仍然非常有限。本章包含了帮助你掌握工具和细节的配方，让你走在前沿，构建基于深度 RL 的云端 Simulation-as-a-Service 和 Agent/Bot-as-a-Service 应用。

本章具体讨论了以下配方：

+   实现 RL 代理的运行时组件

+   构建作为服务的 RL 环境模拟器

+   使用远程模拟器服务训练 RL 代理

+   测试/评估 RL 代理

+   打包 RL 代理以进行部署 – 一个交易机器人

+   将 RL 代理部署到云端 – 一个交易机器人即服务

# 技术要求

本书中的代码在 Ubuntu 18.04 和 Ubuntu 20.04 上经过广泛测试，如果安装了 Python 3.6+，该代码应该能在更新版本的 Ubuntu 上运行。如果安装了 Python 3.6+ 和前面列出的必要 Python 包，代码也应该能在 Windows 和 macOS X 上运行。建议创建并使用名为 `tf2rl-cookbook` 的 Python 虚拟环境来安装包并运行本书中的代码。建议使用 Miniconda 或 Anaconda 进行 Python 虚拟环境管理。

每章每个配方的完整代码可以在这里找到：[`github.com/PacktPublishing/Tensorflow-2-Reinforcement-Learning-Cookbook`](https://github.com/PacktPublishing/Tensorflow-2-Reinforcement-Learning-Cookbook)。

# 实现 RL 代理的运行时组件

在前几章中，我们已经讨论了几个代理算法的实现。你可能已经注意到，在之前的章节（特别是 *第三章*，*实现高级深度 RL 算法*）中的一些配方里，我们实现了 RL 代理的训练代码，其中有些部分的代理代码是有条件执行的。例如，经验回放的例程只有在满足某些条件（比如回放记忆中的样本数量）时才会运行，等等。这引出了一个问题：在一个代理中，哪些组件是必需的，特别是当我们不打算继续训练它，而只是执行一个已经学到的策略时？

本配方将帮助你将 **Soft Actor-Critic** (**SAC**) 代理的实现提炼到一组最小的组件——这些是你的代理运行时绝对必要的组件。

让我们开始吧！

## 做好准备

为了完成这个食谱，首先需要激活`tf2rl-cookbook`的 Python/conda 虚拟环境。确保更新环境以匹配食谱代码库中的最新 conda 环境规格文件（`tfrl-cookbook.yml`）。WebGym 建立在 miniWob-plusplus 基准之上（[`github.com/stanfordnlp/miniwob-plusplus`](https://github.com/stanfordnlp/miniwob-plusplus)），该基准也作为本书代码库的一部分提供，便于使用。如果以下的`import`语句没有问题，你就可以开始了：

```py
import functools
from collections import deque
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Concatenate, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
tf.keras.backend.set_floatx(“float64”)
```

现在，让我们开始吧！

## 如何做到这一点…

以下步骤提供了实现 SAC 智能体所需的最小运行时的详细信息。让我们直接进入细节：

1.  首先，让我们实现演员组件，它将是一个 TensorFlow 2.x 模型：

    ```py
    def actor(state_shape, action_shape, units=(512, 256, 64)):
        state_shape_flattened = \
            functools.reduce(lambda x, y: x * y, state_shape)
        state = Input(shape=state_shape_flattened)
        x = Dense(units[0], name=”L0”, activation=”relu”)\
                 (state)
        for index in range(1, len(units)):
            x = Dense(units[index], name=”L{}”.format(index),
                      activation=”relu”)(x)
        actions_mean = Dense(action_shape[0], \
                             name=”Out_mean”)(x)
        actions_std = Dense(action_shape[0], 
                             name=”Out_std”)(x)
        model = Model(inputs=state, outputs=[actions_mean,
                      actions_std])
        return model
    ```

1.  接下来，让我们实现评论家组件，它也将是一个 TensorFlow 2.x 模型：

    ```py
    def critic(state_shape, action_shape, units=(512, 256, 64)):
        state_shape_flattened = \
            functools.reduce(lambda x, y: x * y, state_shape)
        inputs = [Input(shape=state_shape_flattened), \
                  Input(shape=action_shape)]
        concat = Concatenate(axis=-1)(inputs)
        x = Dense(units[0], name=”Hidden0”, \
                  activation=”relu”)(concat)
        for index in range(1, len(units)):
            x = Dense(units[index], \
                      name=”Hidden{}”.format(index), \
                      activation=”relu”)(x)
        output = Dense(1, name=”Out_QVal”)(x)
        model = Model(inputs=inputs, outputs=output)
        return model
    ```

1.  现在，让我们实现一个实用函数，用于在给定源 TensorFlow 2.x 模型的情况下更新目标模型的权重：

    ```py
    def update_target_weights(model, target_model, tau=0.005):
        weights = model.get_weights()
        target_weights = target_model.get_weights()
        for i in range(len(target_weights)):  
        # set tau% of target model to be new weights
            target_weights[i] = weights[i] * tau + \
                                target_weights[i] * (1 - tau)
        target_model.set_weights(target_weights)
    ```

1.  现在，我们可以开始实现 SAC 智能体的运行时类。我们将把实现分为以下几个步骤。让我们从类的实现开始，并在这一步中定义构造函数的参数：

    ```py
    class SAC(object):
        def __init__(
            self,
            observation_shape,
            action_space,
            lr_actor=3e-5,
            lr_critic=3e-4,
            actor_units=(64, 64),
            critic_units=(64, 64),
            auto_alpha=True,
            alpha=0.2,
            tau=0.005,
            gamma=0.99,
            batch_size=128,
            memory_cap=100000,
        ):
    ```

1.  现在，让我们初始化智能体的状态/观测形状、动作形状、动作限制/边界，并初始化一个双端队列（deque）来存储智能体的记忆：

    ```py
            self.state_shape = observation_shape  # shape of 
            # observations
            self.action_shape = action_space.shape  # number 
            # of actions
            self.action_bound = \
                (action_space.high - action_space.low) / 2
            self.action_shift = \
                (action_space.high + action_space.low) / 2
            self.memory = deque(maxlen=int(memory_cap))
    ```

1.  在这一步中，让我们定义并初始化演员组件：

    ```py
            # Define and initialize actor network
            self.actor = actor(self.state_shape, 
                               self.action_shape, 
                               actor_units)
            self.actor_optimizer = \
                Adam(learning_rate=lr_actor)
            self.log_std_min = -20
            self.log_std_max = 2
            print(self.actor.summary())
    ```

1.  现在，让我们定义并初始化评论家组件：

    ```py
            # Define and initialize critic networks
            self.critic_1 = critic(self.state_shape, 
                                   self.action_shape, 
                                   critic_units)
            self.critic_target_1 = critic(self.state_shape,
                                          self.action_shape,
                                          critic_units)
            self.critic_optimizer_1 = \
                Adam(learning_rate=lr_critic)
            update_target_weights(self.critic_1, 
                                  self.critic_target_1, 
                                  tau=1.0)
            self.critic_2 = critic(self.state_shape, 
                                   self.action_shape, 
                                   critic_units)
            self.critic_target_2 = critic(self.state_shape,
                                          self.action_shape,
                                          critic_units)
            self.critic_optimizer_2 = \
                Adam(learning_rate=lr_critic)
            update_target_weights(self.critic_2, 
                                  self.critic_target_2, 
                                  tau=1.0)
            print(self.critic_1.summary())
    ```

1.  在这一步中，让我们根据`auto_alpha`标志来初始化 SAC 智能体的温度和目标熵：

    ```py
            # Define and initialize temperature alpha and 
            # target entropy
            self.auto_alpha = auto_alpha
            if auto_alpha:
                self.target_entropy = \
                    -np.prod(self.action_shape)
                self.log_alpha = tf.Variable(0.0, 
                                            dtype=tf.float64)
                self.alpha = tf.Variable(0.0, 
                                         dtype=tf.float64)
                self.alpha.assign(tf.exp(self.log_alpha))
                self.alpha_optimizer = \
                    Adam(learning_rate=lr_actor)
            else:
                self.alpha = tf.Variable(alpha, 
                                         dtype=tf.float64)
    ```

1.  让我们通过设置超参数并初始化用于 TensorBoard 日志记录的训练进度摘要字典来完成构造函数的实现：

    ```py
            # Set hyperparameters
            self.gamma = gamma  # discount factor
            self.tau = tau  # target model update
            self.batch_size = batch_size
            # Tensorboard
            self.summaries = {}
    ```

1.  构造函数实现完成后，我们接下来实现`process_action`函数，该函数接收智能体的原始动作并处理它，使其可以被执行：

    ```py
        def process_actions(self, mean, log_std, test=False, 
        eps=1e-6):
            std = tf.math.exp(log_std)
            raw_actions = mean
            if not test:
                raw_actions += tf.random.normal(shape=mean.\
                               shape, dtype=tf.float64) * std
            log_prob_u = tfp.distributions.Normal(loc=mean,
                             scale=std).log_prob(raw_actions)
            actions = tf.math.tanh(raw_actions)
            log_prob = tf.reduce_sum(log_prob_u - \
                         tf.math.log(1 - actions ** 2 + eps))
            actions = actions * self.action_bound + \
                      self.action_shift
            return actions, log_prob
    ```

1.  这一步非常关键。我们将实现`act`方法，该方法将以状态作为输入，生成并返回要执行的动作：

    ```py
        def act(self, state, test=False, use_random=False):
            state = state.reshape(-1)  # Flatten state
            state = np.expand_dims(state, axis=0).\
                                         astype(np.float64)
            if use_random:
                a = tf.random.uniform(
                    shape=(1, self.action_shape[0]), 
                           minval=-1, maxval=1, 
                           dtype=tf.float64
                )
            else:
                means, log_stds = self.actor.predict(state)
                log_stds = tf.clip_by_value(log_stds, 
                                            self.log_std_min,
                                            self.log_std_max)
                a, log_prob = self.process_actions(means,
                                                   log_stds, 
                                                   test=test)
            q1 = self.critic_1.predict([state, a])[0][0]
            q2 = self.critic_2.predict([state, a])[0][0]
            self.summaries[“q_min”] = tf.math.minimum(q1, q2)
            self.summaries[“q_mean”] = np.mean([q1, q2])
            return a
    ```

1.  最后，让我们实现一些实用方法，用于从先前训练的模型中加载演员和评论家的模型权重：

    ```py
        def load_actor(self, a_fn):
            self.actor.load_weights(a_fn)
            print(self.actor.summary())
        def load_critic(self, c_fn):
            self.critic_1.load_weights(c_fn)
            self.critic_target_1.load_weights(c_fn)
            self.critic_2.load_weights(c_fn)
            self.critic_target_2.load_weights(c_fn)
            print(self.critic_1.summary())
    ```

到此为止，我们已经完成了所有必要的 SAC RL 智能体运行时组件的实现！

## 它是如何工作的…

在这个食谱中，我们实现了 SAC 智能体的基本运行时组件。运行时组件包括演员和评论家模型定义、一个从先前训练的智能体模型中加载权重的机制，以及一个智能体接口，用于根据状态生成动作，利用演员的预测并处理预测生成可执行动作。

对于其他基于演员-评论家的 RL 智能体算法，如 A2C、A3C 和 DDPG 及其扩展和变体，运行时组件将非常相似，甚至可能是相同的。

现在是时候进入下一个教程了！

# 将 RL 环境模拟器构建为服务

本教程将引导你将你的 RL 训练环境/模拟器转换为一个服务。这将使你能够提供模拟即服务（Simulation-as-a-Service）来训练 RL 代理！

到目前为止，我们已经在多种环境中使用不同的模拟器训练了多个 RL 代理，具体取决于要解决的任务。训练脚本使用 OpenAI Gym 接口与在同一进程中运行的环境或在不同进程中本地运行的环境进行通信。本教程将引导你完成将任何 OpenAI Gym 兼容的训练环境（包括你自定义的 RL 训练环境）转换为可以本地或远程部署为服务的过程。构建并部署完成后，代理训练客户端可以连接到模拟服务器，并远程训练一个或多个代理。

作为一个具体的例子，我们将使用我们的 `tradegym` 库，它是我们在前几章中为加密货币和股票交易构建的 RL 训练环境的集合，并通过 **RESTful HTTP 接口** 将它们暴露出来，以便训练 RL 代理。

开始吧！

## 准备工作

要完成本教程，你需要首先激活 `tf2rl-cookbook` Python/conda 虚拟环境。确保更新该环境，使其与最新的 conda 环境规范文件（`tfrl-cookbook.yml`）保持一致，该文件在食谱代码仓库中。

我们还需要创建一个新的 Python 模块，名为 `tradegym`，其中包含 `crypto_trading_env.py`、`stock_trading_continuous_env.py`、`trading_utils.py` 以及我们在前几章中实现的其他自定义交易环境。你将在书籍的代码仓库中找到包含这些内容的 `tradegym` 模块。

## 如何操作…

我们的实现将包含两个核心模块——`tradegym` 服务器和 `tradegym` 客户端，这些模块是基于 OpenAI Gym HTTP API 构建的。本教程将重点介绍 HTTP 服务接口的定制和核心组件。我们将首先定义作为 `tradegym` 库一部分暴露的最小自定义环境集，然后构建服务器和客户端模块：

1.  首先，确保 `tradegym` 库的 `__init__.py` 文件中包含最基本的内容，以便我们可以导入这些环境：

    ```py
    import sys
    import os
    from gym.envs.registration import register
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    _AVAILABLE_ENVS = {
        “CryptoTradingEnv-v0”: {
            “entry_point”: \
              “tradegym.crypto_trading_env:CryptoTradingEnv”,
            “description”: “Crypto Trading RL environment”,
        },
        “StockTradingContinuousEnv-v0”: {
            “entry_point”: “tradegym.stock_trading_\
                 continuous_env:StockTradingContinuousEnv”,
            “description”: “Stock Trading RL environment with continous action space”,
        },
    }
    for env_id, val in _AVAILABLE_ENVS.items():
        register(id=env_id, entry_point=val.get(
                                         “entry_point”))
    ```

1.  我们现在可以开始实现我们的 `tradegym` 服务器，命名为 `tradegym_http_server.py`。我们将在接下来的几个步骤中完成实现。让我们首先导入必要的 Python 模块：

    ```py
    import argparse
    import json
    import logging
    import os
    import sys
    import uuid
    import numpy as np
    import six
    from flask import Flask, jsonify, request
    import gym
    ```

1.  接下来，我们将导入 `tradegym` 模块，以便将可用的环境注册到 Gym 注册表中：

    ```py
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import tradegym  # Register tradegym envs with OpenAI Gym 
    # registry
    ```

1.  现在，让我们看看环境容器类的框架，并且有注释说明每个方法的作用。你可以参考本书代码仓库中`chapter7`下的完整实现。我们将从类定义开始，并在接下来的步骤中完成框架：

    ```py
    class Envs(object):
        def __init__(self):
            self.envs = {}
            self.id_len = 8  # Number of chars in instance_id
    ```

1.  在此步骤中，我们将查看两个有助于管理环境实例的辅助方法。它们支持查找和删除操作：

    ```py
        def _lookup_env(self, instance_id):
            """Lookup environment based on instance_id and 
               throw error if not found"""
        def _remove_env(self, instance_id):
            """Delete environment associated with 
               instance_id"""
    ```

1.  接下来，我们将查看其他一些有助于环境管理操作的方法：

    ```py
        def create(self, env_id, seed=None):
            """Create (make) an instance of the environment 
               with `env_id` and return the instance_id"""
        def list_all(self):
            """Return a dictionary of all the active 
               environments with instance_id as keys"""
        def reset(self, instance_id):
            """Reset the environment pointed to by the 
               instance_id"""

        def env_close(self, instance_id):
            """Call .close() on the environment and remove 
               instance_id from the list of all envs"""
    ```

1.  本步骤中讨论的方法支持 RL 环境的核心操作，并且这些方法与核心 Gym API 一一对应：

    ```py
        def step(self, instance_id, action, render):
            """Perform a single step in the environment 
               pointed to by the instance_id and return 
               observation, reward, done and info"""
        def get_action_space_contains(self, instance_id, x):
            """Check if the given environment’s action space 
               contains x"""
        def get_action_space_info(self, instance_id):
            """Return the observation space infor for the 
               given environment instance_id"""
        def get_action_space_sample(self, instance_id):
            """Return a sample action for the environment 
               referred by the instance_id"""
        def get_observation_space_contains(self, instance_id, 
        j):
            """Return true is the environment’s observation 
               space contains `j`. False otherwise"""
        def get_observation_space_info(self, instance_id):
            """Return the observation space for the 
               environment referred by the instance_id"""
        def _get_space_properties(self, space):
            """Return a dictionary containing the attributes 
               and values of the given Gym Spce (Discrete, 
               Box etc.)"""
    ```

1.  有了前面的框架（和实现），我们可以通过**Flask** Python 库将这些操作暴露为 REST API。接下来，我们将讨论核心服务器应用程序的设置以及创建、重置和步骤方法的路由设置。让我们看看暴露端点处理程序的服务器应用程序设置：

    ```py
    app = Flask(__name__)
    envs = Envs()
    ```

1.  现在我们可以查看`v1/envs`的 REST API 路由定义。它接受一个`env_id`，该 ID 应该是一个有效的 Gym 环境 ID（如我们的自定义`StockTradingContinuous-v0`或`MountainCar-v0`，这些都可以在 Gym 注册表中找到），并返回一个`instance_id`：

    ```py
    @app.route(“/v1/envs/”, methods=[“POST”])
    def env_create():
        env_id = get_required_param(request.get_json(), 
                                    “env_id”)
        seed = get_optional_param(request.get_json(), 
                                   “seed”, None)
        instance_id = envs.create(env_id, seed)
        return jsonify(instance_id=instance_id)
    ```

1.  接下来，我们将查看`v1/envs/<instance_id>/reset`的 HTTP POST 端点的 REST API 路由定义，其中`<instance_id>`可以是`env_create()`方法返回的任何 ID：

    ```py
    @app.route(“/v1/envs/<instance_id>/reset/”, 
               methods=[“POST”])
    def env_reset(instance_id):
        observation = envs.reset(instance_id)
        if np.isscalar(observation):
            observation = observation.item()
        return jsonify(observation=observation)
    ```

1.  接下来，我们将查看`v1/envs/<instance_id>/step`端点的路由定义，这是在 RL 训练循环中最可能被调用的端点：

    ```py
    @app.route(“/v1/envs/<instance_id>/step/”,
               methods=[“POST”])
    def env_step(instance_id):
        json = request.get_json()
        action = get_required_param(json, “action”)
        render = get_optional_param(json, “render”, False)
        [obs_jsonable, reward, done, info] = envs.step(instance_id, action, render)
        return jsonify(observation=obs_jsonable, 
                       reward=reward, done=done, info=info)
    ```

1.  对于`tradegym`服务器上剩余的路由定义，请参考本书的代码仓库。我们将在`tradegym`服务器脚本中实现一个`__main__`函数，用于在执行时启动服务器（稍后我们将在本教程中使用它来进行测试）：

    ```py
    if __name__ == “__main__”:
        parser = argparse.ArgumentParser(description=”Start a
                                        Gym HTTP API server”)
        parser.add_argument(“-l”,“--listen”, help=”interface\
                            to listen to”, default=”0.0.0.0”)
        parser.add_argument(“-p”, “--port”, default=6666, \
                            type=int, help=”port to bind to”)
        args = parser.parse_args()
        print(“Server starting at: “ + \
               “http://{}:{}”.format(args.listen, args.port))
        app.run(host=args.listen, port=args.port, debug=True)
    ```

1.  接下来，我们将了解`tradegym`客户端的实现。完整实现可在本书代码仓库的`chapter7`中找到`tradegym_http_client.py`文件中。在本步骤中，我们将首先导入必要的 Python 模块，并在接下来的步骤中继续实现客户端封装器：

    ```py
    import json
    import logging
    import os
    import requests
    import six.moves.urllib.parse as urlparse
    ```

1.  客户端类提供了一个 Python 封装器，用于与`tradegym` HTTP 服务器进行接口交互。客户端类的构造函数接受服务器的地址（IP 和端口信息）以建立连接。让我们看看构造函数的实现：

    ```py
    class Client(object):
        def __init__(self, remote_base):
            self.remote_base = remote_base
            self.session = requests.Session()
            self.session.headers.update({“Content-type”: \
                                         “application/json”})
    ```

1.  在这里重复所有标准的 Gym HTTP 客户端方法并不是对本书有限空间的合理利用，因此我们将重点关注核心的封装方法，如`env_create`、`env_reset`和`env_step`，这些方法将在我们的代理训练脚本中广泛使用。有关完整实现，请参阅本书的代码库。让我们看一下用于在远程`tradegym`服务器上创建 RL 仿真环境实例的`env_create`封装方法：

    ```py
        def env_create(self, env_id):
            route = “/v1/envs/”
            data = {“env_id”: env_id}
            resp = self._post_request(route, data)
            instance_id = resp[“instance_id”]
            return instance_id
    ```

1.  在这一步中，我们将查看调用`reset`方法的封装方法，它通过`tradegym`服务器在`env_create`调用时返回的唯一`instance_id`来操作特定的环境：

    ```py
        def env_reset(self, instance_id):
            route = “/v1/envs/{}/reset/”.format(instance_id)
            resp = self._post_request(route, None)
            observation = resp[“observation”]
            return observation
    ```

1.  `tradegym`客户端的`Client`类中最常用的方法是`step`方法。让我们看一下它的实现，应该对你来说很简单：

    ```py
        def env_step(self, instance_id, action, 
        render=False):
            route = “/v1/envs/{}/step/”.format(instance_id)
            data = {“action”: action, “render”: render}
            resp = self._post_request(route, data)
            observation = resp[“observation”]
            reward = resp[“reward”]
            done = resp[“done”]
            info = resp[“info”]
            return [observation, reward, done, info]
    ```

1.  在其他客户端封装方法到位之后，我们可以实现`__main__`例程来连接到`tradegym`服务器，并调用一些方法作为示例，以测试一切是否按预期工作。让我们编写`__main__`例程：

    ```py
    if __name__ == “__main__”:
        remote_base = “http://127.0.0.1:6666”
        client = Client(remote_base)
        # Create environment
        env_id = “StockTradingContinuousEnv-v0”
        # env_id = “CartPole-v0”
        instance_id = client.env_create(env_id)
        # Check properties
        all_envs = client.env_list_all()
        logger.info(f”all_envs:{all_envs}”)
        action_info = \
            client.env_action_space_info(instance_id)
        logger.info(f”action_info:{action_info}”)
        obs_info = \
            client.env_observation_space_info(instance_id)
        # logger.info(f”obs_info:{obs_info}”)
        # Run a single step
        init_obs = client.env_reset(instance_id)
        [observation, reward, done, info] = \
            client.env_step(instance_id, 1, True)
        logger.info(f”reward:{reward} done:{done} \
                      info:{info}”)
    ```

1.  我们现在可以开始实际创建客户端实例并检查`tradegym`服务！首先，我们需要通过执行以下命令来启动`tradegym`服务器：

    ```py
    (tfrl-cookbook)praveen@desktop:~/tensorflow2-reinforcement-learning-cookbook/src/ch7-cloud-deploy-deep-rl-agents$ python tradegym_http_server.py
    ```

1.  现在，我们可以通过在另一个终端运行以下命令来启动`tradegym`客户端：

    ```py
    (tfrl-cookbook)praveen@desktop:~/tensorflow2-reinforcement-learning-cookbook/src/ch7-cloud-deploy-deep-rl-agents$ python tradegym_http_client.py
    ```

1.  你应该会在你启动`tradegym_http_client.py`脚本的终端中看到类似以下的输出：

    ```py
    all_envs:{‘114c5e8f’: ‘StockTradingContinuousEnv-v0’, ‘6287385e’: ‘StockTradingContinuousEnv-v0’, ‘d55c97c0’: ‘StockTradingContinuousEnv-v0’, ‘fd355ed8’: ‘StockTradingContinuousEnv-v0’}
    action_info:{‘high’: [1.0], ‘low’: [-1.0], ‘name’: ‘Box’, ‘shape’: [1]}
    reward:0.0 done:False info:{}
    ```

这就完成了整个流程！让我们简要回顾一下它是如何工作的。

## 它是如何工作的……

`tradegym`服务器提供了一个环境容器类，并通过 REST API 公开环境接口。`tradegym`客户端提供了 Python 封装方法，通过 REST API 与 RL 环境进行交互。

`Envs`类充当`tradegym`服务器上实例化的环境的管理器。它还充当多个环境的容器，因为客户端可以发送请求创建多个（相同或不同的）环境。当`tradegym`客户端使用 REST API 请求`tradegym`服务器创建一个新环境时，服务器会创建所请求环境的实例并返回一个唯一的实例 ID（例如：`8kdi4289`）。从此时起，客户端可以使用实例 ID 来引用特定的环境。这使得客户端和代理训练代码可以同时与多个环境进行交互。因此，`tradegym`服务器通过 HTTP 提供一个 RESTful 接口，充当一个真正的服务。

准备好下一个流程了吗？让我们开始吧。

# 使用远程模拟器服务训练 RL 代理

在这个教程中，我们将探讨如何利用远程模拟器服务来训练我们的代理。我们将重用前面章节中的 SAC 代理实现，并专注于如何使用远程运行的强化学习模拟器（例如在云端）作为服务来训练 SAC 或任何强化学习代理。我们将使用前一个教程中构建的`tradegym`服务器为我们提供强化学习模拟器服务。

让我们开始吧！

## 准备就绪

为了完成这个教程，并确保你拥有最新版本，你需要先激活`tf2rl-cookbook` Python/conda 虚拟环境。确保更新环境，以匹配食谱代码库中的最新 conda 环境规范文件（`tfrl-cookbook.yml`）。如果以下`import`语句没有问题，说明你已经准备好开始了：

```py
import datetime
import os
import sys
import logging
import gym.spaces
import numpy as np
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tradegym_http_client import Client
from sac_agent_base import SAC
```

让我们直接进入正题。

## 怎么做……

我们将实现训练脚本的核心部分，并省略命令行配置及其他非必要功能，以保持脚本简洁。我们将命名脚本为`3_training_rl_agents_using_remote_sims.py`。

让我们开始吧！

1.  让我们先创建一个应用级别的子日志记录器，添加一个流处理器，并设置日志级别：

    ```py
    # Create an App-level child logger
    logger = logging.getLogger(“TFRL-cookbook-ch7-training-with-sim-server”)
    # Set handler for this logger to handle messages
    logger.addHandler(logging.StreamHandler())
    # Set logging-level for this logger’s handler
    logger.setLevel(logging.DEBUG)
    ```

1.  接下来，让我们创建一个 TensorFlow `SummaryWriter`来记录代理的训练进度：

    ```py
    current_time = datetime.datetime.now().strftime(“%Y%m%d-%H%M%S”)
    train_log_dir = os.path.join(“logs”, “TFRL-Cookbook-Ch4-SAC”, current_time)
    summary_writer = tf.summary.create_file_writer(train_log_dir)
    ```

1.  我们现在可以进入实现的核心部分。让我们从实现`__main__`函数开始，并在接下来的步骤中继续实现。首先设置客户端，使用服务器地址连接到模拟服务：

    ```py
    if __name__ == “__main__”:
        # Set up client to connect to sim server
        sim_service_address = “http://127.0.0.1:6666”
        client = Client(sim_service_address)
    ```

1.  接下来，让我们请求服务器创建我们想要的强化学习训练环境来训练我们的代理：

    ```py
        # Set up training environment
        env_id = “StockTradingContinuousEnv-v0”
        instance_id = client.env_create(env_id)
    ```

1.  现在，让我们初始化我们的代理：

    ```py
        # Set up agent
        observation_space_info = \
            client.env_observation_space_info(instance_id)
        observation_shape = \
            observation_space_info.get(“shape”)
        action_space_info = \
            client.env_action_space_info(instance_id)
        action_space = gym.spaces.Box(
            np.array(action_space_info.get(“low”)),
            np.array(action_space_info.get(“high”)),
            action_space_info.get(“shape”),
        )
        agent = SAC(observation_shape, action_space)
    ```

1.  我们现在准备好使用一些超参数来配置训练：

    ```py
        # Configure training
        max_epochs = 30000
        random_epochs = 0.6 * max_epochs
        max_steps = 100
        save_freq = 500
        reward = 0
        done = False
        done, use_random, episode, steps, epoch, \
        episode_reward = (
            False,
            True,
            0,
            0,
            0,
            0,
        )
    ```

1.  到此，我们已经准备好开始外部训练循环：

    ```py
        cur_state = client.env_reset(instance_id)
        # Start training
        while epoch < max_epochs:
            if steps > max_steps:
                done = True
    ```

1.  现在，让我们处理当一个回合结束并且`done`被设置为`True`的情况：

    ```py
            if done:
                episode += 1
                logger.info(
                    f”episode:{episode} \
                     cumulative_reward:{episode_reward} \
                     steps:{steps} epochs:{epoch}”)
                with summary_writer.as_default():
                    tf.summary.scalar(“Main/episode_reward”, 
                                episode_reward, step=episode)
                    tf.summary.scalar(“Main/episode_steps”,
                                       steps, step=episode)
                summary_writer.flush()
                done, cur_state, steps, episode_reward = (
                    False, 
                client.env_reset(instance_id), 0, 0,)
                if episode % save_freq == 0:
                    agent.save_model(
                        f”sac_actor_episode{episode}_\
                          {env_id}.h5”,
                        f”sac_critic_episode{episode}_\
                          {env_id}.h5”,
                    )
    ```

1.  现在是关键步骤！让我们使用代理的`act`和`train`方法，通过采取行动（执行动作）和使用收集到的经验来训练代理：

    ```py
            if epoch > random_epochs:
                use_random = False
            action = agent.act(np.array(cur_state), 
                               use_random=use_random)
            next_state, reward, done, _ = client.env_step(
                instance_id, action.numpy().tolist()
            )
            agent.train(np.array(cur_state), action, reward,
                        np.array(next_state), done)
    ```

1.  现在，让我们更新变量，为接下来的步骤做准备：

    ```py
            cur_state = next_state
            episode_reward += reward
            steps += 1
            epoch += 1
            # Update Tensorboard with Agent’s training status
            agent.log_status(summary_writer, epoch, reward)
            summary_writer.flush()
    ```

1.  这就完成了我们的训练循环。很简单，对吧？训练完成后，别忘了保存代理的模型，这样在部署时我们就可以使用已训练的模型：

    ```py
        agent.save_model(
            f”sac_actor_final_episode_{env_id}.h5”, \
            f”sac_critic_final_episode_{env_id}.h5”
        )
    ```

1.  你现在可以继续并使用以下命令运行脚本：

    ```py
    (tfrl-cookbook)praveen@desktop:~/tensorflow2-reinforcement-learning-cookbook/src/ch7-cloud-deploy-deep-rl-agents$ python 3_training_rl_agents_using_remote_sims.py
    ```

1.  我们是不是忘了什么？客户端连接的是哪个模拟服务器？模拟服务器正在运行吗？！如果你在命令行看到一个类似以下的长错误信息，那么很可能是模拟服务器没有启动：

    ```py
    Failed to establish a new connection: [Errno 111] Connection refused’))
    ```

1.  这次我们要做对！让我们通过使用以下命令启动`tradegym`服务器，确保我们的模拟服务器正在运行：

    ```py
    (tfrl-cookbook)praveen@desktop:~/tensorflow2-reinforcement-learning-cookbook/src/ch7-cloud-deploy-deep-rl-agents$ python tradegym_http_server.py
    ```

1.  我们现在可以使用以下命令启动代理训练脚本（与之前相同）：

    ```py
    (tfrl-cookbook)praveen@desktop:~/tensorflow2-reinforcement-learning-cookbook/src/ch7-cloud-deploy-deep-rl-agents$ python 3_training_rl_agents_using_remote_sims.py
    ```

1.  你应该看到类似以下内容的输出：

    ```py
    ...
    Total params: 16,257
    Trainable params: 16,257
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None
    episode:1 cumulative_reward:370.45421418744525 steps:9 epochs:9
    episode:2 cumulative_reward:334.52956448599605 steps:9 epochs:18
    episode:3 cumulative_reward:375.27432450733943 steps:9 epochs:27
    episode:4 cumulative_reward:363.7160827166332 steps:9 epochs:36
    episode:5 cumulative_reward:363.2819222532322 steps:9 epochs:45
    ...
    ```

这就完成了我们用于通过远程仿真训练 RL 代理的脚本！

## 它是如何工作的…

到目前为止，我们一直直接使用`gym`库与仿真器交互，因为我们在代理训练脚本中运行 RL 环境仿真器。虽然对于依赖 CPU 的本地仿真器，这样做已经足够，但随着我们开始使用高级仿真器，或使用我们没有的仿真器，甚至在我们不想运行或管理仿真器实例的情况下，我们可以利用我们在本章之前配方中构建的客户端包装器，连接到像`tradegym`这样的 RL 环境，它们公开了 REST API 接口。在这个配方中，代理训练脚本利用`tradegym`客户端模块与远程`tradegym`服务器进行交互，从而完成 RL 训练循环。

有了这些，让我们继续下一个配方，看看如何评估之前训练过的代理。

# 测试/评估 RL 代理

假设你已经使用训练脚本（前一个配方）在某个交易环境中训练了 SAC 代理，并且你有多个版本的训练代理模型，每个模型都有不同的策略网络架构或超参数，或者你对其进行了调整和自定义以提高性能。当你想要部署一个代理时，你肯定希望选择表现最好的代理，对吧？

本配方将帮助你构建一个精简的脚本，用于在本地评估给定的预训练代理模型，从而获得定量性能评估，并在选择合适的代理模型进行部署之前比较多个训练模型。具体来说，我们将使用本章之前构建的`tradegym`模块和`sac_agent_runtime`模块来评估我们训练的代理模型。

让我们开始吧！

## 准备工作

要完成此配方，首先需要激活`tf2rl-cookbook`的 Python/conda 虚拟环境。确保更新环境以匹配最新的 conda 环境规范文件（`tfrl-cookbook.yml`），该文件位于食谱的代码库中。如果以下`import`语句没有问题，说明你已经准备好开始了：

```py
#!/bin/env/python
import os
import sys
from argparse import ArgumentParser
import imageio
import gym
```

## 如何操作…

让我们专注于创建一个简单但完整的代理评估脚本：

1.  首先，让我们导入用于训练环境的`tradegym`模块和 SAC 代理运行时：

    ```py
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import tradegym  # Register tradegym envs with OpenAI Gym registry
    from sac_agent_runtime import SAC
    ```

1.  接下来，让我们创建一个命令行参数解析器来处理命令行配置：

    ```py
    parser = ArgumentParser(prog=”TFRL-Cookbook-Ch7-Evaluating-RL-Agents”)
    parser.add_argument(“--agent”, default=”SAC”, help=”Name of Agent. Default=SAC”)
    ```

1.  现在，让我们为`--env`参数添加支持，以指定 RL 环境 ID，并为`–-num-episodes`添加支持，以指定评估代理的回合数。让我们为这两个参数设置一些合理的默认值，这样即使没有任何参数，我们也能运行脚本进行快速（或者说是懒惰？）测试：

    ```py
    parser.add_argument(
        “--env”,
        default=”StockTradingContinuousEnv-v0”,
        help=”Name of Gym env. Default=StockTradingContinuousEnv-v0”,
    )
    parser.add_argument(
        “--num-episodes”,
        default=10,
        help=”Number of episodes to evaluate the agent.\
              Default=100”,
    )
    ```

1.  让我们还为`–-trained-models-dir`添加支持，用于指定包含训练模型的目录，并为`–-model-version`标志添加支持，用于指定该目录中的特定模型版本：

    ```py
    parser.add_argument(
        “--trained-models-dir”,
        default=”trained_models”,
        help=”Directory contained trained models. Default=trained_models”,
    )
    parser.add_argument(
        “--model-version”,
        default=”episode100”,
        help=”Trained model version. Default=episode100”,
    )
    ```

1.  现在，我们准备好完成参数解析：

    ```py
    args = parser.parse_args()
    ```

1.  让我们从实现`__main__`方法开始，并在接下来的步骤中继续实现它。首先，我们从创建一个本地实例的 RL 环境开始，在该环境中我们将评估代理：

    ```py
    if __name__ == “__main__”:
        # Create an instance of the evaluation environment
        env = gym.make(args.env)
    ```

1.  现在，让我们初始化代理类。暂时我们只支持 SAC 代理，但如果你希望支持本书中讨论的其他代理，添加支持非常容易：

    ```py
        if args.agent != “SAC”:
            print(f”Unsupported Agent: {args.agent}. Using \
                    SAC Agent”)
            args.agent = “SAC”
        # Create an instance of the Soft Actor-Critic Agent
        agent = SAC(env.observation_space.shape, \
                    env.action_space)
    ```

1.  接下来，让我们加载训练好的代理模型：

    ```py
        # Load trained Agent model/brain
        model_version = args.model_version
        agent.load_actor(
            os.path.join(args.trained_models_dir, \
                         f”sac_actor_{model_version}.h5”)
        )
        agent.load_critic(
            os.path.join(args.trained_models_dir, \
                         f”sac_critic_{model_version}.h5”)
        )
        print(f”Loaded {args.agent} agent with trained \
                model version:{model_version}”)
    ```

1.  我们现在准备好使用测试环境中的训练模型来评估代理：

    ```py
        # Evaluate/Test/Rollout Agent with trained model/
        # brain
        video = imageio.get_writer(“agent_eval_video.mp4”,\
                                    fps=30)
        avg_reward = 0
        for i in range(args.num_episodes):
            cur_state, done, rewards = env.reset(), False, 0
            while not done:
                action = agent.act(cur_state, test=True)
                next_state, reward, done, _ = \
                                    env.step(action[0])
                cur_state = next_state
                rewards += reward
                if render:
                    video.append_data(env.render(mode=\
                                                ”rgb_array”))
            print(f”Episode#:{i} cumulative_reward:\
                    {rewards}”)
            avg_reward += rewards
        avg_reward /= args.num_episodes
        video.close()
        print(f”Average rewards over {args.num_episodes} \
                episodes: {avg_reward}”)
    ```

1.  现在，让我们尝试在`StockTradingContinuous-v0`环境中评估代理。请注意，股票交易环境中的市场数据源（`data/MSFT.csv`和`data/TSLA.csv`）可能与用于训练的市场数据不同！毕竟，我们想要评估的是代理学会如何交易！运行以下命令启动代理评估脚本：

    ```py
    (tfrl-cookbook)praveen@desktop:~/tensorflow2-reinforcement-learning-cookbook/src/ch7-cloud-deploy-deep-rl-agents$ python 4_evaluating_rl_agents.py
    ```

1.  根据你训练的代理表现如何，你会在控制台看到类似以下的输出（奖励值会有所不同）：

    ```py
    ...
    ==================================================================================================
    Total params: 16,257
    Trainable params: 16,257
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None
    Loaded SAC agent with trained model version:episode100
    Episode#:0 cumulative_reward:382.5117154452246
    Episode#:1 cumulative_reward:359.27720004181674
    Episode#:2 cumulative_reward:370.92829808499664
    Episode#:3 cumulative_reward:341.44002189086007
    Episode#:4 cumulative_reward:364.32631211784394
    Episode#:5 cumulative_reward:385.89219327764476
    Episode#:6 cumulative_reward:365.2120387185878
    Episode#:7 cumulative_reward:339.98494537310785
    Episode#:8 cumulative_reward:362.7133769241483
    Episode#:9 cumulative_reward:379.12388043270073
    Average rewards over 10 episodes: 365.1409982306931
    ...
    ```

就这些！

## 工作原理……

我们初始化了一个 SAC 代理，只使用了通过`sac_agent_runtime`模块评估代理所需的运行时组件，并加载了先前训练好的模型版本（包括演员和评论家模型），这些都可以通过命令行参数进行自定义。然后，我们使用`tradegym`库创建了一个`StockTradingContinuousEnv-v0`环境的本地实例，并评估了我们的代理，以便获取累积奖励作为评估训练代理模型性能的量化指标。

既然我们已经知道如何评估并选择表现最好的代理，让我们进入下一个步骤，了解如何打包训练好的代理以进行部署！

# 打包强化学习代理以便部署——一个交易机器人

这是本章的一个关键步骤，我们将在这里讨论如何将代理打包，以便我们可以将其作为服务部署到云端（下一个步骤！）。我们将实现一个脚本，该脚本将我们的训练好的代理模型并将`act`方法暴露为一个 RESTful 服务。接着，我们会将代理和 API 脚本打包成一个**Docker**容器，准备好部署到云端！通过本步骤，你将构建一个准备好部署的 Docker 容器，其中包含你的训练好的强化学习代理，能够创建并提供 Agent/Bot-as-a-Service！

让我们深入了解细节。

## 准备工作

要完成这个步骤，你需要首先激活`tf2rl-cookbook`的 Python/conda 虚拟环境。确保更新环境，以便与 cookbook 代码库中的最新 conda 环境规格文件（`tfrl-cookbook.yml`）匹配。如果以下`import`语句没有问题，你就可以进行下一步，设置 Docker 环境：

```py
import os
import sys
from argparse import ArgumentParser
import gym.spaces
from flask import Flask, request
import numpy as np
```

对于这个食谱，您需要安装 Docker。请按照官方安装说明为您的平台安装 Docker。您可以在[`docs.docker.com/get-docker/`](https://docs.docker.com/get-docker/)找到相关说明。

## 如何操作……

我们将首先实现脚本，将代理的`act`方法暴露为 REST 服务，然后继续创建 Dockerfile 以将代理容器化：

1.  首先，让我们导入本章早些时候构建的`sac_agent_runtime`：

    ```py
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from sac_agent_runtime import SAC
    ```

1.  接下来，让我们为命令行参数创建一个处理程序，并将`--agent`作为第一个支持的参数，以便指定我们想要使用的代理算法：

    ```py
    parser = ArgumentParser(
        prog=”TFRL-Cookbook-Ch7-Packaging-RL-Agents-For-Cloud-Deployments”
    )
    parser.add_argument(“--agent”, default=”SAC”, help=”Name of Agent. Default=SAC”)
    ```

1.  接下来，让我们添加参数，以便指定我们代理将要部署的主机服务器的 IP 地址和端口。现在我们将设置并使用默认值，当需要时可以从命令行更改它们：

    ```py
    parser.add_argument(
        “--host-ip”,
        default=”0.0.0.0”,
        help=”IP Address of the host server where Agent  
              service is run. Default=127.0.0.1”,
    )
    parser.add_argument(
        “--host-port”,
        default=”5555”,
        help=”Port on the host server to use for Agent 
              service. Default=5555”,
    )
    ```

1.  接下来，让我们添加对指定包含训练好的代理模型的目录以及使用的特定模型版本的参数支持：

    ```py
    parser.add_argument(
        “--trained-models-dir”,
        default=”trained_models”,
        help=”Directory contained trained models. \
              Default=trained_models”,
    )
    parser.add_argument(
        “--model-version”,
        default=”episode100”,
        help=”Trained model version. Default=episode100”,
    )
    ```

1.  作为支持的最终参数集，让我们添加允许指定基于训练模型配置的观测形状和动作空间规格的参数：

    ```py
    parser.add_argument(
        “--observation-shape”,
        default=(6, 31),
        help=”Shape of observations. Default=(6, 31)”,
    )
    parser.add_argument(
        “--action-space-low”, default=[-1], help=”Low value \
         of action space. Default=[-1]”
    )
    parser.add_argument(
        “--action-space-high”, default=[1], help=”High value\
         of action space. Default=[1]”
    )
    parser.add_argument(
        “--action-shape”, default=(1,), help=”Shape of \
        actions. Default=(1,)”
    )
    ```

1.  现在我们可以完成参数解析器，并开始实现`__main__`函数：

    ```py
    args = parser.parse_args()
    if __name__ == “__main__”:
    ```

1.  首先，让我们加载代理的运行时配置：

    ```py
        if args.agent != “SAC”:
            print(f”Unsupported Agent: {args.agent}. Using \
                    SAC Agent”)
            args.agent = “SAC”
        # Set Agent’s runtime configs
        observation_shape = args.observation_shape
        action_space = gym.spaces.Box(
            np.array(args.action_space_low),
            np.array(args.action_space_high),
            args.action_shape,
        )
    ```

1.  接下来，让我们创建一个代理实例，并从预训练模型中加载代理的演员和评论家网络的权重：

    ```py
        # Create an instance of the Agent
        agent = SAC(observation_shape, action_space)
        # Load trained Agent model/brain
        model_version = args.model_version
        agent.load_actor(
            os.path.join(args.trained_models_dir, \
                         f”sac_actor_{model_version}.h5”)
        )
        agent.load_critic(
            os.path.join(args.trained_models_dir, \
                         f”sac_critic_{model_version}.h5”)
        )
        print(f”Loaded {args.agent} agent with trained model\
                 version:{model_version}”)
    ```

1.  现在我们可以使用 Flask 设置服务端点，这将和以下代码行一样简单。请注意，我们在`/v1/act`端点暴露了代理的`act`方法：

    ```py
        # Setup Agent (http) service
        app = Flask(__name__)
        @app.route(“/v1/act”, methods=[“POST”])
        def get_action():
            data = request.get_json()
            action = agent.act(np.array(data.get(
                               “observation”)), test=True)
            return {“action”: action.numpy().tolist()}
    ```

1.  最后，我们只需要添加一行代码，当执行时启动 Flask 应用程序以启动服务：

    ```py
        # Launch/Run the Agent (http) service
        app.run(host=args.host_ip, port=args.host_port, 
                debug=True)
    ```

1.  我们的代理 REST API 实现已经完成。现在我们可以集中精力为代理服务创建一个 Docker 容器。我们将通过指定基础镜像为`nvidia/cuda:*`来开始实现 Dockerfile，这样我们就能获得必要的 GPU 驱动程序，以便在部署代理的服务器上使用 GPU。接下来的代码行将放入名为`Dockerfile`的文件中：

    ```py
    FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
    # TensorFlow2.x Reinforcement Learning Cookbook
    # Chapter 7: Deploying Deep RL Agents to the cloud
    LABEL maintainer=”emailid@domain.tld”
    ```

1.  现在让我们安装一些必要的系统级软件包，并清理文件以节省磁盘空间：

    ```py
    RUN apt-get install -y wget git make cmake zlib1g-dev && rm -rf /var/lib/apt/lists/*
    ```

1.  为了执行我们的代理运行时并安装所有必需的软件包，我们将使用 conda Python 环境。所以，让我们继续按照说明下载并在容器中设置`miniconda`：

    ```py
    ENV PATH=”/root/miniconda3/bin:${PATH}”
    ARG PATH=”/root/miniconda3/bin:${PATH}”
    RUN apt-get update
    RUN wget \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        && mkdir /root/.conda \
        && bash Miniconda3-latest-Linux-x86_64.sh -b \
        && rm -f Miniconda3-latest-Linux-x86_64.sh
    # conda>4.9.0 is required for `--no-capture-output`
    RUN conda update -n base conda
    ```

1.  现在让我们将本章的源代码复制到容器中，并使用`tfrl-cookbook.yml`文件中指定的软件包列表创建 conda 环境：

    ```py
    ADD . /root/tf-rl-cookbook/ch7
    WORKDIR /root/tf-rl-cookbook/ch7
    RUN conda env create -f “tfrl-cookbook.yml” -n “tfrl-cookbook”
    ```

1.  最后，我们只需为容器设置`ENTRYPOINT`和`CMD`，当容器启动时，这些将作为参数传递给`ENTRYPOINT`：

    ```py
    ENTRYPOINT [ “conda”, “run”, “--no-capture-output”, “-n”, “tfrl-cookbook”, “python” ]
    CMD [ “5_packaging_rl_agents_for_deployment.py” ]
    ```

1.  这完成了我们的 Dockerfile，现在我们准备通过构建 Docker 容器来打包我们的代理。你可以运行以下命令，根据 Dockerfile 中的指令构建 Docker 容器，并为其打上你选择的容器镜像名称。让我们使用以下命令：

    ```py
    (tfrl-cookbook)praveen@desktop:~/tensorflow2-reinforcement-learning-cookbook/src/ch7-cloud-deploy-deep-rl-agents$docker build -f Dockerfile -t tfrl-cookbook/ch7-trading-bot:latest
    ```

1.  如果你是第一次运行前面的命令，Docker 可能需要花费较长时间来构建容器。之后的运行或更新将会更快，因为中间层可能已经在第一次运行时被缓存。当一切顺利时，你会看到类似下面的输出（注意，由于我之前已经构建过容器，因此大部分层已经被缓存）：

    ```py
    Sending build context to Docker daemon  1.793MB
    Step 1/13 : FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
     ---> a3bd8cb789b0
    Step 2/13 : LABEL maintainer=”emailid@domain.tld”
     ---> Using cache
     ---> 4322623c24c8
    Step 3/13 : ENV PATH=”/root/miniconda3/bin:${PATH}”
     ---> Using cache
     ---> e9e8c882662a
    Step 4/13 : ARG PATH=”/root/miniconda3/bin:${PATH}”
     ---> Using cache
     ---> 31d45d5bcb05
    Step 5/13 : RUN apt-get update
     ---> Using cache
     ---> 3f7ed3eb3c76
    Step 6/13 : RUN apt-get install -y wget git make cmake zlib1g-dev && rm -rf /var/lib/apt/lists/*
     ---> Using cache
     ---> 0ffb6752f5f6
    Step 7/13 : RUN wget     https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh     && mkdir /root/.conda     && bash Miniconda3-latest-Linux-x86_64.sh -b     && rm -f Miniconda3-latest-Linux-x86_64.sh
     ---> Using cache
    ```

1.  对于涉及从磁盘进行 COPY/ADD 文件操作的层，指令将会执行，因为它们无法被缓存。例如，你会看到以下来自*第 9 步*的步骤会继续执行而不使用任何缓存。即使你已经构建过容器，这也是正常的：

    ```py
    Step 9/13 : ADD . /root/tf-rl-cookbook/ch7
     ---> ed8541c42ebc
    Step 10/13 : WORKDIR /root/tf-rl-cookbook/ch7
     ---> Running in f5a9c6ad485c
    Removing intermediate container f5a9c6ad485c
     ---> 695ca00c6db3
    Step 11/13 : RUN conda env create -f “tfrl-cookbook.yml” -n “tfrl-cookbook”
     ---> Running in b2a9706721e7
    Collecting package metadata (repodata.json): ...working... done
    Solving environment: ...working... done...
    ```

1.  最后，当 Docker 容器构建完成时，你将看到类似以下的消息：

    ```py
    Step 13/13 : CMD [ “2_packaging_rl_agents_for_deployment.py” ]
     ---> Running in 336e442b0218
    Removing intermediate container 336e442b0218
     ---> cc1caea406e6
    Successfully built cc1caea406e6
    Successfully tagged tfrl-cookbook/ch7:latest
    ```

恭喜你成功打包了 RL 代理，准备部署！

## 它是如何工作的…

我们利用了本章前面构建的 `sac_agent_runtime` 来创建和初始化一个 SAC 代理实例。然后我们加载了预训练的代理模型，分别用于演员和评论员。之后，我们将 SAC 代理的 `act` 方法暴露为一个 REST API，通过 HTTP POST 端点来接受观察值作为 POST 消息，并将动作作为响应返回。最后，我们将脚本作为 Flask 应用启动，开始服务。

在本食谱的第二部分，我们将代理应用程序 actioa-serving 打包为 Docker 容器，并准备好进行部署！

我们现在即将将代理部署到云端！继续下一部分食谱，了解如何操作。

# 将 RL 代理部署到云端——作为服务的交易机器人

训练 RL 代理的终极目标是利用它在新的观察值下做出决策。以我们的股票交易 SAC 代理为例，到目前为止，我们已经学会了如何训练、评估并打包表现最佳的代理模型来构建交易机器人。虽然我们集中在一个特定的应用场景（自动交易机器人），但你可以看到，根据本书前几章中的食谱，如何轻松地更改训练环境或代理算法。本食谱将指导你通过将 Docker 容器化的 RL 代理部署到云端并运行作为服务的机器人。

## 正在准备中

要完成这个教程，你需要访问像 Azure、AWS、GCP、Heroku 等云服务，或其他支持托管和运行 Docker 容器的云服务提供商。如果你是学生，可以利用 GitHub 的学生开发者套餐（[`education.github.com/pack`](https://education.github.com/pack)），该套餐从 2020 年起为你提供一些免费福利，包括 100 美元的 Microsoft Azure 信用或作为新用户的 50 美元 DigitalOcean 平台信用。

有很多指南讲解如何将 Docker 容器推送到云端并作为服务部署/运行。例如，如果你有 Azure 账户，可以参照官方指南：[`docs.microsoft.com/en-us/azure/container-instances/container-instances-quickstart`](https://docs.microsoft.com/en-us/azure/container-instances/container-instances-quickstart)。

本指南将带你通过多种选项（CLI、门户、PowerShell、ARM 模板和 Docker CLI）来部署基于 Docker 容器的代理服务。

## 如何操作……

我们将首先在本地部署交易机器人并进行测试。之后，我们可以将其部署到你选择的云服务上。作为示例，本教程将带你通过将其部署到 Heroku 的步骤（[`heroku.com`](https://heroku.com)）。

我们开始吧：

1.  首先使用以下命令构建包含交易机器人的 Docker 容器。请注意，如果你之前已经按照本章的其他教程构建过容器，那么根据缓存的层和对 Dockerfile 所做的更改，以下命令可能会更快地执行完毕：

    ```py
    (tfrl-cookbook)praveen@desktop:~/tensorflow2-reinforcement-learning-cookbook/src/ch7-cloud-deploy-deep-rl-agents$docker build -f Dockerfile -t tfrl-cookbook/ch7-trading-bot:latest
    ```

1.  一旦 Docker 容器成功构建，我们可以使用以下命令启动机器人：

    ```py
    (tfrl-cookbook)praveen@desktop:~/tensorflow2-reinforcement-learning-cookbook/src/ch7-cloud-deploy-deep-rl-agents$docker run -it -p 5555:5555 tfrl-cookbook/ch7-trading-bot
    ```

1.  如果一切顺利，你应该会看到类似以下的控制台输出，表示机器人已启动并准备好执行操作：

    ```py
    ...==================================================================================================
    Total params: 16,257
    Trainable params: 16,257
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None
    Loaded SAC agent with trained model version:episode100
     * Debugger is active!
     * Debugger PIN: 604-104-903
    ...
    ```

1.  现在你已经在本地（在你自己的服务器上）部署了交易机器人，接下来我们来创建一个简单的脚本，利用你构建的 Bot-as-a-Service。创建一个名为`test_agent_service.py`的文件，内容如下：

    ```py
    #Simple test script for the deployed Trading Bot-as-a-Service
    import os
    import sys
    import gym
    import requests
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import tradegym  # Register tradegym envs with OpenAI Gym # registry
    host_ip = “127.0.0.1”
    host_port = 5555
    endpoint = “v1/act”
    env = gym.make(“StockTradingContinuousEnv-v0”)
    post_data = {“observation”: env.observation_space.sample().tolist()}
    res = requests.post(f”http://{host_ip}:{host_port}/{endpoint}”, json=post_data)
    if res.ok:
        print(f”Received Agent action:{res.json()}”)
    ```

1.  你可以使用以下命令执行该脚本：

    ```py
    (tfrl-cookbook)praveen@desktop:~/tensorflow2-reinforcement-learning-cookbook/src/ch7-cloud-deploy-deep-rl-agents$python test_agent_service.py
    ```

1.  请注意，您的机器人容器仍需要运行。一旦执行前述命令，你将看到类似以下的输出，表示在`/v1/act`端点接收到新的 POST 消息，并返回了 HTTP 响应状态 200，表示成功：

    ```py
    172.17.0.1 - - [00/Mmm/YYYY hh:mm:ss] “POST /v1/act HTTP/1.1” 200 -
    ```

1.  你还会注意到，测试脚本在其控制台窗口中会打印出类似以下的输出，表示它收到了来自交易机器人的一个操作：

    ```py
    Received Agent action:{‘action’: [[0.008385116065491426]]}
    ```

1.  现在是将你的交易机器人部署到云平台的时候了，这样你或其他人就可以通过互联网访问它！正如在*入门*部分所讨论的，你在选择云服务提供商方面有多个选择，可以将你的 Docker 容器镜像托管并部署 RL 代理 Bot-as-a-Service。我们将以 Heroku 为例，它提供免费托管服务和简便的命令行界面。首先，你需要安装 Heroku CLI。按照 [`devcenter.heroku.com/articles/heroku-cli`](https://devcenter.heroku.com/articles/heroku-cli) 上列出的官方说明为你的平台（Linux/Windows/macOS X）安装 Heroku CLI。在 Ubuntu Linux 上，我们可以使用以下命令：

    ```py
    sudo snap install --classic heroku
    ```

1.  一旦安装了 Heroku CLI，你可以使用以下命令登录 Heroku 容器注册表：

    ```py
    heroku container:login
    ```

1.  接下来，从包含代理的 Dockerfile 的目录运行以下命令；例如：

    ```py
    tfrl-cookbook)praveen@desktop:~/tensorflow2-reinforcement-learning-cookbook/src/ch7-cloud-deploy-deep-rl-agents$heroku create
    ```

1.  如果你尚未登录 Heroku，你将被提示登录：

    ```py
    Creating app... !
         Invalid credentials provided.
     ›   Warning: heroku update available from 7.46.2 to 
    7.47.0.
    heroku: Press any key to open up the browser to login or q to exit:
    ```

1.  登录后，你将看到类似如下的输出：

    ```py
    Creating salty-fortress-4191... done, stack is heroku-18
    https://salty-fortress-4191.herokuapp.com/ | https://git.heroku.com/salty-fortress-4191.git
    ```

1.  这就是你在 Heroku 上的容器注册表地址。你现在可以使用以下命令构建你的机器人容器并将其推送到 Heroku：

    ```py
    heroku container:push web
    ```

1.  一旦该过程完成，你可以使用以下命令将机器人容器镜像发布到 Heroku 应用：

    ```py
    heroku container:release web
    ```

1.  恭喜！你刚刚将你的机器人部署到了云端！你现在可以通过新的地址访问你的机器人，例如在之前代码中使用的示例地址 [`salty-fortress-4191.herokuapp.com/`](https://salty-fortress-4191.herokuapp.com/)。你应该能够向你的机器人发送观察数据，并获取机器人的回应动作！恭喜你成功部署了你的 Bot-as-a-Service！

我们现在准备好结束本章内容了。

## 它是如何工作的……

我们首先通过使用 `docker run` 命令并指定将本地端口 `5555` 映射到容器的端口 `5555`，在你的机器上本地构建并启动了 Docker 容器。这将允许主机（你的机器）使用该端口与容器通信，就像它是机器上的本地端口一样。部署后，我们使用了一个测试脚本，该脚本利用 Python 的 `request` 库创建了一个带有观察值示例数据的 POST 请求，并将其发送到容器中的机器人。我们观察到机器人如何通过命令行的状态输出响应请求，并返回成功的回应，包含机器人的交易动作。

然后我们将相同的容器与机器人一起部署到云端（Heroku）。成功部署后，可以通过 Heroku 自动创建的公共 `herokuapp` URL 在网络上访问机器人。

这完成了本章的内容和食谱！希望你在整个过程中感到愉快。下章见。
