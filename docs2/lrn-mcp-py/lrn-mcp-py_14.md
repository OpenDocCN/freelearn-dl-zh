# 附录：使用现代 Python 构建 Web

这是为了那些想要了解如何使用 Python 及其现代构造的你们。你们可能是因为 AI 的普及而刚刚转向 Python，或者可能你们对编程还比较新手。无论如何，热烈的欢迎；这个附录是为你们准备的。它的目标是确保当你们阅读其他章节时，能够理解所有使用的构造，并能够从中受益。

# 类型提示和数据建模

尽管 Python 不强制类型，但它确实从类型中受益。类型提示可以使代码更易于阅读，并有助于早期捕获错误。

下面是一些没有类型提示的将产品添加到购物篮的代码：

```py
basket = []
def add_product_to_basket(product):
    basket.append(product) 
```

虽然这段代码可以工作，但它缺乏关于添加到购物篮中的产品类型的清晰性。还有在访问`product`属性时出错的风险。因此，你应该考虑使用类型提示。

## 带有类型提示的改进代码

类型提示极大地帮助了 IDE 工具，使其更容易捕获错误并理解代码。它们也有助于提高可读性。例如，`str`、`int`等类型提示无需添加库即可正常工作。还有`typing`库，它引入了额外的类型，例如`List`和`Optional`。

使用类型提示可以使之前显示的代码变得更加易于阅读：

```py
from typing import List, Optional
basket: List[dict] = []
def add_product_to_basket(product: dict) -> None:
    basket.append(product) 
```

这已经好多了，因为`product`现在明显是一个字典，而`basket`是一个字典列表。我们可以通过将`product`转换为类来进一步改进这一点，如下所示：

```py
from typing import List
class Product:
    def __init__(self, id: int, name: str, price: float):
        self.id = id
        self.name = name
        self.price = price
basket: List[Product] = []
def add_product_to_basket(product: Product) -> None:
    basket.append(product) 
```

什么比这更好？答案是像**Pydantic**这样的库。

## 添加 Pydantic

那么，我们有哪些问题需要使用验证库呢？以下是一些例子：

+   **确保数据完整性**，以确保正在处理的数据是准确和可靠的

+   **减少样板代码**，因为它可以自动生成验证逻辑

+   **提供清晰的错误消息**，使调试问题更容易

+   **简化复杂的数据验证逻辑**，使代码更易于维护

### 没有使用 Pydantic

想象一下，如果你从网络请求中接收数据；数据以字典的形式存在，但需要特定类型，因此我们需要将其转换为修正它。以下是我们用 Python 代码描述的情况：

```py
product_dictionary = {
    "id": 1,
    "name": "Sample Product",
    "price": 19.99
}
class Product:
    def __init__(self, id: int, name: str, price: float):
        self.id = id
        self.name = name
        self.price = price
def from_dict(data: dict) -> Product:
    return Product(
        id=data["id"],
        name=data["name"],
        price=data["price"]
    )
def service(data: dict) -> Product:
    product = from_dict(data)
    # Here you would typically save the product to a database
    return product
p = service(product_dictionary)
print(p) 
```

在这里，我们有一个数据以字典形式存在的情况，即`product_dictionary`，但我们需要将其转换为特定类型`Product`。我们需要创建一个`from_dict`函数来处理这种转换。这正是 Pydantic 大放异彩的地方，因为它可以帮助我们轻松验证和解析这些数据。

### 使用 Pydantic

介绍 Pydantic，让我们看看我们如何利用它来满足我们的用例。通过从`BaseModel`继承，我们可以创建一个模型，该模型会自动验证和解析我们的输入数据。请看以下示例，其中我们定义了一个`Product`模型：

```py
from pydantic import BaseModel
class Product(BaseModel):
    id: int
    name: str
    price: float
product_dictionary = {
    "id": 1,
    "name": "Sample Product",
    "price": 19.99
}
product = Product(**product_dictionary)
print(product) 
```

在前面的代码中，我们消除了手动转换函数的需求。Pydantic 负责验证和解析输入数据，确保其符合预期的结构。这不仅简化了我们的代码，还使其更加健壮且易于维护。

但这里有一个问题：如果我们向字典中提供错误的数据类型或缺失的字段，Pydantic 将引发验证错误。实际上，这段代码会导致崩溃，所以让我们看看我们如何确保捕获任何验证错误。

### 有信心地进行转换

如果字典不符合预期的结构，Pydantic 将引发验证错误，清楚地表明出了什么问题。让我们确保我们捕获任何验证错误：

```py
from pydantic import ValidationError, BaseModel
class Product(BaseModel):
    id: int
    name: str
    price: float
product_dictionary = {
    "id": 1,
    "name": "Sample Product",
    "price": 19.99
}
try:
    product = Product(**product_dictionary)
    print(product)
except ValidationError as e:
    print("Validation error:", e) 
```

使用这段代码，我们用 `try`-`except` 块包装对 Pydantic 的调用，以优雅地处理任何验证错误。Pydantic 不仅处理字符串、数字和布尔值；它还可以处理更复杂的数据类型，如列表和字典。

### 更高级的对象

让我们来看一个稍微复杂一些的例子，其中包含一位教授及其办公时间可用性：

```py
from pydantic import BaseModel, ValidationError
from typing import List, Dict
professor_dictionary = {
    "id": 1,
    "name": "Dr. Smith",
    "office_hours": [
        {"day": "Monday", "from_": 9, "to_": 12},
        {"day": "Wednesday", "from_": 14, "to_": 17}
    ]
}
class OfficeHour(BaseModel):
    day: str
    from_: int
    to_: int
class Professor(BaseModel):
    id: int
    name: str
    office_hours: List[OfficeHour]
professor = Professor(**professor_dictionary)
print(professor) 
```

在此代码中，我们定义了一个包含 ID、姓名和办公时间列表的 `Professor` 模型。每个办公时间由一个 `OfficeHour` 模型表示，该模型包括星期几以及开始和结束时间。这种结构使我们能够轻松地使用 Pydantic 验证和操作复杂的数据类型。

### 序列化

有时我们会有这样的情况，需要从 Pydantic 模型实例回到字典。这在序列化或我们需要与期望特定格式的 API 交互时很有用。为此，我们可以使用存在于每个模型上的 `model_dump` 方法。以下是使用方法：

```py
from pydantic import BaseModel
from typing import List, Dict
class OfficeHour(BaseModel):
    day: str
    from_: int
    to_: int
class Professor(BaseModel):
    id: int
    name: str
    office_hours: List[OfficeHour]
professor_dict = {
    "id": 1,
    "name": "Dr. Smith",
    "office_hours": [
        {"day": "Monday", "from_": 9, "to_": 12},
        {"day": "Wednesday", "from_": 14, "to_": 17}
    ]
}
professor = Professor(**professor_dict)
professor_serialized = professor.model_dump() # {"id": 1, "name":
    "Dr. Smith", "office_hours": [{"day": "Monday", "from_": 9,
    "to_": 12}, {"day": "Wednesday", "from_": 14, "to_": 17}]}}
print(professor)
print(professor_serialized) 
```

您甚至可以使用 `model_dump(include=...)` 和 `model_dump(exclude=...)` 决定包含哪些字段：

```py
print(m.model_dump(include={'foo', 'bar'}))
#> {'foo': 'hello', 'bar': {'whatever': 123}}
print(m.model_dump(exclude={'foo', 'bar'})) 
```

Pydantic 在 MCP 内部 SDK 中被大量使用，并鼓励您在输入和输出验证中使用它。

您甚至可以编写自己的验证器，但我会把这留作您的练习。请参阅官方文档[`docs.pydantic.dev/latest/concepts/validators/`](https://docs.pydantic.dev/latest/concepts/validators/)。

# async 和 await

验证代码很重要，但我们还需要理解另一个重要方面，那就是异步编程。异步编程允许我们编写能够同时执行多个任务而不阻塞主线程的代码。这在需要处理 I/O 密集型操作的场景中特别有用，例如进行 API 调用或从数据库中读取。

让我们谈谈 **async/await**，这是在 Python 中编写异步代码的语法。当你想要编写非阻塞代码，能够同时处理多个任务时，你应该使用 `async`/`await`。如果你的代码是阻塞的，那么最终用户的体验会受到影响，因为他们必须等待每个任务完成才能继续。想象一下在拥有许多用户的 Web 服务器上这个问题是如何成倍的。好吧，那么我们需要知道什么？首先，让我们从概念开始：

+   `协程`：当你用 `async def` 标记一个函数时，就创建了一个所谓的协程。这意味着这个函数可以被暂停和恢复，允许在此期间运行其他任务。让我们在下面的代码中看看这个暂停和恢复的行为：

    ```py
    import asyncio 
    async def fetch_data():
        await asyncio.sleep(1)
        return {"data": "some data"} 
    ```

在此代码中，我们定义了一个带有 `async` 关键字的函数 `fetch_data`。该函数本身调用 `await asyncio.sleep(1)`，这意味着我们希望代码在这里停止一秒钟。然后，我们恢复并返回一个字典。这是非阻塞的吗？是的，因为当 `asyncio.sleep` 运行时，其他工作可以继续进行。

+   **事件循环**：事件循环是运行协程并管理其执行的部分。要与事件循环交互并运行你的 `async` 代码，请调用 `asyncio.run(fetch_data())`：

    ```py
    import asyncio
    async def main():
        data = await fetch_data()
        print(data)
    asyncio.run(main()) 
    ```

另一种与事件循环交互的方式是通过调用 `asyncio.get_running_loop()`。这将给你当前的事件循环实例：

+   `await`：你已经看到它的使用了，但任何标记为 `async` 的函数在调用时都应该使用 `await`。

+   `asyncio`：我们已经在调用它的 `run` 方法时展示了它。它是一个在 Python 中提供异步编程支持的库。它允许你使用 `async`/`await` 语法编写并发代码。当你在与 FastAPI 等网络框架一起工作时，你经常会使用 `asyncio`。

让我们看看使用 FastAPI 的一个 Web 应用程序示例：

```py
from fastapi import FastAPI
app = FastAPI()
async def fetch_data():
    await asyncio.sleep(1)
    return {"data": "some data"}
@app.get("/data")
async def get_data():
    data = await fetch_data()
    return data 
```

+   `asyncio.gather`：`asyncio` 提供了另一个有用的方法，名为 `gather`，它允许你并发运行多个协程并等待它们全部完成。当你需要所有任务的结果时，它通常比 `wait` 更方便。以下是使用它的方法：

    ```py
    import asyncio
    async def fetch_data(url: str):
        print("Fetching data...")
        await asyncio.sleep(1)
        return {"data": f" Result from {url}: some data"}
    async def main():
        # Gather multiple coroutines correctly by passing them as     separate arguments
        results = await asyncio.gather(
            fetch_data("google.com"),
            fetch_data("bing.com"),
            fetch_data("yahoo.com"),
        )
    print(results)
    asyncio.run(main()) 
    ```

在这里，每次调用 `fetch_data` 都将作为参数传递给 `gather`。虽然 `gather` 可能更方便，但它并不像 `wait` 那样提供对单个任务的很多控制。说到 `wait`，让我们看看它是如何工作的。

+   `asyncio.wait`：`asyncio` 提供了一些有用的方法，其中之一就是 `wait` 方法，它允许你等待多个任务完成。当你想要并发运行多个任务并等待它们全部完成时，这特别有用。以下是使用它的方法：

    ```py
    # python
    import asyncio
    from typing import List, Optional
    async def search_task(name: str, delay: int, workload: List[int],
        find_value: int, stop: asyncio.Event) -> Optional[str]:
        try:
            print(f"Task {name} started")
            await asyncio.sleep(delay)             # simulate I/O
            if stop.is_set():
                return None
            for no in workload:
                await asyncio.sleep(0)            # yield to allow
                    cancellation
                if no == find_value:
                    stop.set()
                    return name
            return None
        except asyncio.CancelledError:
            print(f"Task {name} cancelled")
            raise
    async def main():
        stop = asyncio.Event()
        tasks = [
            asyncio.create_task(search_task("A", 3, [1,2,3], 2, stop)),
            asyncio.create_task(search_task("B", 1, [4,5,6], 2, stop)),
            asyncio.create_task(search_task("C", 5, [7,8,9], 2, stop)),
       ]
        try:
            for finished in asyncio.as_completed(tasks):
                res = await finished
                if res:
                    print("Found in", res)
                    break
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    asyncio.run(main()) 
    ```

在这里，代码创建了三个搜索任务，每个任务都有不同的延迟和工作负载。这些任务通过 `asyncio.create_task` 并发启动。使用 `asyncio.as_completed` 函数来处理完成的结果。如果一个任务找到了目标值，它将设置 `stop` 事件，这将取消其他任务。最后，所有任务都被等待以确保适当的清理。

正如您所看到的，在 Python 中使用`asyncio`管理并发任务的方式有很多灵活性。让我们再提一点：到目前为止，您已经看到了如何使您的函数`async`化，甚至您的 Web 框架。还有一个专门用于使 Web 请求`async`化的库，称为`httpx`。

`httpx`是一个功能齐全的 Python 3 HTTP 客户端，它提供了`async`能力。它允许您以异步方式发出 HTTP 请求，这使得它非常适合 I/O 密集型任务。以下是一个如何使用`httpx`和`async`的简单示例：

```py
import httpx
import asyncio
# make a web request to google.com
async def fetch_web_page(url: str) -> httpx.Response:
    async with httpx.AsyncClient() as client:
        return await client.get(url)
async def main():
    response = await fetch_web_page("https://www.google.com")
    print(response.status_code)
asyncio.run(main()) 
```

太好了，现在我们对`async`有了更好的理解，让我们看看如何在 Web 环境中使用它。

# Uvicorn：ASGI 服务器

首先，让我们看看我们是从哪里来的，即 WSGI。**WSGI**代表**Web Server Gateway Interface**，它是 Web 服务器和 Python Web 应用之间的一个标准接口。多年来，它一直是 Python Web 应用的事实标准。然而，为了满足现代 Web 应用的需求，特别是那些需要异步能力的应用，引入了**异步服务器网关接口**（**ASGI**）。因此，使用 ASGI，我们可以更有效地处理异步请求，这是 WSGI 所不能做到的。

Uvicorn 是一个快速的 ASGI 服务器实现，使用`uvloop`和`httptools`。它非常适合提供 FastAPI 应用服务，可以处理 HTTP 和 WebSocket 协议。`uvloop`是事件循环的快速实现，这是 Python 异步编程的核心。`httptools`是一个用于解析 HTTP 请求和响应的库。Uvicorn 与 FastAPI 无缝协作，例如，它可以启动服务器，使您的应用能够并发处理请求。以下是一个 Web 服务器示例：

```py
#main.py
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
async def read_root():
    return {"Hello": "World"} 
```

您可以使用以下命令使用 Uvicorn 运行此 FastAPI 应用：

```py
uvicorn main:app --reload 
```

前面的代码所做的是运行一个名为`main.py`的文件，并查找名为`app`的 FastAPI 应用实例。然后它将启动服务器并启用热重载，因此您对代码所做的任何更改都将自动应用，无需重新启动服务器。

您还可以指定其他选项，例如主机和端口：

```py
uvicorn main:app --host 0.0.0.0 --port 8000 
```

这只是使用 Uvicorn 的开始，但能够使用 SSE 或 Streamable HTTP 等传输创建 MCP 服务器是一个很好的起点。

# 上下文管理器

在使用 MCP SDK 工作时，您还会经常看到上下文管理器的使用，那么这些是什么？上下文管理器是一个 Python 对象，它定义了执行`with`语句时要建立的运行时上下文。最常见的用例是资源管理，您希望确保资源被正确获取和释放。例如，如果您需要设置数据库连接或其他类型的资源，使用它是个好主意。以下是一个简单示例：

```py
with db_resource("sqlite.db") as conn:
    # Perform database operations
    pass 
```

您可能会问，是什么让我们能够使用`with`关键字。好吧，上下文管理器能够使用`with`关键字是因为它们实现了两个特殊方法：`__enter__`和`__exit__`。当执行`with`语句时，会调用`__enter__`方法，而当退出`with`语句内部的块时，会调用`__exit__`方法。这允许自动执行设置和清理操作。让我们看看前面的类是如何实现这些方法的：

```py
class db_resource:
    def __init__(self, db_name):
        self.db_name = db_name
    def __enter__(self):
        # Code to establish the database connection
        self.conn = sqlite3.connect(self.db_name)
        return self.conn
    def __exit__(self, exc_type, exc_value, traceback):
        # Code to close the database connection
        if self.conn:
            self.conn.close() 
```

上述代码演示了使用上下文管理器来管理数据库连接的使用方法。您可以看到`__enter__`方法是如何建立连接并返回它的，而`__exit__`方法确保在退出块时关闭连接，即使发生异常也是如此。

`with`语句还有另一个版本，即`async with`，它用于异步上下文管理器。当处理异步代码时，这些特别有用，例如使用`asyncio`或处理异步 I/O 操作时。SDK 在客户端使用这种模式，例如，用于建立初始服务器连接，如下所示：

```py
server_params = StdioServerParameters(
    command="python",
    args=["server.py"]
)
async with stdio_client(server_params) as (read, write): 
```

在此代码中，现在它调用异步上下文管理器的`__aenter__`和`__aexit__`方法，这有助于创建服务器连接。如果您继续查看客户端的 SDK 代码，您将看到完整的初始化代码如下：

```py
async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # do something with session 
```

这里发生了什么？这里正在联合使用两个异步上下文管理器。外部上下文管理器（`stdio_client`）负责管理服务器连接的标准输入/输出流，而内部上下文管理器（`ClientSession`）负责管理客户端会话的生命周期。从实现的角度来看，这里有一些示例代码：

```py
class stdio_client:
    def __init__(self, params):
        self.params = params
    async def __aenter__(self):
        self.read, self.write = await self._create_client()
        return self.read, self.write
    async def __aexit__(self, exc_type, exc_value, traceback):
        await self._cleanup()
    async def _create_client(self):
        process = await asyncio.create_subprocess_exec(
            self.params.command, *self.params.args,
            env=self.params.env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        return process.stdin, process.stdout
    async def _cleanup(self):
        pass
class ClientSession:
    def __init__(self, read, write):
        self.read = read
        self.write = write
    async def __aenter__(self):
        # Code to initialize the client session
        return self
    async def __aexit__(self, exc_type, exc_value, traceback):
        # Code to clean up the client session
        pass
    async def initialize(self):
        # Code to initialize the client session
        pass
async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
       # do something with session
       await session.initialize() 
```

注意`stdio_client`中的`__aenter__`方法负责创建客户端连接并返回必要的流，而`ClientSession`中的`__aenter__`方法负责使用这些流初始化会话。所以，现在您可能已经理解为什么代码看起来是这个样子。也就是说，一些必要的初始化操作是在上下文管理器创建时发生的，而一些清理操作是在退出时发生的。如果我们不使用这种模式，很容易忘记清理部分，从而导致资源泄露。

## `contextmanager`库

Python 中的`contextmanager`库提供了一种使用生成器函数创建上下文管理器的方法。您不需要使用`__enter__`和`__exit__`方法来定义类，而是可以使用`@contextmanager`装饰器将生成器函数转换为上下文管理器。让我们看看一个例子：

```py
from contextlib import contextmanager
import asyncio
# create a database resource using context manager
@contextmanager
def database_connection():
    conn = create_connection()
    try:
        yield conn
    finally:
        conn.close()
# use it
def main():
    with database_connection() as db:
        # use the database connection
        pass 
```

您可以看到我们不需要编写`__enter__`和`__exit__`，而是可以用`@contextmanager`装饰器来装饰一个函数，以达到相同的效果。这个库也被 MCP SDK 用来为各种资源设置上下文管理器。

# uv：您的开发环境管理器

让我们总结这个附录的最后一个主题，即 `uv`。这不是您必须使用的东西，但强烈建议您这样做。原因如下：

+   它提供了一个一致的界面来管理您的开发环境。这意味着您可以在不担心冲突的情况下轻松地在不同的项目和它们的依赖项之间切换。

+   它简化了设置和管理虚拟环境的过程，使您能够更容易地在具有不同要求的多个项目上工作。因此，您不必键入 `python -m venv env` 和 `source env/bin/activate`，只需简单地使用 `uv init` 来创建和激活虚拟环境，这也会启动您的项目。

+   它简化了安装和管理依赖项的过程，允许您轻松地按需添加、删除和更新包。您习惯于键入 `pip install <package>` 来安装一个包，但使用 `uv`，您可以使用 `uv add <package>` 代替。

+   使用 `project.toml` 进行项目管理，使操作更简单，允许您在单个文件中定义您项目的依赖项和设置，而不是分散在多个文件中。

+   您甚至可以使用 `uv` 来管理您的 Docker 容器和镜像，使在一致的环境中部署您的应用程序变得更加容易。

您可以这样做：

1.  使用以下命令安装：

    ```py
    pip install uv. 
    ```

1.  在您的项目根目录中创建一个 `uv` 配置文件：

    ```py
    uv init 
    ```

1.  使用 `uv add` 命令添加依赖项：

    ```py
    uv add <package> 
    ```

1.  使用以下命令运行您的应用程序：

    ```py
    uv run 
    ```

我们已经到达了这个附录的结尾。希望您对现代 Python 以及 `uv`、`asyncio` 和 Uvicorn 等工具有了更深入的了解。

|

#### 现在解锁这本书的独家优惠

扫描此二维码或访问 [`packtpub.com/unlock`](https://packtpub.com/unlock)，然后通过名称搜索此书。 | ![解锁-01.png](img/Unlock-01.png)![](img/Unlock1.png) |

| **注意**：在开始之前，请准备好您的购买发票。* |
| --- |

![新 Packt 标志](img/New_Packt_Logo1.png)

[www.packtub.com](https://www.packtub.com)

订阅我们的在线数字图书馆，全面访问超过 7,000 本书和视频，以及领先的工具，帮助您规划个人发展并推进您的职业生涯。更多信息，请访问我们的网站。

# 为什么订阅？

+   使用来自 4,000 多位行业专业人士的实用电子书和视频，减少学习时间，增加编码时间

+   通过为您量身定制的技能计划提高您的学习效果

+   每月免费获得一本电子书或视频

+   完全可搜索，便于轻松访问关键信息

+   复制粘贴、打印和收藏内容

在 [www.packtpub.com](https://www.packtpub.com)，您还可以阅读一系列免费的技术文章，订阅各种免费通讯，并享受 Packt 书籍和电子书的独家折扣和优惠。

# 您可能还会喜欢的其他书籍

如果您喜欢这本书，您可能会对 Packt 的以下其他书籍感兴趣：

![9781803238944.jpg](https://www.packtpub.com/en-us/product/nodejs-design-patterns-9781803238944)

**Node.js 设计模式，第四版**

卢西亚诺·马米诺和马里奥·卡西亚罗

ISBN: 978-1-80323-894-4

+   理解 Node.js 基础知识及其异步事件驱动架构

+   使用回调、承诺和 async/await 编写正确的异步代码

+   利用 Node.js 流创建数据驱动的处理管道

+   为生产级应用实现可信的软件设计模式

+   编写可测试的代码和自动化测试（单元测试、集成测试、端到端测试）

+   使用高级食谱：缓存、批处理、异步初始化、卸载 CPU 密集型工作

+   使用 Node.js 构建和扩展微服务和分布式系统

![](https://www.packtpub.com/en-us/product/responsive-web-design-with-html5-and-css-9781837028238)

**使用 HTML5 和 CSS 的响应式网页设计，第五版**

本·弗莱恩

ISBN: 978-1-83702-823-8

+   利用颜色函数混合颜色并在颜色空间之间转换

+   使用媒体查询和容器查询来检测触摸/鼠标和颜色偏好

+   利用 HTML 语义来编写可访问的标记

+   使用 SVG 提供分辨率无关的图像，并学习高效地显示它们

+   仅使用 CSS 创建动画，当项目进入和离开视口时

+   发现 CSS 自定义属性并利用新的 CSS 函数

+   向 HTML 表单添加验证和界面元素

+   检查由 AI 工具生成的前端代码是否满足您的目标

# Packt 正在寻找像您这样的作者

如果您有兴趣成为 Packt 的作者，请访问[authors.packtpub.com](https://authors.packtpub.com)并今天申请。我们已与成千上万的开发人员和科技专业人士合作，就像您一样，帮助他们与全球科技社区分享他们的见解。您可以提交一般申请，申请我们正在招募作者的特定热门话题，或提交您自己的想法。

# 分享您的想法

现在您已经完成了使用 Python 学习模型上下文协议，我们非常乐意听到您的想法！如果您从亚马逊购买了这本书，请点击此处直接转到该书的亚马逊评论页面并分享您的反馈或在该购买网站上留下评论。

您的评论对我们和科技社区非常重要，并将帮助我们确保我们提供高质量的内容。
