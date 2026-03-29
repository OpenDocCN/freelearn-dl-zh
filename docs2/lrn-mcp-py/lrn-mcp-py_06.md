

# 第六章：高级服务器

在*第三章*中，你看到了如何构建 MCP 服务器。然而，还有另一种构建这些服务器的方法——即使用更高级的方法。使用这种更高级方法的原因是你想要有更多的控制。

在本章中，你将学习以下内容：

+   使用上下文管理器来管理你的服务器生命周期

+   改进你的服务器架构

+   理解对 MCP 服务器的低级访问

本章涵盖了以下主题：

+   为什么选择低级方法？

+   上下文管理器

+   MCP 服务器中的上下文管理器

+   低级访问

+   组织你的架构

# 为什么选择低级方法？

到目前为止，你可能正在想，“我为什么要这样做？之前的方法是如此简单！”好吧，有几个原因你可能想使用这种方法。你可能想做以下事情：

+   **使用上下文管理器来管理你的服务器生命周期**：在这里，你可以做一些事情，比如连接到数据库或其他与你的服务器相连的服务。通过更多地控制服务器的生命周期，你可以确保当服务器不再需要时，服务器被正确地初始化和清理。

+   **改进你的服务器架构**：对服务器构建方式有更多控制，允许在注册工具和资源以及处理传入请求方面有更多自由。这种增加的控制允许你以更易于维护和扩展的方式组织代码。本章将向你展示如何使用低级服务器和普通 MCP 服务器来组织你的代码。在两种情况下都可以创建一个干净的架构。然而，可以争辩说，低级服务器方法更为干净，因为你不必传递服务器实例。你将在本章后面了解更多关于最后那个陈述的含义。

+   **在某些情况下前进的唯一方式**：有些情况下，当你处理某些特性时，除了低级方法之外，没有其他前进的方式。这在使用*第九章*（关于采样）和*第十章*（关于启发式）时是正确的。

让我们深入探讨更低级的方法，这样你知道它是什么样子，以便你可以选择最适合你项目的方案。

# 上下文管理器

那么，什么是上下文管理器呢？**上下文管理器**是一种结构，它允许你在需要的时候精确地分配和释放资源。使用上下文管理器最常见的方式是使用`with`语句，这确保了即使在发生错误的情况下，资源也会被正确清理。通过使用上下文管理器，你的代码变得更干净、更易读。让我们看看一个简单的例子：

```py
with Database_connection() as conn:
    # Use the connection
    result = conn.execute("SELECT * FROM table")
    for row in result:
        print(row) 
```

## 使用 contextlib 创建上下文管理器

使用上下文管理器的另一种方法是使用`contextlib`模块（有一个对应的 NPM 库叫做`contextlib`）创建一个自定义上下文管理器。这允许你创建上下文管理器而不需要定义一个类。以下是一个例子：

```py
import contextlib
@contextlib.contextmanager
def database_connection():
    conn = connect_to_database()
    try:
        yield conn  # This is where the resource is provided to the block
    finally:
        close_connection(conn)  # Cleanup happens here 
```

这里，你可以看到如何将`DatabaseConnection`类替换为一个使用`contextlib.contextmanager`装饰器的`database_connection()`函数。`yield`语句提供了资源给代码块，并且在代码块退出时执行`yield`语句之后的代码，确保了正确的清理。这比之前的例子更好吗？嗯，至少你输入的代码更少。

## 实现一个上下文管理器

如果你想知道如何实现一个上下文管理器，因为你好奇或者因为你不想再添加另一个依赖，以下是你可以这样做的方法：

```py
class DatabaseConnection:
    def __enter__(self):
        self.conn = self.connect_to_database()
        return self.conn
    def __exit__(self, exc_type, exc_value, traceback):
        self.close_connection(self.conn)
    def connect_to_database(self):
        # Logic to connect to the database
        pass
    def close_connection(self, conn):
        # Logic to close the database connection
        pass 
```

在前面，`DatabaseConnection`类通过定义`__enter__`和`__exit__`方法实现了上下文管理器协议。当进入`with`块时调用`__enter__`方法，并返回资源（在这种情况下，是一个数据库连接）。当块退出时调用`__exit__`方法，并处理任何必要的清理，例如关闭连接。我们可以这样调用它：

```py
with DatabaseConnection() as conn:
    # Use the connection
    result = conn.execute("SELECT * FROM table")
    for row in result:
        print(row) 
```

没有上下文管理器，你的代码看起来会是这样：

```py
conn = DatabaseConnection().connect_to_database()
try:
    # Use the connection
    result = conn.execute("SELECT * FROM table")
    for row in result:
        print(row)
finally:
    DatabaseConnection().close_connection(conn) 
```

想象一下如果你在`finally`块中忘记关闭连接会发生什么？你可能是一个非常自律的程序员，总是记得关闭连接，但在更大的代码库中，很容易忘记。上下文管理器通过确保资源总是被正确清理来帮助你避免这样的陷阱。

让我们看看在 MCP 服务器上下文中如何使用上下文管理器。

# MCP 服务器中的上下文管理器

MCP 允许你控制你资源的生命周期管理。让我们看看一些代码：

```py
async def load_settings() -> dict:
    """Load settings from a configuration file."""
    # Simulate loading settings
    return {"setting1": "value1", "setting2": "value2"}
@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[dict]:
    """Manage server startup and shutdown lifecycle."""
    # Initialize resources on startup
    db = await Database.connect()
    settings = await load_settings()
    try:
        yield {"db": db, "settings": settings}
    finally:
        # Clean up on shutdown
        await db.disconnect() 
```

在这里，我们做了几件事情：

+   定义了一个异步上下文管理器`server_lifespan`，用于管理服务器的生命周期

+   在服务器启动时初始化资源，如数据库连接和设置，在服务器关闭时清理它们

+   将这些资源暴露给服务器的请求上下文，使得处理器可以轻松访问它们

说到处理器，让我们看看在以下代码中你如何在服务器的处理器中访问这些资源：

```py
# Pass lifespan to server
server = Server("example-server", lifespan=server_lifespan)
# Access lifespan context in handlers
@server.call_tool()
async def query_db(name: str, arguments: dict) -> list:
    ctx = server.request_context
    db = ctx.lifespan_context["db"]
    settings = ctx.lifespan_context["settings"]
    # TODO: Use the database connection and settings
    return await db.query(arguments["query"]) 
```

在前面的代码中，我们做了以下几件事情：

+   定义了一个工具`query_db`，它访问在`server_lifespan`上下文管理器中初始化的资源。

+   通过使用`db = ctx.lifespan_context["db"]`和`settings = ctx.lifespan_context["settings"]`代码从服务器的请求上下文的`lifespan_context`中访问数据库连接和设置。

太好了，现在我们理解了上下文管理以及它的存在原因。让我们在下一节中更详细地看看低级访问。

# 低级访问

让我们回顾一下我们最初是如何构建服务器的，这样我们就可以轻松地将其与低级访问的不同之处进行比较。以下是使用高级 API 构建简单 MCP 服务器的方法：

```py
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Echo")
@mcp.resource("echo://{message}")
def echo_resource(message: str) -> str:
    """Echo a message as a resource"""
    return f"Resource echo: {message}"
@mcp.tool()
def echo_tool(message: str) -> str:
    """Echo a message as a tool"""
    return f"Tool echo: {message}" 
```

`FastMCP`类是你用来实例化服务器的东西——在这个例子中，一个名为`mcp`的实例。然后我们使用`@mcp`来定义资源、工具和提示。

这是一种高级构建服务器的方法，但如果你想要更多控制服务器构建的方式呢？以下是使用低级访问来实现这一点的办法。

让我们看看如何注册功能的不同之处。在过去，你可能习惯于使用与特定工具或资源等相关的装饰器。在低级服务器中有什么不同之处呢？你需要自己处理所有请求。而不是一次处理一个工具或资源，你需要在同一个地方处理与工具、资源和提示相关的所有请求。

首先，导入方式不同。看看我们是怎样从`from mcp.server.lowlevel`和`Server`导入的：

```py
from mcp.server.lowlevel import Server 
```

接着是实例化服务器，如下所示：

```py
server = Server("low-level-server") 
```

而不是使用`@mcp.tools()`或`@mcp.resource()`，你可以使用处理程序如`@server.list_tools()`和`@server.call_tool()`来注册。这是一个巨大的区别。区别在于，在低级服务器中，你不需要为每个功能都有一个装饰器，你需要自己处理所有工具请求，无论是调用工具还是列出工具、资源或提示。这意味着`@server.list_tools()`负责列出所有工具，你需要自己实现这个逻辑，而高级服务器会为你做这件事。以下是如何实现`@server.list_tools()`的一个例子：

```py
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    tool_list = []
    print(tools)
    for tool in tools.tools.values():
        tool_list.append(
            types.Tool(
                name=tool["name"],
                description=tool["description"],
                inputSchema=tool["input_schema"],
            )
        )
    return tool_list 
```

在前面的代码中，我们做了以下几件事：

+   定义了一个处理程序，`handle_list_tools`，它响应`list_tools`请求。

+   使用了`@server.list_tools()`装饰器将此处理程序注册到服务器上。

+   遍历`tools.tools.values()`来收集所有工具，并以所需格式返回它们——在这个例子中，是一个`types.Tool`对象的列表。在这种情况下，`tools.tools`是一个包含我们创建的服务器上注册的所有工具的字典，所以它看起来像这样：

    ```py
    {
        "tools": {
            "echo_tool": {
                "name": "echo_tool",
                "description": "Echo a message as a tool",
                "input_schema": {"type": "object", "properties":
                    {"message": {"type": "string"}}}
            }
        }
    } 
    ```

这不是增加了工作量吗？实际上，让我们在下一节中探讨一下，这可能是组织代码的一个很好的方法。

# 组织你的架构

使用低级服务器的一个巨大优势是你可以控制你服务器的架构。你可以以对你项目有意义的方式组织你的代码。例如，你可以在名为`tools`的文件夹中定义所有你的工具。工具也不需要知道服务器实例。听起来很有希望，对吧？让我们看看下一步。

你也可以用高级服务器很好地组织你的代码，但通常会更混乱，因为你需要传递服务器实例，就像我们在本章最初所说的那样。

到目前为止，你已经在高级服务器中这样定义了 MCP 服务器功能：

```py
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Echo")
@mcp.tool()
def echo_tool(message: str) -> str:
    """Echo a message as a tool"""
    return f"Tool echo: {message}"
@mcp.tool()
def add_tool(a: int, b: int) -> int:
    """Add two numbers as a tool"""
    return a + b
@mcp.tool()
def subtract_tool(a: int, b: int) -> int:
    """Subtract two numbers as a tool"""
    return a - b
@mcp.tool()
def multiply_tool(a: int, b: int) -> int:
    """Multiply two numbers as a tool"""
    return a * b
@mcp.tool()
def divide_tool(a: int, b: int) -> float:
    """Divide two numbers as a tool"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b 
```

定义一个服务器及其功能，如前面的示例，并没有什么不妥。然而，你可能倾向于保持你的入口点文件相对空（除了服务器定义之外）并定义所有功能在该文件中。因此，你可能会采用如下所示的项目结构：

```py
project/
├── server.py
├── tools.py 
```

然后，你的`server.py`可能看起来像这样：

```py
# server.py
from mcp.server.fastmcp import FastMCP
import tools
mcp = FastMCP("Echo")
tools.register_tools(mcp)
# code for running the server 
```

你的`tools.py`可能看起来像这样：

```py
from mcp.server.fastmcp import FastMCP
def register_tools(mcp: FastMCP):
    @mcp.tool()
    def echo_tool(message: str) -> str:
        """Echo a message as a tool"""
        return f"Tool echo: {message}"
    @mcp.tool()
    def add_tool(a: int, b: int) -> int:
        """Add two numbers as a tool"""
        return a + b
    @mcp.tool()
    def subtract_tool(a: int, b: int) -> int:
        """Subtract two numbers as a tool"""
        return a - b
    @mcp.tool()
    def multiply_tool(a: int, b: int) -> int:
        """Multiply two numbers as a tool"""
        return a * b
    @mcp.tool()
    def divide_tool(a: int, b: int) -> float:
        """Divide two numbers as a tool"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b 
```

你甚至可以为每个工具创建专门的文件：

```py
project/
├── server.py
├── tools/
│   ├── echo.py
│   ├── add.py
│   ├── subtract.py
│   ├── multiply.py
│   └── divide.py 
```

所以，是的，你可以以对你项目有意义的方式组织你的代码。很难摆脱传递服务器实例的需要。应该指出的是，高级服务器确实提供了一个`create_tool()`函数，允许你创建工具，而无需使用`@mcp.tool()`装饰器。然而，这是作者的观点，使用低级服务器来做这个目的更简单。让我们在下一节看看如何做到这一点。

## 在低级服务器中构建工具列表响应

那么，让我们看看低级访问是否可以帮助我们进一步改进我们的架构。目标是拥有一个服务器，它可以以易于维护和扩展的方式注册工具、资源和提示。

因此，到目前为止，我们关于低级服务器的说法如下：

+   代替使用`FastMCP`，你使用`mcp.server.lowlevel`中的`Server`类

+   你可以使用处理程序如`@server.list_tools()`和`@server.call_tool()`来注册功能，以处理列出所有工具和处理所有工具调用

让我们更详细地研究这些处理程序以及我们如何构建它们，看看我们如何利用它们：

```py
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    tool_list = []
    tool_list.append(
        types.Tool(
            name=tool["name"],
            description=tool["description"],
            inputSchema=tool["input_schema"],
        )
    )
    return tool_list 
```

在这里，我们观察到返回类型是`types.Tool`对象的列表。这在代码中得到了反映，我们像这样将一个`types.Tool`对象追加到`tool_list`中：

```py
tool_list.append(
    types.Tool(
        name=tool["name"],
        description=tool["description"],
        inputSchema=tool["input_schema"],
    )
) 
```

## 组织你的代码并创建工具和模式

现在我们知道了如何注册工具，让我们看看我们如何组织代码。目标是实现以下内容：

+   为每个工具创建一个文件，这样我们可以轻松地管理代码

+   在一个地方注册所有工具

听起来像是一个伟大的目标，对吧？谁不想有可维护性？让我们创建一个如下所示的文件夹结构：

```py
server.py
tools/
├── __init__.py
├── add.py 
```

我们看到的是，我们有一个`server.py`文件，它将成为我们服务器的入口点，一个包含所有工具的`tools/`文件夹。`tools/init.py`文件用于注册收集所有工具，`server.py`和`@mcp.list_tools()`将用于注册所有工具。让我们首先看看`__init__.py`：

```py
from .add import tool_add
tools = {
    tool_add["name"] : tool_add
} 
```

这看起来超级简单——只是一个字典，它从`add.py`文件中导入`tool_add`函数并将其添加到`tools`字典中。现在让我们看看`add.py`文件：

```py
# add.py
from .schema import AddInputModel
async def add_handler(args) -> float:
    try:
        # Validate input using Pydantic model
        input_model = AddInputModel(**args)
    except Exception as e:
        raise ValueError(f"Invalid input: {str(e)}")
    # TODO: add Pydantic, so we can create an AddInputModel and validate     args
    """Handler function for the add tool."""
    return float(input_model.a) + float(input_model.b)
tool_add = {
    "name": "add",
    "description": "Adds two numbers",
    "input_schema": AddInputModel,
    "handler": add_handler
} 
```

在前面的代码中，我们做了以下操作：

+   导入了`AddInputModel`并将其作为`input_schema`添加

+   定义了一个`add_handler`函数，它接受参数并返回它们的和

+   创建了一个包含工具名称、描述、输入模式和处理器函数的`tool_add`字典

如您所见，没有导入任何 MCP 相关的内容，因此看起来相当干净。

最后，让我们看看`schema.py`文件：

```py
from pydantic import BaseModel
class AddInputModel(BaseModel):
    a: float
    b: float 
```

在这里，我们使用**Pydantic**库来定义输入模型。随着我们向解决方案中添加更多工具，我们可以扩展此文件以包含新类型。

## 处理被调用的工具

到目前为止，你已经看到了我们如何处理要求列出所有工具的调用。但还有一个我们需要处理的情况，即当客户端尝试调用工具时。为此，我们需要对传入的工具调用请求执行以下操作：

+   识别要调用的工具。

+   解析参数，并在解析过程中验证它们。这正是我们的 Pydantic 模式将帮助我们的地方。

让我们从服务器上的请求开始：

```py
# server.py
@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, str] | None
) -> list[types.TextContent]:
    pass 
```

注意我们如何需要引用`@server.call_tool`装饰器，以及返回类型需要是`list[types.TextContext]`。

接下来，让我们看看我们能否识别出正确的工具：

```py
# server.py
if name not in tools.tools:
    raise ValueError(f"Unknown tool: {name}")

tool = tools.tools[name] 
```

好的，让我们准备调用工具上的处理程序并构造对调用客户端的响应：

```py
# server.py
try:
    result = await tool"handler"
except Exception as e:
    raise ValueError(f"Error calling tool {name}: {str(e)}")
return [
    types.TextContent(type="text", text=str(result))
] 
```

注意我们如何在`tool`对象上调用`handler`属性，并将`arguments`作为参数。然而，工具上的处理程序看起来是什么样子，它是如何工作的呢？

```py
# add.py
from .schema import AddInputModel
async def add_handler(args) -> float:
    try:
        # Validate input using Pydantic model
        input_model = AddInputModel(**args)
    except Exception as e:
        raise ValueError(f"Invalid input: {str(e)}")
    print (f"Adding {args['a']} and {args['b']}")
    """Handler function for the add tool."""
    return float(args['a']) + float(args['b']) 
```

在这里，我们使用`try-catch`将`args`传递给`AddInputModel`类。它接受一个字典，通过将其作为`**args`传递，我们将字典解包为关键字参数。如果输入无效，将引发`ValueError`，并显示一条消息说明出了什么问题。这样，您可以确保工具的输入始终有效，并符合预期的模式。

太好了，我们现在已经成功处理了列出所有工具和调用特定工具的情况，同时确保输入参数的某些验证也发生了。

我相信你可以进一步改进这个设置，但与最初的设置相比，这已经好多了。

# 摘要

在本章中，你学习了如何使用低级服务器为你的 MCP 服务器创建一个更易于维护的架构。你看到了如何注册工具、处理请求和使用模式验证输入。这种方法允许你轻松添加新工具并管理现有工具，使你的服务器更加灵活且易于维护。

在我们下一章中，我们将介绍如何构建可以与我们的 MCP 服务器交互的客户端。

# 作业

让我们看看我们能否组织我们在*第三章*中创建的电子商务服务器。以下是代码供参考。创建一个工具目录 - 使用 Pydantic 和低级 API。

```py
# server.py
from mcp.server.fastmcp import FastMCP
import uuid
# Create an MCP server
mcp = FastMCP("Demo")
class Customer:
    def __init__(self,id: int, name: str, email: str):
        self.id = id
        self.name = name
        self.email = email
class Category:
    def __init__(self, name: str, description: str):
        self.id = uuid.uuid4()
        self.name = name
        self.description = description
class Product:
    def __init__(self, name: str, price: float, description: str):
        self.name = name
        self.price = price
        self.description = description
class CartItem:
    def __init__(self, cart_id: int, product_id: int, quantity: int):
        if cart_id != 0:
            self.cart_id = cart_id
        else:
            self.cart_id = uuid.uuid4()
        self.product_id = product_id
        self.quantity = quantity
class Cart:
    def __init__(self, cart_id: int, customer_id: int):
        if cart_id != 0:
            self.cart_id = cart_id
        else:
            self.cart_id = uuid.uuid4()
        self.customer_id = customer_id
class Order:
    def __init__(self, order_id: int, customer_id: int):
        if order_id != 0:
            self.order_id = order_id
        else:
            self.order_id = uuid.uuid4()
        self.customer_id = customer_id
products = [
    Product("Product 1", 10.0, "Description of Product 1"),
    Product("Product 2", 20.0, "Description of Product 2"),
    Product("Product 3", 30.0, "Description of Product 3")
]
orders = [
    Order(1, 101),
    Order(0, 101),
    Order(0, 102)
]
carts = []
customers = [
    Customer(1, "Customer 1", "email")
]
categories = [
    Category("Category 1", "Description of Category 1"),
    Category("Category 2", "Description of Category 2"),
    Category("Category 3", "Description of Category 3")
]
product_catalog = [
    {
        "name": "Product 1",
        "price": 10.0,
        "description": "Description of Product 1",
        "category_id": 1
    },
    {
        "name": "Product 2",
        "price": 20.0,
        "description": "Description of Product 2",
        "category_id": 2
    },
    {
        "name": "Product 3",
        "price": 30.0,
        "description": "Description of Product 3",
        "category_id": 3
    }
]
# get orders
@mcp.tool()
def get_orders(customer_id:int = 0) -> [Order]:
    """get all orders"""
    if customer_id != 0 and not any(customer.id == customer_id for         customer in customers):
        raise ValueError(f"Invalid customer_id: {customer_id}")
    filtered_orders = orders
    if customer_id != 0:
        filtered_orders = [order for order in orders if order.customer_id             == customer_id]
    return [{"type": "text", "name": f"ID: {order.order_id},customer:         {order.customer_id}"} for order in filtered_orders]
# get order by id
@mcp.tool()
def get_order(order_id:int) -> Order:
    """get order by id"""
    for order in orders:
        if order.order_id == order_id:
            return {"type": "text", "name": f"ID: {order.order_                id},customer: {order.customer_id}"}
    return None
# place order
@mcp.tool()
def place_order(customer_id:int) -> Order:
    """place order"""
    if customer_id != 0 and not any(customer.id == customer_id for         customer in customers):
        raise ValueError(f"Invalid customer_id: {customer_id}")
    new_order = Order(0, customer_id)
    orders.append(new_order)
    return {"type": "text", "name": f"ID: {new_order.order_id},customer:         {new_order.customer_id}"}
# get carts
@mcp.tool()
def get_cart(customer_id:int) -> [Cart]:
    """get a singular cart"""
    if customer_id != 0 and not any(customer.id == customer_id for         customer in customers):
        raise ValueError(f"Invalid customer_id: {customer_id}")
    cart = next((cart for cart in carts if cart.customer_id == customer_        id), None)
    if cart:
        return {"type": "text", "name": f"ID: {cart.cart_id},customer:             {cart.customer_id}"}
    else:
        return None

# get cart items
@mcp.tool()
def get_cart_items(cart_id:int) -> [CartItem]:
    """get cart items"""
    cart_items = [item for item in carts if item.cart_id == cart_id]
    return [{"type": "text", "name": f"ID: {item.cart_id},product: {item.        product_id},quantity: {item.quantity}"} for item in cart_items]
# add to cart
@mcp.tool()
def add_to_cart(cart_id:int, product_id:int, quantity:int) -> CartItem:
    """add to cart"""
    new_cart_item = CartItem(cart_id, product_id, quantity)
    carts.append(new_cart_item)
    return {"type": "text", "name": f"ID: {new_cart_item.cart_id},product:         {new_cart_item.product_id},quantity: {new_cart_item.quantity}"}
# tool, all products
@mcp.tool()
def get_all_products() -> [Product]:
    """Get all products"""
    return [{"type": "text", "name": f"ID: {product.name},price: {product.        price},description: {product.description}"} for product in products]
# tool, product by id
@mcp.tool()
def get_product(product_id: int) -> Product:
    """Get product by ID"""
    for product in products:
        if product.name == product_id:
            return {"type": "text", "name": f"ID: {product.name},price:                 {product.price},description: {product.description}"}
    return None
# tool, all categories
@mcp.tool()
def get_all_categories() -> [Category]:
    """Get all categories"""
    return [{"type": "text", "name": f"ID: {category.name},description:         {category.description}"} for category in categories]
# tool, all customers
@mcp.tool()
def get_all_customers() -> [Customer]:
    """Get all customers"""
    return [{"type": "text", "name": f"ID: {customer.id},name: {customer.        name},email: {customer.email}"} for customer in customers] 
```

# 解决方案

您可以通过[`github.com/PacktPublishing/Learn-Model-Context-Protocol-with-Python/blob/main/Chapter06/solutions/README.md`](https://github.com/PacktPublishing/Learn-Model-Context-Protocol-with-Python/blob/main/Chapter0﻿6/solutions/README.md)访问解决方案。

# 问答

使用低级服务器的优点有哪些？

+   A: 你使用的内存更少

+   B: 你可以更好地控制请求的处理方式

+   C: 你可以定义自己的传输

您可以在[`github.com/PacktPublishing/Learn-Model-Context-Protocol-with-Python/blob/main/Chapter06/solutions/solution-quiz.md`](https://github.com/PacktPublishing/Learn-Model-Context-Protocol-with-Python/blob/main/Chapter06/solutions/solution-quiz.md)访问解决方案。

|

#### 现在解锁这本书的独家优惠

扫描此二维码或访问[`packtpub.com/unlock`](https://packtpub.com/unlock)，然后通过书名搜索此书。 | ![](img/Unlock-01.png)![](img/Unlock1.png) |

| **注意**：在开始之前准备好您的购买发票。* |
| --- |
