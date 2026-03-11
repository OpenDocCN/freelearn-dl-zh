# 7

# 使用 ChatGPT 进行 Salesforce Web Services 和 Callouts

Salesforce Web Services 和 Callouts 构成了 Salesforce 生态系统的重要组成部分，它们使得无缝集成和扩展功能成为可能，这些功能超越了标准 Salesforce 功能。在一个日益互联的世界中，不同系统需要相互通信，这些工具比以往任何时候都更加关键。让我们深入了解它们是什么以及它们是如何工作的。

在本章中，你将学习启用和优化 Salesforce 与外部系统之间通信渠道的基本知识。贯穿这些页面，你将牢固掌握 Web Services 和 Callouts 的基础元素、它们在 Salesforce 生态系统中的重要性以及有效设置它们的逐步流程。本章侧重于实用性，提供了一条途径，以优化这些集成，确保可靠性和性能，帮助你使用 ChatGPT 高效地解决常见问题。通过实际场景和案例研究，你将学习如何将用户故事转化为安全、健壮的集成，确保你的 Salesforce 环境能够与其他平台无缝通信，同时遵守最高标准的数据安全和完整性。

# 技术要求

当开始编写 Salesforce Apex 之旅时，选择开发工具和环境对于生产力和成功至关重要。Apex 是 Salesforce 的专有编程语言，旨在在 Lightning 平台上执行，并需要一套强大的工具来管理其生命周期，从开发到部署。

Apex 开发工具包的核心是 Salesforce **集成开发环境**（**IDE**）。最突出且广泛使用的 IDE 是带有 Salesforce 扩展包的 **Visual Studio Code**（**VS Code**）。这个由 Salesforce 正式认可的工具包为开发者提供了诸如语法高亮、自动完成、集成测试和调试功能等特性。它还提供了直接部署到 Salesforce org 以及版本控制集成，这对于维护代码质量和协作至关重要。开发者可以利用 VS Code 强大的源代码编辑器和定制选项，将他们的环境调整到特定项目需求。

开发环境中的另一个重要方面是 Salesforce 沙盒的使用。这些是 Salesforce 环境的隔离副本，开发者可以在不影响实时生产环境的情况下构建和测试他们的 Apex 代码。所使用的沙盒类型可能从适合独立编码和测试的开发者沙盒，到包含生产环境数据完整副本的完整复制沙盒，用于性能和用户验收测试。利用沙盒是一种最佳实践，因为它允许安全地开发和测试新功能、错误修复和改进。在这些沙盒中编写和测试的代码可以通过变更集或 Salesforce DX 推送到生产环境，具体取决于项目的复杂性和治理要求。

IntelliJ IDEA 配置了 Illuminated Cloud 插件是 Salesforce 开发的另一种替代 IDE。这个强大的组合为 Salesforce 开发提供了一整套定制的功能，包括智能补全、代码导航、集成的 SOQL 查询构建器等，尽管它更受那些已经对 JetBrains 的 IDE 套件有偏好倾向的开发者的青睐。无论选择 VS Code 还是 IntelliJ，这些 IDE 与版本控制系统（如 Git）的集成对于协作和可扩展的开发实践至关重要。这些 IDE 的工具集，得益于其元数据感知能力，确保开发者能够有效地管理源代码、跟踪变更并处理复杂的部署流程，从而为 Salesforce Apex 提供了一个更加流畅和高效的开发生命周期。

# Salesforce Web 服务

Salesforce Web 服务通过标准的 SOAP 或 REST 协议公开自定义逻辑。它们允许外部系统直接在 Salesforce 数据上执行查询、搜索和 CRUD 操作。这些 Web 服务在 Apex 代码中定义，并可定制以提供针对性的功能。例如，你可能有一个触发特定工作流或根据复杂业务逻辑更新记录的 Web 服务。Salesforce Web 服务的美妙之处在于它们使 Salesforce 平台可扩展且开放，便于各种外部交互。

Salesforce 中的 Web 服务允许不同的应用程序通过互联网相互通信。它们在 Salesforce 与其他系统之间的互操作性和集成中发挥着关键作用。以下是一些核心概念：

+   **简单对象访问协议**（**SOAP**）：一种用于实现 Web 服务的结构化信息交换协议。SOAP Web 服务使用 XML 消息，并且是强类型的。

+   **表征状态转移**（**REST**）：一种软件架构风格，使用标准的 HTTP 方法。REST Web 服务使用 JSON 或 XML 消息，并且是松散类型的。

+   `global` 访问修饰符以及如 `@WebService` 用于 SOAP 或 `@RestResource(urlMapping='/yourUrl')` 用于 REST 的注解。

+   **Web 服务描述语言**（**WSDL**）：如果您正在使用或公开 SOAP Web 服务，您将遇到 WSDL 文件，这是一个描述如何调用 Web 服务、它期望的参数以及它返回的数据结构的 XML 文档。

+   `https://yourInstance.salesforce.com/services/apexrest/yourUrl`。

+   **身份验证和授权**：安全性通常是任何 Web 服务的关键部分。Salesforce 提供了各种方式来验证 Web 服务调用，例如 OAuth 2.0、会话 ID 和基本身份验证。这确保只有授权用户才能访问或操作数据。

+   **Governor 限制**：Salesforce 对资源（如内存、CPU 时间、检索或修改的记录数）实施多个 Governor 限制。这些限制也适用于 Web 服务，因此编写高效的代码至关重要。

+   **异常处理**：一个健壮的 Web 服务应该能够优雅地处理异常，并向客户端返回有意义的错误消息。Salesforce 允许自定义异常类型和错误合同来有效地处理错误。

+   `@future` 方法，允许耗时的操作在后台处理，从而使您的 Web 服务更具可扩展性和用户友好性。

理解这些核心概念对于在 Salesforce 中使用外部 Web 服务以及将 Salesforce 数据和操作作为 Web 服务公开给外部系统至关重要。

# Salesforce Apex 调用。

相反，Salesforce Apex 调用本质上是对外部服务的出站请求。它们使 Salesforce 能够与外部 API 通信，无论是基于 REST 还是 SOAP。当 Salesforce 需要从外部服务获取数据或 Salesforce 中的操作应触发另一个系统中的操作时，调用至关重要。例如，当新的潜在客户添加到 Salesforce 时，调用可以自动调用外部电子邮件营销工具发送欢迎邮件。

理解 Salesforce Web 服务和调用对于任何希望创建深度集成、自动化和高效系统的人来说至关重要。无论是将 Salesforce 与 ERP 系统同步、实时分析数据流还是自动化跨系统工作流，这些工具都提供了您需要的灵活性和功能。

来自 Web 服务的类似概念可以应用于 Salesforce 调用。在 Salesforce 中，调用指的是从您的 Salesforce 组织向外部服务器发起 Web 服务请求的行为。这通常是为了从 Salesforce 向其他平台检索或发送数据，或者将 Salesforce 与其他系统集成。以下是与 Salesforce 调用相关的核心概念：

+   **HTTP 方法**：Salesforce 支持各种 HTTP 方法，如 GET、POST、PUT、PATCH 和 DELETE，用于进行调用。方法的选择取决于你打算在外部系统上执行的操作。

+   **命名凭据**：命名凭据是存储调用所需端点和身份验证细节的一种更安全、更方便的方式。它抽象化了端点，简化了调用过程，尤其是在维护代码时。

+   使用`HttpRequest`、`HttpResponse`和`Http`类创建和执行 HTTP 调用。这些类允许你设置头信息、指定 HTTP 方法并处理响应。

+   **身份验证**：当与外部服务集成时，你通常需要验证你的请求。Salesforce 支持各种身份验证形式，如基本身份验证、OAuth 或 API 密钥。这些通常可以在 HTTP 头或 URL 中设置。

+   `@future`注解或`Queueable`接口。异步调用对于可能需要很长时间才能完成且不希望阻塞用户界面的操作非常有用。

+   **SOAP 和 REST 调用**：你可以从 Salesforce 进行 SOAP 和 REST 调用。SOAP 调用使用基于 XML 的消息传递，通常使用 WSDL 文件进行结构化。REST 调用更灵活，通常使用 JSON 或 XML 作为消息格式。

+   **限制器限制**：Salesforce 对外部调用的数量和持续时间实施限制器限制。对于同步事务，你可以进行最多 100 次调用，每次调用最多可占用 120 秒。异步调用有不同的限制。

+   **错误处理**：有效的错误处理对于流畅的用户体验至关重要。Salesforce 允许你在调用过程中捕获和处理异常，使你能够返回用户友好的错误消息，并在必要时实现重试逻辑。

+   **响应解析**：一旦你从外部服务器收到响应，你通常需要解析此响应以提取有用信息。Salesforce 提供了内置支持，用于解析 JSON 和 XML 响应。

理解这些核心概念可以显著提高在 Salesforce 中进行调用的效率和可靠性。

ChatGPT 可以通过多种方式为在多个方面与 Web 服务和调用一起工作的 Salesforce 开发者提供重大帮助。让我们来看看。

## 代码片段和原型

开发 Web 服务或调用涉及编写 Apex 代码、制作 HTTP 请求或与 SOAP/REST 协议一起工作。ChatGPT 可以为特定场景生成代码片段和原型示例。这有助于开发者了解如何设置 Web 服务或执行调用，节省可能花费在研究或初始代码编写上的时间。

## 调试辅助

在开发或维护 Web 服务和调用时，常常会出现问题——无论是请求/响应格式不良、授权问题还是逻辑错误。ChatGPT 可以帮助诊断常见问题并提供调试建议。虽然它不能直接访问你的 Salesforce 实例为你进行调试，但它可以引导你进行思考过程，并根据你描述的症状推荐可能的解决方案。

## 最佳实践

Salesforce 有关于高效和安全实施 Web 服务和调用的指南和最佳实践。ChatGPT 可以提供关于这些最佳实践的见解，例如处理限制器限制、确保容错性或管理 API 版本，帮助你构建更健壮和可扩展的解决方案。

### 审查和重构建议

假设你已经为网络服务或调用编写了一些代码。在这种情况下，ChatGPT 可以帮助你逻辑上审查该代码，并提出重构建议以改进性能、可维护性或可读性。这可能包括优化 SOQL 查询、更优雅地处理异常或模块化代码以提高可重用性。

### 交互式学习

ChatGPT 可以模拟涉及网络服务和调用的真实世界场景。这使得经验较少的开发者可以通过交互式对话了解这些服务的生命周期，包括请求和响应处理、数据处理和错误管理。

通过在这些领域提供帮助，ChatGPT 成为了 Salesforce 开发者的宝贵伴侣，使他们能够编写更好的代码、更快地调试和理解得更深入。对于可能没有立即获得 Salesforce 网络服务和调用专业知识的独立开发者或小型团队来说，它尤其有用。

# 在 ChatGPT 的帮助下设置网络服务

考虑以下简单的用户故事示例，允许外部库存系统更新库存水平。

用户故事 – *集成 Salesforce 与外部* *库存系统*：

+   **目标**：作为一名后端开发者，我希望在 Salesforce 中创建一个网络服务，以便外部库存管理系统可以在商品库存水平发生变化时更新 Salesforce。

+   **验收标准**：

    +   **给定** 外部库存管理系统对商品库存水平进行了更新

    +   **当** 外部系统发送包含更新库存水平和商品 ID 的 POST 请求时

    +   **然后** Salesforce 网络服务应使用新的库存水平更新产品对象中的相应记录

    +   **给定** 外部系统发送无效的商品 ID 或库存水平

    +   **当** Salesforce 网络服务接收到请求时

*然后** 网络服务应响应适当的错误消息和 HTTP 状态码，指示问题

*给定** 网络服务已成功更新 Salesforce 记录

*当**操作完成时**

*然后** 网络服务应响应成功消息和 HTTP 状态码，指示更新成功

我们可以将用户与请求 ChatGPT 编写代码和配置指南结合起来：

| **提示** | 你能请写这个用户故事的代码和配置吗？用户故事 - 将 Salesforce 与外部库存系统集成。

+   **目标**：作为一个后端开发者，我想在 Salesforce 中创建一个网络服务，以便外部库存管理系统可以在商品库存水平发生变化时更新 Salesforce。

+   **验收标准**：

    +   **给定**外部库存管理系统对某个商品的库存水平进行了更新

    +   **当**外部系统发送带有更新后的库存水平和商品 ID 的 POST 请求时

    +   **然后**Salesforce 网络服务应该使用新的库存水平更新产品对象中的相应记录

    +   **给定**外部系统发送了一个无效的商品 ID 或库存水平

    +   **当**Salesforce 网络服务收到请求时

    +   **然后**网络服务应该响应一个适当的错误消息和 HTTP 状态码，指示问题

|

|  |
| --- |

+   **给定**网络服务已成功更新 Salesforce 记录

+   **当**操作完成时

+   **然后**网络服务应该响应一个成功消息和 HTTP 状态码，指示更新成功

|

| **响应** | 当然！以下是一个示例 Apex 类和方法，它将作为处理来自外部库存管理系统的库存水平更新的网络服务。此代码旨在说明用户故事如何实现；它尚未在实时 Salesforce 环境中进行测试。**网络服务 Apex 类**：

```py
@RestResource(urlMapping='/inventory/')
```

```py
global with sharing class InventoryWebService {
```

```py
    @HttpPost
```

```py
    global static String updateStockLevel() {
```

```py
        RestRequest req = RestContext.request;
```

```py
        RestResponse res = RestContext.response;
```

```py
        // Parse JSON payload from request
```

```py
        Map<String, Object> requestBody = (Map<String, Object>) JSON.deserializeUntyped(req.requestBody.toString());
```

```py
        // Extract the Item ID and new stock level
```

```py
        String itemId = (String) requestBody.get('itemId');
```

```py
        Integer newStockLevel = Integer.valueOf(requestBody.get('newStockLevel'));
```

```py
        // Validate Item ID and Stock Level
```

```py
        if(String.isBlank(itemId) &#124;&#124; newStockLevel == null &#124;&#124; newStockLevel < 0) {
```

```py
            res.statusCode = 400;
```

```py
            return 'Bad Request: Invalid Item ID or Stock Level';
```

```py
        }
```

```py
        // Query the Product object to find the matching record
```

```py
        List<Product__c> products = [SELECT Id, Stock_Level__c FROM Product__c WHERE Id = :itemId LIMIT 1];
```

|

| |
| --- |

```py
        // Check if the Product exists
```

```py
        if(products.isEmpty()) {
```

```py
            res.statusCode = 404;
```

```py
            return 'Not Found: Item ID not found';
```

```py
        }
```

```py
        // Update Stock Level
```

```py
        Product__c productToUpdate = products[0];
```

```py
        productToUpdate.Stock_Level__c = newStockLevel;
```

```py
        update productToUpdate;
```

```py
        // Return Success
```

```py
        res.statusCode = 200;
```

```py
        return 'Success: Stock Level Updated';
```

```py
    }
```

```py
}
```

配置步骤：1. Product__c 对象应该存在于您的 Salesforce 架构中，并有一个名为 Stock_Level__c 的字段来存储库存水平。此字段应为数字或整数类型。2. 确保 Apex 类具有与将与此网络服务交互的配置文件适当的权限。3. 将网络服务暴露给外部系统，并共享 REST 端点 URL。4. 确保外部系统向此端点发送包含 itemId 和 newStockLevel 的 JSON 有效负载的 POST 请求。此代码和配置应有助于您满足用户故事中指定的验收标准。|

如果我们花点时间分析 ChatGPT 的响应，我们会看到它包括了代码和配置步骤。它做出了几个好的决定，例如包括一些与数据一致性相关的验证和错误，并且它没有选择一个不适合这个业务要求的标准 Salesforce 对象，如`Product2`。

作为一个小问题，ChatGPT 建议将`库存水平`字段设置为`数字`或`整数`类型。虽然`数字`是可行的，但在 Salesforce 中`整数`不是字段类型。

现在，让我们请求测试类和 JSON 以测试其是否工作：

| **提示** | 你也能提供测试类和示例 JSON 来测试它吗？ |
| --- | --- |

| **响应** | 绝对可以！以下是一个用于测试 Web 服务的 Apex 测试类，以及一个可以用于测试 Web 服务的示例 JSON 有效负载。**Apex** **测试类**：

```py
@IsTest
```

```py
public class InventoryWebServiceTest {
```

```py
    @IsTest static void testUpdateStockLevel() {
```

```py
        // Create a test Product record
```

```py
        Product__c testProduct = new Product__c(Name='TestProduct', Stock_Level__c=10);
```

```py
        insert testProduct;
```

```py
        // Prepare the test request and response
```

```py
        RestRequest req = new RestRequest();
```

```py
        RestResponse res = new RestResponse();
```

```py
        req.requestURI = '/services/apexrest/inventory/';
```

```py
        req.httpMethod = 'POST';
```

```py
        req.requestBody = Blob.valueOf(JSON.serialize(new Map<String, Object>{
```

```py
            'itemId' => testProduct.Id,
```

```py
            'newStockLevel' => 20
```

```py
        }));
```

```py
        // Set up test context
```

```py
        RestContext.request = req;
```

```py
        RestContext.response = res;
```

```py
        // Execute the web service method
```

```py
        String responseMessage = InventoryWebService.updateStockLevel();
```

```py
        // Asserts
```

```py
        System.assertEquals('Success: Stock Level Updated', responseMessage);
```

```py
        System.assertEquals(200, res.statusCode);
```

|

| |
| --- |

```py
        // Query the updated record
```

```py
        Product__c updatedProduct = [SELECT Stock_Level__c FROM Product__c WHERE Id = :testProduct.Id LIMIT 1];
```

```py
        System.assertEquals(20, updatedProduct.Stock_Level__c);
```

```py
    }
```

```py
}
```

**示例** **JSON 有效负载**以下是一个示例 JSON 有效负载，您可以使用它通过工具（如 Postman）测试此 Web 服务：

```py
{
```

```py
    "itemId": "a00xxxxxxxxxxxx",  // Replace with actual Product__c record Id
```

```py
    "newStockLevel": 20
```

```py
}
```

简单地将 itemId 值替换为 Salesforce 实例中的实际 Product__c 记录 ID，并将 newStockLevel 设置为要更新的库存水平。记住，前面的示例和测试类假设您有一个自定义对象 Product__c，其中包含一个自定义字段 Stock_Level__c。请确保调整代码以适应您实际的 Salesforce 架构。

在这种情况下，ChatGPT 的学习方法甚至有助于操作示例 JSON，这对于不熟悉 Salesforce 功能功能的测试人员来说可能非常有用。

当此代码和测试类部署到开发者沙箱时，它们可以正常工作。然而，测试类只覆盖了 77%的 Web 服务代码，因为它只测试了正面的结果。本书的作者将此留给你作为使用 ChatGPT 将测试类覆盖率提高到 95%的练习。

# 使用 ChatGPT 优化和管理调用

在 Salesforce 中有效地管理调用对于确保最佳性能、可扩展性和可靠性至关重要。以下是管理调用的最佳实践：

+   **批量处理调用**：在可能的情况下，设计调用以便它们一次处理多个记录。这将帮助您保持在单个事务中调用次数的 Salesforce 限制之内。

+   **异步处理**：使用异步 Apex，如 Future 方法或 Queueable Apex，来执行调用。这允许调用在后台执行，从而不会阻塞用户界面并提高整体用户体验。

+   **错误处理**：实现健壮的错误处理机制来管理超时、状态码以及调用过程中可能出现的任何其他问题。始终检查响应状态并优雅地处理异常。

+   **可重用代码**：创建可重用的工具类和方法来制作调用。这有助于提高代码的可维护性和更容易更新。

+   **超时**：注意调用超时。默认值为 10 秒，但您可以将其设置为最多 120 秒。

+   **调用模式**：对于更复杂的调用，考虑使用设计模式，如 Singleton 用于共享配置或 Facade 用于简化复杂的调用。

+   **命名凭据**：使用命名凭据来安全地存储端点和认证设置。这使得管理敏感信息变得更加容易。

+   **单元测试**：编写全面的测试方法以覆盖各种场景，包括错误条件。使用模拟来模拟外部系统的行为。

+   **监控和日志记录**：实现日志记录机制以捕获有关调用及其响应的基本信息。这可以用于调试和审计目的。

+   **API 限制**：始终关注 API 限制，尤其是如果你正在使用具有自身限制的外部服务。

我们不会在这本书中涵盖所有内容；我们只会关注调用模式和可重用代码。但所有这些都需要记住，以确保组织的性能不受负面影响。

让我们从调用模式开始，并假设自己是一个不确定最佳模式的开发者：

| **提示** | 你能帮我理解 Salesforce 中的调用模式吗？ |
| --- | --- |

| **响应** | 当然可以！Salesforce 中的调用模式本质上是一种设计策略，用于管理对外部服务的调用。有效的调用模式可以使你的集成更高效、更易于维护和更可靠。在这里，我概述了一个在 Salesforce 中常用的高性能调用模式。**批处理和队列模式****概述**：此模式结合了批处理和 Queueable Apex 来批量异步执行调用。当你需要执行大量调用或处理大量数据时，特别有用。以下是其工作原理：

+   Batch Apex 将记录分组为小块

+   每个记录块都传递给 Queueable Apex，它执行调用

|

|  |
| --- |

+   Queueable Apex 可以自我链式调用，允许异步按顺序处理多个调用

**实施步骤**：

+   **批处理类**：创建一个查询必要 Salesforce 记录的 Batch Apex 类：

    ```py
        public class MyBatchClass implements Database.Batchable<sObject>{
            public Database.QueryLocator start(Database.BatchableContext bc){
                return Database.getQueryLocator('SELECT Id, Name FROM MyObject__c WHERE SomeField__c = SomeValue');
            }
            public void execute(Database.BatchableContext bc, List<sObject> records){
                List<Id> recordIds = new List<Id>();
                for(sObject record : records){
                    recordIds.add(record.Id);
                }
                System.enqueueJob(new MyQueueableClass(recordIds));
            }
            public void finish(Database.BatchableContext bc){
                // Post-processing code
            }
        }
    ```

+   **队列类**：创建一个执行调用的 Queueable Apex 类：

    ```py
        public class MyQueueableClass implements Queueable {
            private List<Id> recordIds;
            public MyQueueableClass(List<Id> recordIds){
                this.recordIds = recordIds;
            }
            public void execute(QueueableContext context) {
                // Perform callout logic here
            }
        }
    ```

|

|  | 优点：

+   批量处理：通过使用 Batch Apex，你可以批量处理记录，从而优化对外部服务的 API 调用

+   异步：Queueable Apex 的异步特性允许在后台进行调用

+   错误处理：你可以在 Queueable 类中实现强大的错误处理，捕获问题并可能重试调用

考虑事项：

+   API 限制：注意 Salesforce 和你正在调用的外部服务的限制

+   执行顺序：Queueable 作业是异步执行的，不一定按照入队顺序执行

通过遵循这种模式，你可以优化 Salesforce 调用的性能和可靠性。|

不建议在没有真正必要的情况下结合使用批处理和 Queueable Apex。这强调了使用 ChatGPT 作为指南并调查/验证基于 ChatGPT 响应所做的决定的重要性。

注意，ChatGPT 的响应，它来自一个缺乏详细信息的提示，特别详细和全面。提示没有指定用例，也没有明确数据同步或数量的特定要求。尽管如此，ChatGPT 选择了一个适合大量数据调用的调用模式——这个决定在反思时与查询的隐含需求完全一致。为了在未来的查询中提高清晰度和精确度，提供更具体的细节将是有益的：

| **提示** | 你能在 Salesforce 中编写一个代码模式来管理对具有 REST API 的外部系统的实时调用？我们预计每天的调用量较低，但 JSON 会变化。 |
| --- | --- |
| **响应** | 当然可以！当你处理对低量级外部 REST API 的实时调用，并且预期 JSON 结构会频繁变化时，你可能希望使用灵活且模块化的设计。以下是一个示例代码模式，概述了执行实时调用的方法。代码模式： |

|  | Apex 类用于 HTTP 调用：|

```py
public class RealTimeCalloutService {
```

```py
    public static HttpResponse makeCallout(String endPoint, String httpMethod, String jsonBody) {
```

```py
        Http http = new Http();
```

```py
        HttpRequest req = new HttpRequest();
```

```py
        HttpResponse res = new HttpResponse();
```

```py
        req.setEndpoint(endPoint);
```

```py
        req.setMethod(httpMethod);
```

```py
        req.setHeader('Content-Type', 'application/json');
```

```py
        req.setBody(jsonBody);
```

```py
        try {
```

```py
            res = http.send(req);
```

```py
        } catch(Exception e) {
```

```py
            // Handle callout exception here
```

```py
            System.debug('Callout error: ' + e.getMessage());
```

```py
        }
```

```py
        return res;
```

```py
    }
```

```py
}
```

利用自定义元数据或自定义设置：鉴于 JSON 可能会频繁变化，你可以在自定义元数据或自定义设置中存储 JSON 模式或映射。这允许管理员在不更改代码的情况下进行更改。动态 JSON 解析：为了处理变化的 JSON 结构，使用 Salesforce 的 JSON.deserializeUntyped 方法来动态解析：

```py
Map<String, Object> parsedResponse = (Map<String, Object>) JSON.deserializeUntyped(response.getBody());
```

|

|  | 调用调用的示例触发器：|

```py
 trigger MyRealTimeTrigger on MyObject__c (after insert, after update) {
```

```py
    List<String> jsonBodies = new List<String>();
```

```py
    for (MyObject__c obj : Trigger.new) {
```

```py
        // Prepare JSON
```

```py
        String jsonBody = '{ "field1": "' + String.escapeSingleQuotes(obj.Field1__c) + '", "field2": "' + String.escapeSingleQuotes(obj.Field2__c) + '"}';
```

```py
        jsonBodies.add(jsonBody);
```

```py
    }
```

```py
    // Call future method if there are records to process
```

```py
    if (!jsonBodies.isEmpty()) {
```

```py
        CalloutHandler.makeCallouts(jsonBodies);
```

```py
    }
```

```py
}
```

考虑事项：错误处理：添加错误处理逻辑，捕获调用返回的任何异常或错误状态代码处理限制：由于这是一个实时调用，确保你了解同步调用限制模式更改：如果预期模式会频繁更改，考虑使用自定义元数据或自定义设置来动态管理这些更改，而无需更改代码。使用此模式，你可以以更模块化和动态的方式管理实时调用。|

越好的 ChatGPT 提示，ChatGPT 的响应就越好。ChatGPT 提供了一种理想的实时调用模式，帮助动态创建 JSON，并提出使用自定义元数据或自定义设置的建议，以帮助处理问题的动态特性。

我们可以更进一步，要求一个管理调用的框架：

| **提示** | 你能提供一个可重用的调用框架吗？ |
| --- | --- |

| **响应** | 当然可以！为调用创建一个可重用的框架可以使你的 Salesforce 实现更加易于维护和灵活。以下是一个使用接口和类处理 REST API 调用的简化示例。此模式旨在处理低量级、实时调用，并可能变化的 JSON 负载，符合你的要求：

```py
// ICalloutService Interface
```

```py
public interface ICalloutService {
```

```py
    HttpResponse makeCallout(Map<String, Object> params);
```

```py
}
```

```py
// Abstract Callout Service
```

```py
public abstract class AbstractCalloutService implements ICalloutService {
```

```py
    public HttpRequest createRequest(String endpoint, String method, String body) {
```

```py
        HttpRequest req = new HttpRequest();
```

```py
        req.setEndpoint(endpoint);
```

```py
        req.setMethod(method);
```

```py
        req.setHeader('Content-Type', 'application/json;charset=UTF-8');
```

```py
        if (String.isNotBlank(body)) {
```

```py
            req.setBody(body);
```

```py
        }
```

```py
        return req;
```

```py
    }
```

```py
}
```

```py
// Concrete Callout Service for Inventory Management
```

```py
public class InventoryCalloutService extends AbstractCalloutService {
```

```py
    public HttpResponse makeCallout(Map<String, Object> params) {
```

```py
        String endpoint = 'https://api.inventory.com/items/' + params.get('itemId');
```

```py
        String body = JSON.serialize(params);
```

```py
        HttpRequest req = createRequest(endpoint, 'POST', body);
```

```py
        Http http = new Http();
```

```py
        HttpResponse res = http.send(req);
```

```py
        return res;
```

|

|  |
| --- |

```py
    }
```

```py
}
```

```py
// Handler class to determine which service to use
```

```py
public class CalloutHandler {
```

```py
    public static HttpResponse executeCallout(String serviceName, Map<String, Object> params) {
```

```py
        ICalloutService service;
```

```py
        if (serviceName == 'Inventory') {
```

```py
            service = new InventoryCalloutService();
```

```py
        }
```

```py
        // Add more services as needed
```

```py
        if (service != null) {
```

```py
            return service.makeCallout(params);
```

```py
        } else {
```

```py
            return null;
```

```py
        }
```

```py
    }
```

```py
}
```

```py
// Sample usage in some Apex code
```

```py
public class MyController {
```

```py
    public void updateInventory() {
```

```py
        Map<String, Object> params = new Map<String, Object>{
```

```py
            'itemId' => '123',
```

```py
            'stockLevel' => 50
```

```py
        };
```

```py
        HttpResponse res = CalloutHandler.executeCallout('Inventory', params);
```

```py
        // Handle response
```

```py
    }
```

```py
}
```

**如何** **使用**：

+   `ICalloutService`接口定义了所有调用服务的合同

+   `AbstractCalloutService`提供了一个基本的`HttpRequest`创建方法

+   实际实现调用逻辑的具体类，例如`InventoryCalloutService`

|

|  |
| --- |

+   `CalloutHandler`类根据某些条件（这里是指`serviceName`）将调用委托给适当的服务

+   `MyController`类展示了如何使用此框架进行调用

使用这种模式，您可以通过创建实现`ICalloutService`的新类来轻松添加更多类型的调用。这使得管理 JSON 有效负载结构的更改或甚至完全不同类型的调用变得简单。|

在 Salesforce 中使用 Apex 框架可以显著简化开发过程，通过提供结构化、可重用的代码库，这可以导致一致且可维护的编码实践。Apex 框架还可以提高效率，让开发者能够专注于独特的业务逻辑，而不是样板代码，并且由于它们的标准化架构，便于更容易地进行调试和测试。这些框架通常包含内置的最佳实践，确保解决方案不仅稳健，而且可扩展，并与 Salesforce 不断发展的生态系统保持一致。

# 调试网络服务和调用问题

ChatGPT 可以作为开发者解决 Salesforce 中网络服务和调用问题的一个宝贵资源。无论您是在请求格式、HTTP 方法选择、头部配置还是响应处理方面遇到挑战，ChatGPT 都可以引导您通过常见的陷阱，并提供最佳实践的建议。

例如，如果您遇到意外的 HTTP 状态码，ChatGPT 可以帮助您了解该状态码的可能原因以及解决问题的方法。同样，如果您的 SOAP 或 REST 调用没有按预期工作，ChatGPT 可以提供关于常见问题（如格式错误的 XML/JSON、未经授权的访问或超出限制）的见解，并提出调试或重构代码的建议。

此外，正如我们之前所看到的，ChatGPT 可以帮助您编写调用测试类，这是 Salesforce 开发中经常被忽视但至关重要的一个部分。通过提供代码示例或指出可以实施模拟类和方法的位置，ChatGPT 增强了您产生稳健、无错误的服务的功能。这个工具不仅具有反应性，帮助您解决问题，而且具有主动性，为您提供知识和最佳实践，以避免未来的问题。

凭借其先进的功能和广泛的知识库，ChatGPT 旨在成为您解决网络服务和调用问题的可靠助手，最终帮助您构建更稳健、更高效的 Salesforce 应用程序。

让我们从一些格式错误的 JSON 示例开始：

```py
{
    "name": "John",
    "age": 30,
    "email": john.doe@domain.com  // Invalid because the email is not enclosed in quotes
}
```

在这里，我们通过创建我们的客户 John 来测试我们的网络服务，但它没有正确工作，并且响应包含典型的错误：格式错误的 JSON。ChatGPT 可以轻松地指引我们正确的方向：

| **提示** | 你能帮我理解为什么以下 JSON 格式不正确，并给我提供一个格式正确的 JSON 吗？

```py
{
```

```py
    "name": "John",
```

```py
    "age": 30,
```

```py
    "email": john.doe@domain.com
```

```py
 }
```

|

| **响应** | 当然可以！提供的 JSON 中的问题是电子邮件值 `john.doe@domain.com` 没有被双引号包围，这是 JSON 中字符串值所必需的。因为这个原因，JSON 解析器将无法理解这个值，并且它将被视为格式不正确。以下是修正后的版本：

```py
{
```

```py
    "name": "John",
```

```py
    "age": 30,
```

```py
    "email": "john.doe@domain.com"
```

```py
}
```

在这个版本中，所有字符串值和属性名都被双引号包围，使其成为一个格式正确的 JSON 对象。 |

对于 ChatGPT 来说，提出一个好的例子并解释问题的根本原因很容易。但为什么我们不更进一步呢？处理调用时常见的一个问题是收到以下错误：

`错误 - '你有未提交的工作'`

ChatGPT 可以帮助识别问题并提出解决方案：

| **提示** | 为什么我会收到这个错误：`错误 - '你有未提交的工作'`？ |
| --- | --- |

| **响应** | 错误“你有未提交的工作”通常在 Salesforce 中发生，当你在同一交易中执行 DML 操作后尝试进行调用时。Salesforce 不允许你在同一交易中进行任何 DML 操作，以确保数据一致性。以下是一些可能发生这种情况的典型场景：

+   你执行一个 DML 操作（插入、更新、删除等），然后立即尝试进行 HTTP 调用。

+   你开始一个数据库事务，执行一些操作，然后尝试在提交事务之前进行调用。

为了解决这个问题，你有几个选择：

+   `@future` 方法将调用移动到单独的交易中，允许在调用发生之前提交 DML 操作。但请记住，`@future` 方法有资源限制：

    ```py
        @future(callout=true)
        public static void makeCalloutMethod() {
            // your callout logic
        }
    ```

+   `@future`，可排队 Apex 在单独的交易中运行。这提供了更多的灵活性，并且是未来方法的现代替代品：

    ```py
        public class MyQueueable implements Queueable, Database.AllowsCallouts {
            public void execute(QueueableContext context) {
                // your callout logic
            }
        }
    ```

+   **重新排列你的逻辑**：有时，你可以重新排列你的代码逻辑，在所有调用完成后执行所有 DML 操作。

+   **首先执行调用**：如果可能的话，先执行调用，然后进行 DML 操作。

通过确保 DML 操作和调用分离到不同的交易中，你可以避免这个错误。 |

有了这些，我们可以总结一下我们学到的内容。

# 摘要

在本章中，你丰富了在 Salesforce 生态系统中对 Web 服务和调用的理解，学习了如何有效地设置、优化和排除这些关键集成功能的技术。你掌握了核心概念，精确地配置了 Web 服务，并探索了提高性能和可靠性的优化技术。ChatGPT 提供的指导使你能够迅速识别和解决常见的集成问题，确保你的 Salesforce 解决方案能够与其他系统无缝通信，同时遵守稳健的安全协议。你将用户故事转化为成功的 Salesforce 集成，从而提高了你在管理复杂数字交互方面的熟练度和信心。

在下一章中，你将深入探索 Salesforce 触发器的领域，了解它们在自动化流程中的关键作用以及 ChatGPT 如何提升其创建和维护。本章将引导你了解触发器设计的复杂性，借助 ChatGPT 的见解确保这些触发器不仅有效，而且遵循最佳实践。你将学习优化和重构现有触发器以实现最佳性能的关键技能，并获得宝贵的故障排除技术，以便迅速处理任何出现的错误。本章将传授基础知识，同时结合设计、优化和调试触发器的能力，从而提高 Salesforce 环境的响应性和可靠性。
