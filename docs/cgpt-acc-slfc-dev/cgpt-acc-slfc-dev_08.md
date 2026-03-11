

# 第八章：使用 ChatGPT 进行 Salesforce 触发器

在 Salesforce 的世界中，`触发器` 突出作为基本构建块之一，作为自动化流程启动的前锋。简单来说，Salesforce 的 `触发器` 是一段代码，它在 Salesforce 记录的特定事件（如插入、更新或删除）之前或之后自动执行。

触发器的主要功能是在数据创建、更新、删除或恢复时，管理、修改和简化 Salesforce 平台内的数据。以下是它们的工作方式：想象一下 Salesforce 中的一个操作，比如更新联系人的信息。在这个事件发生之前或之后，触发器可以介入，执行预定义的逻辑。例如，如果联系人的城市发生变化，触发器可能会自动更新他们的相应邮寄地址。

触发器的启动依赖于各种记录操作。以下是主要的催化剂：

+   `INSERT` 或 `UPDATE`) 发生时。这种预防性方法对于确保数据质量和一致性特别有用。

+   **后触发器**: 一旦数据操作完成并且数据已提交到数据库，"后"触发器就会启动。它们通常用于访问新添加或修改的记录并对它们执行操作，例如更新相关记录或与外部系统交互。

+   `插入`: 当添加新记录时

+   `更新`: 当记录被修改时

+   `删除`: 当记录被移除时

+   `恢复删除`: 当删除的记录被恢复时

+   `合并`: 当记录被合并时

Salesforce 触发器充当平台内数据操作的哨兵。通过设置对数据变化的自动响应，组织可以确保他们的 Salesforce 记录保持准确和最新，并遵循特定的业务逻辑，从而提高整体数据完整性和运营效率。

在本章中，你将掌握 Salesforce 触发器的理解以及如何使用 ChatGPT 提高你的开发流程。本章将传授设计精确和高效的触发器、优化和维护现有触发器、有效处理错误和调试的知识。它强调将需求转化为可操作的触发器配置，并侧重于在结合最佳实践的同时，采取有条理的触发器管理方法。到本章结束时，你将不仅能够创建和改进触发器，还能确保它们在动态的 Salesforce 环境中的治理。

# 探索需要 Salesforce 触发器的业务流程

Salesforce 触发器为自动化和增强各种业务流程提供了一个有吸引力的解决方案。现代商业景观的复杂性往往需要比 Salesforce 标准功能更多的功能。以下是需要定制触发器以满足特定操作需求的一些业务场景示例：

+   可以将`Insert`触发器设计为监控何时添加新潜在客户。一旦检测到，触发器会自动根据预定义的区域标准将潜在客户分配给销售代表，确保潜在客户被引导到最相关的人员，而无需人工干预。

+   可以在这里实现`Update`触发器。当交易记录被记录并超过指定的货币限额时，此触发器将同时将客户的状况更新为“VIP”，并向账户经理发送自动通知，从而确保及时确认，并可能为顶级客户提供增强的服务。

+   `Before Insert`或`Before Update`触发器可能很有价值。每当添加新联系人或修改现有联系人时，触发器会在数据提交到数据库之前将电话号码字段重新格式化为公司选择的规范格式。

+   产品记录上的`Update`触发器可以确保当产品价格发生变化时，所有相关的订单记录都会自动更新为新定价，从而防止潜在的收益差异。

从本质上讲，Salesforce 触发器在适应各种组织的独特、动态和有时复杂的业务流程中发挥着不可或缺的作用。通过提供针对数据事件的定制自动化响应，触发器增强了运营效率、准确性和响应能力。

# Salesforce 触发器处理程序——编排效能

在 Salesforce 架构中，虽然触发器作为数据驱动自动化的先锋，但触发器处理程序则编排这种自动化。深入探讨，我们可以更清晰地看到它们在简化 Salesforce 实施中的重要性：

1.  **定义触发器处理程序**：在基础层面，触发器处理程序是一个类——一个独立的代码单元——其中包含触发器将要执行的逻辑。而不是将所有逻辑直接放在触发器中，触发器会调用这些类中的方法来执行所需操作。通过将逻辑外部化到处理程序中，代码变得更加有组织、模块化和易于维护。每个函数或操作都位于其各自的隔间中，触发器本身保持精简，主要作为这些隔间的调度员。

1.  **使用触发器处理程序的理由**：采用触发器处理程序强调了几个令人信服的理由的最佳实践：

    +   **可维护性**：随着业务流程的发展，Salesforce 中的底层逻辑可能需要修改。使用触发器处理程序，可以直接在处理程序类中的特定方法内进行修改，而不是在密集的、单体化的触发器代码中筛选。

    +   **可测试性**：Salesforce 要求在部署之前，包括触发器在内的自定义代码必须有一定比例的单元测试覆盖率。通过在处理类中隔离逻辑，更容易为特定功能编写和管理单元测试。

    +   **可扩展性**：随着企业的成长，其运营的复杂性也随之增加。触发器处理程序允许实现可扩展的架构，其中新功能可以无缝地作为处理程序类中的独立方法添加，而不会干扰现有操作。

    +   **执行顺序**：Salesforce 触发器可以在单个事务中多次触发，尤其是在涉及级联操作的场景中。触发器处理程序可以帮助管理和控制执行顺序，确保操作可预测且高效。

1.  在其处理程序类中的 `updateMethod()` 方法。此方法包含执行逻辑，无论是数据验证、记录修改还是任何其他任务。执行后，控制权返回到触发器，完成其周期并结束操作。

在 Salesforce 开发的宏伟画卷中，触发器处理程序在确保自动化景观不仅功能性强而且优雅高效方面发挥着关键作用。通过隔离逻辑和促进模块化，它们倡导可持续和可扩展的软件开发原则。

## 利用 ChatGPT 进行 Salesforce 触发器开发

在 CRM 系统快速发展的领域，利用 ChatGPT 等复杂的人工智能驱动工具开辟了创新途径。尽管 Salesforce 触发器至关重要，但有时可能很复杂，需要针对特定业务流程定制逻辑。将 ChatGPT 等平台集成到这个开发周期中提供了有趣的潜力：

+   **快速原型设计**：在 Salesforce 触发器创建过程中，ChatGPT 最引人注目的用途之一是快速原型设计。开发者可以用自然语言向 ChatGPT 描述特定的业务逻辑或用例场景，ChatGPT 可以帮助草拟触发器的基本版本或提供伪代码概要。这个原型作为基础，通过为开发者提供蓝图来精炼和详细阐述，从而加速开发过程。

+   **知识增强**：ChatGPT 拥有庞大的知识库，可以帮助澄清与 Salesforce 触发器开发相关的疑问或建议最佳实践。无论是关于触发器执行顺序、限制器限制还是特定上下文变量的问题，ChatGPT 都可以充当按需顾问，减少对外部研究的广泛需求，并确保遵守 Salesforce 的最佳实践。

+   **代码审查辅助**：虽然 ChatGPT 不能替代彻底的人工代码审查，但它仍然可以在初步审查中提供帮助。开发者可以将他们的触发器代码片段与 ChatGPT 分享，以寻求见解和潜在的优化，或识别任何明显的红旗。通过促进这些初步检查，ChatGPT 可以提高整体代码质量，确保触发器高效、有效且无常见陷阱。

+   **文档和解释**：通常，封装在触发器中的逻辑需要被那些可能没有深厚技术背景的利益相关者理解。ChatGPT 可以被用来将复杂的触发器功能转化为更易于理解的解释或摘要。这些由 AI 驱动的描述有助于编写文档和用户手册，或者简单地在确保更广泛的团队理解所实施的自动化。

+   **集成 ChatGPT 的优势**：通过在 Salesforce 触发器开发过程中使用 ChatGPT，组织可以受益于更快的开发周期、持续遵守最佳实践以及知识共享的增强。此外，AI 能够根据自然语言交互理解并生成代码的能力降低了入门门槛，甚至那些编码经验有限的人也能参与到触发器开发的初步阶段。

随着 ChatGPT 继续在软件开发领域取得进展，它集成到 CRM 平台如 Salesforce 中是一个合乎逻辑的进步。利用 ChatGPT 进行触发器开发将人工智能的力量与 Salesforce 的灵活性相结合，引领 CRM 自动化高效、信息丰富和包容性的新时代。

## Salesforce 触发器最佳实践

如果设计得当，Salesforce 触发器可以显著简化并自动化 CRM 中的业务流程。然而，如果处理不当，它们可能导致不可预见的行为、性能问题和限制器限制违规。遵守最佳实践对于确保触发器的可靠、高效和可扩展至关重要。以下是对这些最佳实践的全面概述：

+   **每个对象一个触发器**：不要为单个 Salesforce 对象设置多个触发器，而是将它们合并为一个触发器。这避免了执行顺序的问题，并使得代码的管理和维护更加容易。触发器可以根据操作（插入、更新、删除等）将特定任务委派给辅助类或方法。

+   **避免硬编码 ID**：硬编码记录 ID（或任何其他数据）是一种危险的做法。这些值可能在不同的环境（沙盒、生产等）之间发生变化，导致部署问题和运行时错误。始终以编程方式检索动态值或使用自定义设置或自定义元数据来配置值。

+   **批量化你的代码**：Salesforce 在批量处理模型上运行。始终确保你的触发器可以一次处理多个记录。避免在循环中放置 SOQL 查询或 DML 操作，因为这可能导致你迅速达到限制器限制。

+   `Trigger.new` 和 `Trigger.old`，以便开发人员可以访问触发器所引发的记录。理解和有效使用这些变量以适应不同的场景（例如区分插入和更新操作）。

+   **选择触发器处理器**：如前所述，而不是将所有逻辑直接放在触发器中，将繁重的工作委托给触发器处理器类。这种方法使触发器保持简洁，并促进模块化和可重用性。

+   **实现适当的错误处理**：确保您的触发器代码能够优雅地处理异常，并向最终用户提供有意义的错误信息。这有助于故障排除，并确保用户不会收到晦涩或通用的系统错误信息。

+   **限制 SOQL 和 DML 操作**：注意限制器限制。始终优化 SOQL 查询和 DML 操作的数量。通过减少这些操作，您确保触发器高效运行，不会耗尽系统资源。

+   **彻底测试**：每个触发器都应该有相关的测试类，涵盖各种场景。Salesforce 要求至少 75% 的代码覆盖率，但目标更高。除了覆盖率之外，确保您的测试验证触发器的实际功能以及边缘情况。

+   **避免递归触发器**：递归触发器，即触发器意外地触发自身，可能导致意外的行为并迅速达到限制器限制。实现检查或静态变量以防止不想要的递归调用。

+   **执行顺序意识**：理解 Salesforce 的执行顺序（自动化任务如验证规则、工作流和触发器的处理顺序）。这有助于确保您的触发器按预期行为，尤其是在存在多个自动化任务时。

+   **注释和文档**：始终记录触发器（及其相关辅助类）的目的和功能。这有助于其他开发者理解代码背后的逻辑和意图。代码中的定期注释也有助于阐明复杂部分和决策点。

+   **保持更新并定期审查**：Salesforce 每年都会进行多次更新。定期审查您的触发器，以确保它们针对最新的平台功能进行了优化，并且没有使用已弃用的功能。

总结来说，在 Salesforce 中创建触发器并不仅仅是关于脚本自动化。它关乎构建一个可靠、高效且可维护的 Salesforce 生态系统的一部分。通过遵循上述最佳实践，开发者可以确保他们的触发器不仅满足当前的业务需求，而且能够经受住时间的考验、可扩展性和平台演化的考验。

## ChatGPT 在其中扮演什么角色？

在 Salesforce 触发器的开发中利用 ChatGPT 可以极大地加快进程，同时保持或甚至提高代码质量。以下是 ChatGPT 可以应用的方式：

1.  **为了自动化代码生成过程**：通过其对 Salesforce 的 Apex 编程语言的深入理解，ChatGPT 可以用来自动生成触发器的代码片段。开发者可以提供具体要求，例如触发对象、触发事件（例如，在插入之前、在更新之后等）以及期望的操作。然后 ChatGPT 可以将这些要求转换成结构良好且语法正确的 Apex 触发器代码。这不仅加快了开发过程，还降低了人为错误的可能性。

1.  **为了增强协作和理解**：ChatGPT 的自然语言处理能力在技术和非技术利益相关者之间架起了一座桥梁。将复杂的需求转化为可执行的代码，并以人类可读的方式解释代码，促进了团队成员之间更好的理解和协作。这在处理需要业务逻辑和规则与组织需求精确对齐的复杂触发器时尤其有益。

1.  **为了测试和验证辅助**：ChatGPT 可以帮助创建触发器的测试类，这是 Salesforce 开发中确保代码覆盖率和功能性的重要部分。通过理解触发器的逻辑，它可以生成相关的测试场景和相应的 Apex 测试代码。这有助于开发者快速验证触发器，确保它们满足定义的标准并按预期工作。

1.  **为了持续学习和改进**：ChatGPT 处理大量信息并持续学习的能力，使其能够紧跟最佳实践和 Salesforce 更新。这意味着生成的代码不仅准确，而且遵循最新的标准和推荐。它甚至可以提供优化现有触发器、利用最佳模式和避免常见陷阱的见解和建议。

1.  **为了促进合规和安全**：在 Salesforce 开发中，安全和合规至关重要。ChatGPT 可以根据组织的特定合规要求进行定制，并将其整合到开发过程中。无论是执行字段级安全还是确保触发器高效处理批量操作，ChatGPT 都可以指导开发者将代码与最佳实践对齐，从而确保稳健、安全且合规的解决方案。

ChatGPT 是 Salesforce 开发者工具库中的宝贵工具，尤其是在处理触发器时。通过自动化代码生成、增强协作、协助测试、确保遵循最佳实践和合规标准，它为更快、更高品质的开发提供了途径。它与 Salesforce 开发的迭代和响应性特性相契合，促进了构建强大和可扩展触发器的更高效和有效的方法。

# 在 ChatGPT 的指导下设计触发器

让我们首先定义一个应该由触发器解决的问题。每当账户地址发生变化时，需要将更新后的地址发送到 ERP 系统。假设已经编写了调用出方法，触发器只需调用调用出方法即可。这将是一个**后**触发器，因为账户记录中插入/更新的内容并未发生变化。作为一个良好的设计实践，我们将使用处理程序，这就是我们请求的内容。对于这个例子，目前还没有任何账户触发器：

| **提示** | 请编写一个 Salesforce 后触发器，在账户记录中的任何地址字段发生变化时调用一个方法。请使用触发器处理程序编写此代码。请编写插入和更新。 |
| --- | --- |

| **响应** | 当然可以！以下是如何为账户对象结构化一个 Salesforce 后触发器，以检测地址字段的变化，并在这些字段更改时调用一个方法。这将包括一个触发器和触发器处理程序类：

+   `AccountTrigger.trigger`:

    ```py
    trigger AccountTrigger on Account (after insert, after update) {
    ```

|

|  |
| --- |

```py
    if (Trigger.isAfter) {
```

```py
        if (Trigger.isInsert &#124;&#124; Trigger.isUpdate) {
```

```py
            AccountTriggerHandler.handleAddressChange(Trigger.new, Trigger.oldMap);
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

+   `AccountTriggerHandler.cls`:

    ```py
    public class AccountTriggerHandler {
        public static void handleAddressChange(List<Account> newAccounts, Map<Id, Account> oldMap) {
            List<Account> accountsWithChangedAddresses = new List<Account>();
            for (Account acc : newAccounts) {
                if (oldMap == null &#124;&#124; addressChanged(acc, oldMap.get(acc.Id))) {
                    accountsWithChangedAddresses.add(acc);
                }
            }
            if (!accountsWithChangedAddresses.isEmpty()) {
                callYourMethod(accountsWithChanged Addresses);
            }
        }
        private static Boolean addressChanged(Account newAccount, Account oldAccount) {
            return (newAccount.BillingStreet != oldAccount.BillingStreet &#124;&#124;
                    newAccount.BillingCity != oldAccount.BillingCity &#124;&#124;
                    newAccount.BillingState != oldAccount.BillingState &#124;&#124;
                    newAccount.BillingPostalCode != oldAccount.BillingPostalCode &#124;&#124;
                    newAccount.BillingCountry != oldAccount.BillingCountry &#124;&#124;
                    newAccount.ShippingStreet != oldAccount.ShippingStreet &#124;&#124;
    ```

|

|  |
| --- |

```py
                newAccount.ShippingCity != oldAccount.ShippingCity &#124;&#124;
```

```py
                newAccount.ShippingState != oldAccount.ShippingState &#124;&#124;
```

```py
                newAccount.ShippingPostalCode != oldAccount.ShippingPostalCode &#124;&#124;
```

```py
                newAccount.ShippingCountry != oldAccount.ShippingCountry);
```

```py
    }
```

```py
    private static void callYourMethod(List<Account> accounts) {
```

```py
        // Your logic here. This is the method that gets called when any address field changes.
```

```py
    }
```

```py
}
```

这种设置将逻辑从触发器中分离出来，使触发器保持简单，并将大部分逻辑推送到处理程序类。它检查账单和发货地址的变化。如果检测到变化，将调用`callYourMethod`函数。您可以根据需要在此函数中实现您的逻辑。**注意**：确保您已编写测试类，以维护 Salesforce 的代码覆盖率要求。 |

分析 ChatGPT 的响应后，很明显触发器的底层结构构建得相当专业。触发器考虑了记录修改，并纳入了对标准字段的考虑。它甚至包括了一个创建相应测试类的提醒。然而，代码的质量并未达到实际项目部署所需的标准。最明显的改进领域是缺乏内联注释和缺少注释头块。现在，让我们请求 ChatGPT 整合这些注释以及测试类：

| **提示** | 请您使用内联注释和注释头块重写代码，并包括一个测试类？请不要修改代码。 |
| --- | --- |

| **响应** | 当然！我将提供一个改进的 Salesforce 触发器代码及其处理程序，包括详细的注释头块、内联注释和测试类的轮廓：

+   `AccountTrigger.trigger`:

    ```py
    /**
     * AccountTrigger
     *
     * This trigger detects address changes on the Account record.
    ```

|

|  |
| --- |

```py
 * It is designed to operate after insert and after update operations.
```

```py
 *
```

```py
 * Author: [Your Name]
```

```py
 * Date: [Creation Date]
```

```py
 */
```

```py
trigger AccountTrigger on Account (after insert, after update) {
```

```py
    // After Insert/Update operations: Check for address changes
```

```py
    if (Trigger.isAfter && (Trigger.isInsert &#124;&#124; Trigger.isUpdate)) {
```

```py
        AccountTriggerHandler.handleAddressChange(Trigger.new, Trigger.oldMap);
```

```py
    }
```

```py
}
```

+   `AccountTriggerHandler.cls`:

    ```py
    /**
     * AccountTriggerHandler
     *
     * A handler class for the AccountTrigger. It houses the logic to detect
     * address changes on Account records and calls appropriate methods if changes are found.
     *
     * Author: [Your Name]
     * Date: [Creation Date]
     */
    public class AccountTriggerHandler {
        /**
         * Processes the address change on Account records.
         *
         * @param newAccounts - New versions of the Account records
         * @param oldMap - Old versions of the Account records for comparison
         */
        public static void handleAddressChange(List<Account> newAccounts, Map<Id, Account> oldMap) {
            List<Account> accountsWithChangedAddresses = new List<Account>();
    ```

|

|  |
| --- |

```py
        // Iterate through the new Account records
```

```py
        for (Account acc : newAccounts) {
```

```py
            // If the record is a new insertion or its address has changed, add to the list
```

```py
            if (oldMap == null &#124;&#124; addressChanged(acc, oldMap.get(acc.Id))) {
```

```py
                accountsWithChangedAddresses.add(acc);
```

```py
            }
```

```py
        }
```

```py
        // If address changes are detected, call the designated method
```

```py
        if (!accountsWithChangedAddresses.isEmpty()) {
```

```py
            processChangedAddresses(accountsWith ChangedAddresses);
```

```py
        }
```

```py
    }
```

```py
    /**
```

```py
     * Compares address fields between old and new Account records.
```

```py
     *
```

```py
     * @param newAccount - The updated Account record
```

```py
     * @param oldAccount - The original Account record
```

```py
     * @return - Returns true if there are changes, otherwise false
```

```py
     */
```

```py
    private static Boolean addressChanged(Account newAccount, Account oldAccount) {
```

```py
        return (newAccount.BillingStreet != oldAccount.BillingStreet &#124;&#124;
```

```py
                ... // [Other address fields comparisons]
```

```py
                newAccount.ShippingCountry != oldAccount.ShippingCountry);
```

```py
    }
```

```py
    /**
```

```py
     * Method to process the Account records with changed addresses.
```

```py
     * Placeholder for the desired operations on changed address records.
```

```py
     *
```

|

|  |
| --- |

```py
     * @param accounts - List of Account records with address changes
```

```py
     */
```

```py
    private static void processChangedAddresses(List<Account> accounts) {
```

```py
        // Logic to process the address changes
```

```py
    }
```

```py
}
```

+   `AccountTriggerTest.cls`:

    ```py
    /**
     * AccountTriggerTest
     *
     * Test class for the AccountTrigger and its handler.
     * Ensures that address change detection and processing are functioning as expected.
     *
     * Author: [Your Name]
     * Date: [Creation Date]
     */
    @isTest
    private class AccountTriggerTest {
        @isTest static void testAddressChangeDetection() {
            // Setup: Create an Account record
            Account testAcc = new Account(Name='Test Account', BillingStreet='Old Street');
            insert testAcc;
            // Modify the BillingStreet field
            testAcc.BillingStreet = 'New Street';
            update testAcc;
            // [Assertions to validate that the address change processing logic was executed]
        }
    }
    ```

请记住用适当的值填写`[您的姓名]`和`[创建日期]`占位符。提供的测试类是一个基本框架，您将需要根据对地址更改的具体处理进行扩展。 |

这是一种改进。通过包含注释、测试类和调整某些注释部分的提醒，代码现在已准备好用作新 Salesforce 项目中代码的模板。

在本节中，我们探讨了如何利用 ChatGPT 生成 Salesforce 代码模板，例如 Apex 触发器。但我们接下来该怎么做呢？

# 优化和重构现有触发器

虽然 ChatGPT 是一个强大的工具，但它不能取代正式的代码审查流程、专门的 linting 工具或 Salesforce 内置的开发工具。相反，它作为补充资源，增强您的开发工作流程并提高代码质量。ChatGPT 可以通过几种方式成为 Salesforce 开发人员的宝贵工具，以确保并提高其代码的质量。在本节中，我们将回顾以下两种方式：

+   **最佳实践**：

    +   **指导**：如果您不确定某个方法是否符合 Salesforce 的最佳实践，ChatGPT 可以提供见解和建议

    +   **模式和反模式**：您可以查询针对 Salesforce 开发的常见设计模式和反模式

+   **性能优化**：

    +   **查询优化**：ChatGPT 可以提供有关如何优化 SOQL 查询以避免 governor 限制和提高性能的建议

    +   **批量化**：确保您的代码已批量化以高效处理多个记录。如果您不确定如何批量化某些逻辑，ChatGPT 可以提供帮助

让我们考虑以下示例。假设我们想要编写一个触发器，在`Email`字段更改时更新`Contact`对象上的`Custom_Status__c`自定义字段。

这是未批量化的触发器：

```py
trigger ContactTrigger on Contact (before update) {
    // Loop through each Contact in the trigger context
    for (Contact newContact : Trigger.new) {
        // Retrieve the old version of the Contact for comparison
        Contact oldContact = Trigger.oldMap.get(newContact.Id);
        // Check if the Email field has changed
        if (oldContact.Email != newContact.Email) {
            // Query the Contact to get additional fields (non-bulkified)
            Contact queriedContact = [SELECT Id, Custom_Status__c FROM Contact WHERE Id = :newContact.Id LIMIT 1];
            // Update the custom field
            queriedContact.Custom_Status__c = 'Email Updated';
            // Update the Contact (non-bulkified)
            update queriedContact;
        }
    }
}
```

这段代码不安全，因为它有以下原因：

1.  它在循环内查询，这意味着如果触发器同时处理多个记录，您将迅速达到 SOQL 查询限制

1.  它在循环内执行 DML 操作，如果您正在处理多个记录，这可能会迅速达到 DML 限制

现在，让我们让 ChatGPT 批量化代码：

| **提示** | 假设我们想要编写一个触发器，在`Email`字段更改时更新`Contact`对象上的`Custom_Status__c`自定义字段。您能否批量化以下触发器？

```py
trigger ContactTrigger on Contact (before update) {
```

```py
    // Loop through each Contact in the trigger context
```

```py
    for (Contact newContact : Trigger.new) {
```

```py
        // Retrieve the old version of the Contact for comparison
```

```py
        Contact oldContact = Trigger.oldMap.get(newContact.Id);
```

```py
        // Check if the Email field has changed
```

```py
        if (oldContact.Email != newContact.Email) {
```

```py
            // Query the Contact to get additional fields (non-bulkified)
```

```py
            Contact queriedContact = [SELECT Id, Custom_Status__c FROM Contact WHERE Id = :newContact.Id LIMIT 1];
```

```py
            // Update the custom field
```

```py
            queriedContact.Custom_Status__c = 'Email Updated';
```

|

| |
| --- |

```py
            // Update the Contact (non-bulkified)
```

```py
            update queriedContact;
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

|

| **响应** | 当然！让我们将给定的触发器批量化。为了批量化触发器，我们将执行以下操作：

1.  收集电子邮件发生变化的联系人的 ID。

1.  在循环外执行单个查询以获取所有这些联系人的所需字段。

1.  修改检索到的联系人。

1.  在单个批量更新操作中更新修改后的联系人。

这是触发器的批量化版本：

```py
trigger ContactTrigger on Contact (before update) {
```

```py
    // A list to store Contacts whose email has changed and need to be updated
```

```py
    List<Contact> contactsToUpdate = new List<Contact>();
```

```py
    // Loop through each Contact in the trigger context to identify ones with changed emails
```

```py
    for (Contact newContact : Trigger.new) {
```

```py
        Contact oldContact = Trigger.oldMap.get(newContact.Id);
```

```py
        // Check if the Email field has changed
```

```py
        if (oldContact.Email != newContact.Email) {
```

```py
            // Add the current contact Id to the list (we'll query in bulk outside the loop)
```

```py
            contactsToUpdate.add(new Contact(Id = newContact.Id));
```

```py
        }
```

```py
    }
```

|

|  |
| --- |

```py
    // Bulk query to get all contacts with changed emails
```

```py
    if (!contactsToUpdate.isEmpty()) {
```

```py
        List<Contact> queriedContacts = [SELECT Id, Custom_Status__c FROM Contact WHERE Id IN :contactsToUpdate];
```

```py
        // Update the custom field for all queried contacts
```

```py
        for (Contact c : queriedContacts) {
```

```py
            c.Custom_Status__c = 'Email Updated';
```

```py
        }
```

```py
        // Bulk update operation
```

```py
        update queriedContacts;
```

```py
    }
```

```py
}
```

现在，无论处理的联系人数量如何，触发器都只执行一个 SOQL 查询和一个 DML 操作。这使得它在 Salesforce 限制范围内可扩展且高效。|

ChatGPT 不仅使用标准方法批量化了触发器，还概述了教育目的的步骤。这种指导对初学 Salesforce 编码的开发者来说非常有价值。此外，根据之前的反馈，相关的评论被整合到代码中。

我们可以使用类似的方法来处理更复杂的模式——单例模式。单例模式是一种设计模式，用于限制一个类只能被实例化一次。这种模式确保一个类只有一个实例，并提供了一个全局访问点来访问这个实例。当系统需要恰好一个对象来协调操作时，它特别有用。例如，你可能有一个数据库连接池、日志服务或配置管理器，这些最好使用单例模式来实现。

在面向对象的编程语言中，单例模式通常通过创建一个带有方法的类来实现，如果不存在实例，该方法会创建一个新的实例。如果已经存在实例，它将返回对该实例的引用。为了确保在第一个线程执行代码时不会由另一个线程创建实例，该方法通常被标记为同步，或者采取等效措施。

单例模式有几种优点。首先，它提供了一个单一的控制点，这在日志记录、驱动对象、缓存、线程池或数据库连接等方面可能是有益的。它也很容易进行子类化，你可以选择实例化过程。然而，值得注意的是，在某些情况下，单例可能被视为反模式，因为它将全局状态引入应用程序，并可能使代码更难测试和推理。所以，虽然它可能非常有用，但它应该谨慎实现。

让我们考虑以下代码：

```py
trigger AccountTrigger on Account (before insert) {
    // Loop through each Account in the trigger context
    for(Account acc : Trigger.new) {
        // Update the Description field
        acc.Description = 'Updated by trigger';
    }
}
```

这个触发器没有使用单例模式。它很简单，在触发器内部处理任务。现在，让我们考虑提示和响应：

| **提示** | 你能重写以下 Apex 触发器，使其实现单例模式吗？

```py
 trigger AccountTrigger on Account (before insert) {
```

```py
    // Loop through each Account in the trigger context
```

```py
    for(Account acc : Trigger.new) {
```

```py
        // Update the Description field
```

```py
        acc.Description = 'Updated by trigger';
```

```py
    }
```

```py
}
```

|

| **响应** | 当然可以！单例模式在 Salesforce 中经常被用来确保在单个事务中，特定的类只被实例化一次。这可以用于维护一个单一的控制点，例如用于缓存或设置全局变量。以下是使用单例模式重写触发器的方法：

```py
// Singleton class to handle Account logic
```

```py
public class AccountTriggerHandler {
```

```py
    private static AccountTriggerHandler instance;
```

```py
    // Private constructor to prevent direct instantiation
```

```py
    private AccountTriggerHandler() {}
```

```py
    // Public method to get the instance
```

```py
    public static AccountTriggerHandler getInstance() {
```

```py
        if (instance == null) {
```

```py
            instance = new AccountTriggerHandler();
```

```py
        }
```

```py
        return instance;
```

```py
    }
```

|

|  |
| --- |

```py
    // Method to update Account descriptions
```

```py
    public void updateAccountDescriptions(List<Account> accounts) {
```

```py
        for(Account acc : accounts) {
```

```py
            acc.Description = 'Updated by trigger with Singleton pattern';
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
// Account trigger
```

```py
trigger AccountTrigger on Account (before insert) {
```

```py
    AccountTriggerHandler handler = AccountTriggerHandler.getInstance();
```

```py
    handler.updateAccountDescriptions(Trigger.new);
```

```py
}
```

在这个例子中，`AccountTriggerHandler` 是一个单例类，它包含更新账户描述的逻辑。触发器本身（`AccountTrigger`）使用这个单例类来执行实际操作。|

之前的提示展示了 Salesforce 开发中的几个重要概念，特别是关于 Apex 触发器和设计模式：

+   `AccountTriggerHandler`)。这使得代码更容易维护和测试。

+   `AccountTriggerHandler`) 被实现为一个单例。这确保在整个事务中只使用一个处理程序实例。这在需要维护状态或缓存数据的情况下特别有用。

+   **封装**：单例类使用私有构造函数和公共方法来控制其实例化。这封装了创建逻辑，并确保外部类不能直接实例化它，遵循良好的面向对象原则。

+   **可重用性**：通过在处理程序类中隔离逻辑，代码变得更加可重用。相同的逻辑可以从其他触发器或类中调用，而无需重复。

+   **可读性**：当业务逻辑与触发器逻辑分离时，代码的可读性会提高。这使其他开发者更容易理解代码在做什么，这在有多个开发者的较大项目中尤为重要。

当我们从重构 Salesforce 触发器的任务转向提升其性能和可维护性时，解决不可避免的调试和错误处理至关重要。重构带来了优化功能性的承诺，但即使是经过最优雅重构的触发器也不是没有错误的。下一节将深入探讨故障排除的基本技能：识别意外行为的根本原因、理解错误信息的细微差别，并解决问题以确保我们的触发器不仅高效运行，而且在所有预期条件下都能可靠执行。这一部分旅程强调了触发器设计中弹性的重要性，使我们能够自信且精确地导航调试的复杂性。

# 触发器中的调试和错误处理

当在 Salesforce 中使用 Apex 触发器进行调试和错误处理时，ChatGPT 是一个宝贵的资源。虽然 Apex 触发器对于自动化复杂业务流程至关重要，但它们也可能引入各种挑战，如未处理的异常、递归问题或限制器限制违规。ChatGPT 可以通过分析代码片段、提供结构化触发器的最佳实践以及解释如何使用防御性编码技术（如 try-catch 块）来帮助开发者识别常见陷阱。

ChatGPT 的实用性远不止于仅仅指出语法错误或运行时问题。它还根据 Salesforce 生态系统已建立的指南提供上下文解释和建议。例如，如果一个开发者正在处理批量化问题或不确定如何有效管理触发上下文变量，ChatGPT 可以提供针对性的建议以及代码示例。这使得开发者不仅能解决他们眼前的问题，还能理解有效 Apex 触发器开发背后的“为什么”和“如何”。

此外，ChatGPT 可以指导开发者创建测试类，这对于部署和确保触发器逻辑的健壮性至关重要。通过利用 ChatGPT 的功能，开发者可以更深入地理解调试和错误处理技术，从而创建出更健壮、高效和易于维护的 Apex 触发器。

作为实际例子，在 Salesforce 中，你无法在触发器内直接对 `Trigger.new` 或 `Trigger.old` 集合执行 DML 操作。尝试这样做会导致运行时错误。以下是一个说明问题的示例：

```py
trigger AccountTrigger on Account (before update) {
    delete Trigger.old;
}
```

在这个例子中，触发器试图直接从 `Trigger.old` 列表中删除记录，这是不允许的，并且会导致运行时错误。

我们可以编写一个提示来请求 ChatGPT 帮助代码。这里假设我是一个初级开发者，并且我不熟悉 Apex：

| **提示** | 这个触发器有什么问题？

```py
trigger AccountTrigger on Account (before update) {
```

```py
    delete Trigger.old;
```

```py
}
```

|

| **响应** | `delete Trigger.old;` 正在尝试直接在 `Trigger.old` 列表上执行删除 DML 操作。在 Salesforce Apex 触发器中这是不允许的。在 Salesforce 中，`Trigger.old` 和 `Trigger.new` 是只读的 sObjects 列表，并且你不能直接对它们执行 DML 操作。具体来说，考虑以下情况：1. `Trigger.old` 和 `Trigger.new` 在“before”和“after”触发器中都是只读的。2. 你不能直接在这些集合上执行“insert”、“update”或“delete”等 DML 操作。如果你想在“before update”触发器操作期间根据某些条件删除记录，你应该将你想要删除的记录的 ID 或实例放入一个单独的列表中，然后在该列表上执行删除操作。以下是如何重写触发器以删除账户的方法：

```py
trigger AccountTrigger on Account (before update) {
```

```py
    List<Account> accountsToDelete = new List<Account>();
```

```py
    for(Account acc : Trigger.old) {
```

```py
        // Apply your condition here, if any
```

```py
        accountsToDelete.add(new Account(Id = acc.Id));
```

```py
    }
```

```py
    if(!accountsToDelete.isEmpty()) {
```

```py
        delete accountsToDelete;
```

```py
    }
```

```py
}
```

在这个重写的版本中，创建了一个名为 `accountsToDelete` 的单独列表，然后用于删除操作。这样，你就不直接在 `Trigger.old` 上执行 DML 操作，并且这将符合 Salesforce 的规则。|

ChatGPT 已经识别了问题，提出了更好的方法，提供了代码，并介绍了 Salesforce 概念。真是个大帮助！

# 摘要

在本章中，你扩展了使用 Salesforce 触发器的熟练度，并在 ChatGPT 的支持下，对其设计、实施和优化有了深入了解。本章提供了对触发器的基础理解，并展示了 ChatGPT 的知识如何指导开发更高效和健壮的触发器。你学习了如何设计脚本触发器，使其符合业务需求和最佳实践，以及通过重构现有触发器来提高性能。此外，你还获得了故障排除的技能，学会了快速解决与触发器相关的问题。最后，本章培养了你对治理和最佳实践的欣赏，确保你能够维护一个井然有序且性能卓越的 Salesforce 环境。

在下一章中，你将跟随 ChatGPT 的能力，踏上一段穿越**闪电 Web 组件**（**LWC**）王国的旅程。你将深入探索 LWC 在 Salesforce UI 中的变革力量，了解如何利用 ChatGPT 的洞察力构建动态且复杂的组件。本章将涵盖构建、样式化和与 LWC 交互的基本要素，以及故障排除和调试的关键技术。随着你的进步，你将熟练掌握 LWC 开发，从提高创建响应式 UI 元素的专业技能到精通用户体验设计的微妙之处。你还将获得快速识别和解决 LWC 问题的宝贵技能，确保你的组件达到最高质量和性能标准。在整个下一章中，重点将放在持续学习和适应上，为你准备在 Salesforce UI 开发这个快速发展的世界中保持领先。
