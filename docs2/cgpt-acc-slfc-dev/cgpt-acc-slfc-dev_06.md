# 6

# 使用 ChatGPT 进行 Salesforce Apex 开发

Apex 是 Salesforce 提供的一种专有编程语言，旨在允许开发者在 Salesforce 服务器上执行流程和事务控制语句。它使开发者能够将业务逻辑添加到系统事件中，例如按钮点击或更新，并通过 Apex 类、批处理作业和可调度类创建自定义逻辑。Apex 与 Salesforce 的数据模型紧密集成，并针对构建与 Salesforce 应用程序交互的复杂业务流程进行了优化。

ChatGPT 在这个领域中提供了显著的价值，尤其是在 Apex 开发方面，通过协助一系列活动，从代码生成到调试，甚至架构设计考虑。例如，ChatGPT 可以指导开发者遵循编写 Apex 类的最佳实践，包括错误处理、限制器限制和测试类创建。它还可以分析现有代码的可读性、复杂性和性能问题，提供重构建议以实现更好的可维护性和效率。ChatGPT 甚至可以帮助设计可扩展和可重用的 Apex 类模式，确保您不仅解决了当前的问题，而且构建了一个持久的解决方案。

通过利用 ChatGPT 的功能，开发者可以加速他们的开发周期，确保他们遵循最佳实践，同时也能在问题成为大问题之前捕捉到潜在的问题。这不仅使开发过程更加高效，还有助于生产出更高质量、更可靠的 Apex 代码。这种独特的功能组合使 ChatGPT 成为任何希望掌握 Salesforce 中 Apex 开发的人的宝贵工具，无论是新手试图学习基础知识，还是经验丰富的开发者寻求优化解决方案。

在 Salesforce 中编写高质量的 Apex 代码会带来一系列独特的挑战，开发人员需要解决这些问题以实现健壮和可维护的解决方案。其中最常见的问题之一是处理限制器限制。Salesforce 强制实施各种类型的限制器限制，如 SOQL 查询限制、DML 语句限制和 CPU 时间限制，以确保在多租户环境中没有单个租户垄断共享资源。这里的挑战是编写高效且批量安全的代码，能够在这些限制内完成所需任务，这通常需要仔细规划、代码批量化以及使用批处理和可排队 Apex 等模式。

另一个主要挑战是确保 Apex 代码中的稳健安全措施。Apex 以系统模式运行，这意味着它不天生尊重分配给运行用户的对象和字段级权限。开发者必须明确检查 CRUD、FLS 和共享规则，尤其是在代码旨在以用户上下文运行时。确保代码遵守 Salesforce 的安全模型，同时保持功能性和效率，需要对 Apex 和平台的安全机制有细微的理解。

最后，测试和调试可能特别具有挑战性。Apex 需要至少 75%的代码覆盖率才能部署，但高代码覆盖率并不意味着代码经过充分测试。开发者经常面临编写有意义的测试用例的挑战，这些测试用例需要涵盖各种场景和边缘情况，包括批量操作、部分失败等。Apex 的调试工具正在改进，但仍然需要在提供其他语言中可用的实时、交互式调试功能方面走得更远。这些因素使得测试和调试过程比其他情况下更加繁琐和耗时。

# 在 ChatGPT 的指导下编写 Apex 代码

如前所述，编写 Apex 代码可能是一项复杂且微妙的工作，这不仅需要深入理解 Salesforce 的功能，还需要敏锐地关注最佳实践和高效的问题解决方法。无论你是创建触发器、批量类还是通用实用类，你的 Apex 代码质量直接影响到 Salesforce 解决方案的健壮性和可维护性。这正是 ChatGPT 可以提供无与伦比指导的地方。

由于 Salesforce 的拖放功能持续改进，以下用户故事可能无需 Apex 代码即可解决。然而，作为一个实际示例，我们可以考虑以下用户故事：

用户故事 – *自动化账户* *审查通知*

+   **目标**：作为一名销售经理，我希望自动向过去 30 天内未更新的账户的所有者发送电子邮件通知

+   **验收标准**：

    +   **给定**一个账户在过去 30 天内未更新

    +   **当**夜间批量作业运行时

    +   **然后**账户所有者应收到一封电子邮件通知，提醒他们审查和更新账户详情

这个用户故事将需要一个 Apex 批量类来查询所有账户并检查它们的最后修改日期，然后向符合标准的账户发送电子邮件。你可能还希望有一个自定义对象或字段来记录发送的通知：

| **提示** | 请您为以下用户故事创建 Apex 代码。请指定配置并确保代码实现错误处理框架：用户故事 – 自动化账户审查通知

+   **目标**：作为销售经理，我希望自动向过去 30 天内未更新的账户所有者发送电子邮件通知

+   **验收标准**：

    +   **给定** 一个账户在过去 30 天内未更新

    +   **当**夜间批量作业运行时

    +   **那么**账户所有者应该收到一封电子邮件通知，提醒他们审查和更新账户详情

|

| **回复** | 当然可以！对于给定的用户故事，您需要一个批量 Apex 类来识别在过去 30 天内未更新的账户，然后向账户所有者发送电子邮件。此外，您还需要调度这个批量 Apex 以每晚运行。让我们深入了解代码和配置细节：**Apex** **批量类**：

```py
public class AccountReviewNotificationBatch implements Database.Batchable<sObject>, Database.Stateful {
```

```py
    public Database.QueryLocator start(Database.BatchableContext bc) {
```

```py
        String query = 'SELECT Id, Name, Owner.Email FROM Account WHERE LastModifiedDate <= LAST_N_DAYS:30';
```

```py
        return Database.getQueryLocator(query);
```

```py
    }
```

```py
    public void execute(Database.BatchableContext bc, List<sObject> scope) {
```

```py
        List<Messaging.SingleEmailMessage> emails = new List<Messaging.SingleEmailMessage>();
```

```py
        for (sObject record : scope) {
```

```py
            Account acc = (Account)record;
```

```py
            Messaging.SingleEmailMessage email = new Messaging.SingleEmailMessage();
```

|

|  |
| --- |

```py
            email.setToAddresses(new String[] {acc.Owner.Email});
```

```py
            email.setSubject('Reminder: Account Review Needed for ' + acc.Name);
```

```py
            email.setPlainTextBody('Please review and update the account details for ' + acc.Name);
```

```py
            emails.add(email);
```

```py
        }
```

```py
        if (!emails.isEmpty()) {
```

```py
            try {
```

```py
                Messaging.sendEmail(emails);
```

```py
            } catch (Exception e) {
```

```py
                System.debug('Error sending emails: ' + e.getMessage());
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
    public void finish(Database.BatchableContext bc) {
```

```py
        // You can add any post-processing code here if required.
```

```py
    }
```

```py
}
```

**调度批量类**：您还需要调度这个批量类以每晚运行。您可以通过编程方式或通过 Salesforce 的 UI 来完成此操作。**编程调度**：System.schedule(‘Nightly Account Review Notification’, ‘0 0 0 * * ?’, new ScheduledClass());错误处理框架：前面的批量类包括一个简单的错误处理机制，它会在发送电子邮件的过程中捕获任何异常并将它们写入调试日志。对于更复杂的情况，您可以实现一个自定义的日志系统，将这些错误写入自定义对象或向管理员发送通知。

|  | **配置**：将 AccountReviewNotificationBatch 类部署到您的 Salesforce 环境中调度批量类每晚运行，无论是通过编程方式还是通过 Salesforce UI。按照这些步骤，您应该能够满足用户故事中指定的验收标准。 |
| --- | --- |

很好的回复。这次，我们要求额外的技术要求，例如错误处理框架——它已经非常轻量级地实现了——以及要遵循的配置步骤列表。再次强调，这非常适合用作您的 Apex 开发的蓝图。

让我们更进一步：

| **提示** | 请您使用 ExceptionForce 错误处理框架重写之前的代码？ |
| --- | --- |

| **回复** | 当然可以！使用如 ExceptionForce 这样的错误处理框架可以帮助您以更结构化的方式管理异常，使调试和监控代码变得更加容易。假设 ExceptionForce 有如 logException 这样的方法来记录错误，更新的批量 Apex 类将看起来像这样：

```py
public class AccountReviewNotificationBatch implements Database.Batchable<sObject>, Database.Stateful {
```

```py
    public Database.QueryLocator start(Database.BatchableContext bc) {
```

```py
        String query = 'SELECT Id, Name, Owner.Email FROM Account WHERE LastModifiedDate <= LAST_N_DAYS:30';
```

```py
        return Database.getQueryLocator(query);
```

```py
    }
```

```py
    public void execute(Database.BatchableContext bc, List<sObject> scope) {
```

```py
        List<Messaging.SingleEmailMessage> emails = new List<Messaging.SingleEmailMessage>();
```

```py
        for (sObject record : scope) {
```

```py
            Account acc = (Account)record;
```

```py
            Messaging.SingleEmailMessage email = new Messaging.SingleEmailMessage();
```

|

|  |
| --- |

```py
            email.setToAddresses(new String[] {acc.Owner.Email});
```

```py
            email.setSubject('Reminder: Account Review Needed for ' + acc.Name);
```

```py
            email.setPlainTextBody('Please review and update the account details for ' + acc.Name);            emails.add(email);
```

```py
        }
```

```py
        if (!emails.isEmpty()) {
```

```py
            try {
```

```py
                Messaging.sendEmail(emails);
```

```py
            } catch (Exception e) {
```

```py
                ExceptionForce.logException(e, 'Failure in sending email notifications in AccountReviewNotificationBatch');
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
    public void finish(Database.BatchableContext bc) {
```

```py
        // You can add any post-processing code here if required.
```

```py
    }
```

```py
}
```

这段代码使用 ExceptionForce.logException 记录在发送电子邮件过程中发生的任何异常。这个方法将记录异常详情，并可能根据 ExceptionForce 的配置向管理员发送警报。您仍然需要按照前一个示例中的说明调度批量类，并确保 ExceptionForce 框架在您的 Salesforce 环境中可用。

这次表现并不太好。它提供了一个更复杂的错误处理框架，但没有将其放在一个知名框架的上下文中。话虽如此，这使我们熟悉这些概念节省了大量时间。

现在，让我们请求测试类并识别触碰到限制器的风险：

| **提示** | 您能否识别以下 Apex 代码触碰到限制器的风险，并创建测试类？

```py
Database.Batchable<sObject>, Database.Stateful {
```

```py
    public Database.QueryLocator start(Database.BatchableContext bc) {
```

```py
        String query = 'SELECT Id, Name, Owner.Email FROM Account WHERE LastModifiedDate <= LAST_N_DAYS:30';
```

```py
        return Database.getQueryLocator(query);
```

```py
    }
```

```py
    public void execute(Database.BatchableContext bc, List<sObject> scope) {
```

```py
        List<Messaging.SingleEmailMessage> emails = new List<Messaging.SingleEmailMessage>();
```

```py
        for (sObject record : scope) {
```

```py
            Account acc = (Account)record;
```

```py
            Messaging.SingleEmailMessage email = new Messaging.SingleEmailMessage();
```

```py
            email.setToAddresses(new String[] {acc.Owner.Email});
```

```py
            email.setSubject('Reminder: Account Review Needed for ' + acc.Name);
```

```py
            email.setPlainTextBody('Please review and update the account details for ' + acc.Name);
```

```py
            emails.add(email);
```

```py
        }
```

```py
        if (!emails.isEmpty()) {
```

```py
            try {
```

```py
                Messaging.sendEmail(emails);
```

```py
            } catch (Exception e) {
```

```py
                ExceptionForce.logException(e, 'Failure in sending email notifications in AccountReviewNotificationBatch');
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

|  |
| --- |

```py
    public void finish(Database.BatchableContext bc) {
```

```py
        // You can add any post-processing code here if required.
```

```py
    }
```

```py
}
```

|

| **回复** | 提供的用于批量处理的 Apex 代码旨在向账户所有者发送电子邮件，如果他们的账户在过去 30 天内未被修改。尽管代码看起来相对简单，但应考虑与限制器相关的潜在风险：

+   **SOQL 查询限制**：start()方法使用带有 Database.getQueryLocator 的 SOQL 查询。然而，值得注意的是，无论返回多少记录，SOQL 查询都只计算为 1 个限制器，因为它是批处理类的一部分。

+   **电子邮件限制**：Salesforce 在使用 Messaging.sendEmail 方法时，对单封和群发电子邮件都有出站电子邮件限制。请确保您不要超过这些限制。鉴于您为每个账户记录发送一封电子邮件，这可能会成为一个关注点。

+   **CPU 时间限制**：尽管在这个简单的脚本中不太可能发生，但如果您计划在 execute()方法中添加更复杂的逻辑，请留意 CPU 时间。

+   **错误日志记录**：代码尝试捕获在发送电子邮件操作期间抛出的任何异常。请确保 ExceptionForce.logException 方法不执行可能触碰到其他限制器的操作，例如 DML 操作。

|

|  | 以下是一个针对前面代码的测试类的示例：

```py
@isTest
```

```py
private class TestAccountReviewNotificationBatch {
```

```py
    @isTest static void testBatch() {
```

```py
        // Create test Account
```

```py
        Account acc = new Account(Name = 'Test Account', LastModifiedDate = System.now().addDays(-31));
```

```py
        insert acc;
```

```py
        Test.startTest();
```

```py
        // Instantiate the batch class & execute
```

```py
        AccountReviewNotificationBatch batchJob = new AccountReviewNotificationBatch();
```

```py
        Id batchId = Database.executeBatch(batchJob);
```

```py
        Test.stopTest();
```

```py
        // Perform assertions, for example by querying a logging object or inspecting any emails sent.
```

```py
        // Note that we cannot check the Messaging.sendEmail in a test method. It's considered a no-op (no operation).
```

```py
    }
```

```py
}
```

请确保将 AccountReviewNotificationBatch 替换为您的 Batch Apex 类的实际名称。|

ChatGPT 的两个有用功能是它能够轻松生成测试类，以及识别与代码相关的风险的能力。这些功能不仅有助于文档编制，还有助于持续监控，从而有助于我们组织的整体健康。

# 优化和重构 Apex 代码

在许多 Salesforce 项目中，通常最难重构的指标是循环复杂度。循环复杂度是一个软件指标，用于指示程序的复杂度。它是通过开发代码的控制流图来计算的，然后确定通过图中的路径数量。较高的值表示代码更复杂，并且可能更容易出错。

让我们看看一个具有高循环复杂度的类：

```py
public class HighComplexityClass {
    public void complexMethod(Integer val) {
        if (val == 1) {
            System.debug('Value is 1');
        } else if (val == 2) {
            System.debug('Value is 2');
            if (val < 3) {
                System.debug('Value is less than 3');
            }
        } else if (val == 3) {
            System.debug('Value is 3');
        } else if (val == 4) {
            System.debug('Value is 4');
            for (Integer i = 0; i < 5; i++) {
                if (i == 2) {
                    System.debug('Loop value is 2');
                } else {
                    System.debug('Loop value is not 2');
                }
            }
        } else {
            System.debug('Value is not between 1 and 4');
        }
        switch on val {
            when 5 {
                System.debug('Value is 5');
            }
            when 6 {
                System.debug('Value is 6');
            }
            when 7 {
                System.debug('Value is 7');
            }
            when else {
                System.debug('Value is not between 5 and 7');
            }
        }
    }
}
```

此代码有多个分支，这将增加其循环复杂度。然而，请记住，高循环复杂度通常被认为是一种不良实践，因为它使得代码更难阅读、测试和维护。ChatGPT 可以帮助重构类以降低此指标：

| **提示** | 你能请重构这个类以降低循环复杂度吗？

```py
public class HighComplexityClass {
```

```py
    public void complexMethod(Integer val) {
```

```py
        if (val == 1) {
```

```py
            System.debug('Value is 1');
```

```py
        } else if (val == 2) {
```

```py
            System.debug('Value is 2');
```

```py
            if (val < 3) {
```

```py
                System.debug('Value is less than 3');
```

```py
            }
```

```py
        } else if (val == 3) {
```

```py
            System.debug('Value is 3');
```

```py
        } else if (val == 4) {
```

```py
            System.debug('Value is 4');
```

```py
            for (Integer i = 0; i < 5; i++) {
```

```py
                if (i == 2) {
```

```py
                    System.debug('Loop value is 2');
```

```py
                } else {
```

```py
                    System.debug('Loop value is not 2');
```

```py
                }
```

```py
            }
```

```py
        } else {
```

```py
            System.debug('Value is not between 1 and 4');
```

```py
        }
```

```py
       switch on val {
```

```py
            when 5 {
```

```py
                System.debug('Value is 5');
```

```py
            }
```

```py
            when 6 {
```

```py
                System.debug('Value is 6');
```

```py
            }
```

```py
            when 7 {
```

```py
                System.debug('Value is 7');
```

```py
            }
```

|

|  |
| --- |

```py
            when else {
```

```py
                System.debug('Value is not between 5 and 7');
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
}
```

|

| **响应** | 当然可以！降低类中循环复杂度的一种方法是将逻辑分解成更小的方法并最小化条件分支的数量。以下是重构后的版本：

```py
public class RefactoredClass {
```

```py
    public void complexMethod(Integer val) {
```

```py
        handleDebugForValuesOneToFour(val);
```

```py
        handleDebugForValuesFiveToSeven(val);
```

```py
    }
```

```py
    private void handleDebugForValuesOneToFour (Integer val) {
```

```py
        switch on val {
```

```py
            when 1 {
```

```py
                System.debug('Value is 1');
```

```py
            }
```

```py
            when 2 {
```

```py
                System.debug('Value is 2');
```

```py
                debugForValueTwo(val);
```

```py
            }
```

```py
            when 3 {
```

```py
                System.debug('Value is 3');
```

```py
            }
```

```py
            when 4 {
```

```py
                System.debug('Value is 4');
```

```py
                debugForLoopValues();
```

```py
            }
```

```py
            when else {
```

```py
                System.debug('Value is not between 1 and 4');
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

|  |
| --- |

```py
    private void debugForValueTwo(Integer val) {
```

```py
        if (val < 3) {
```

```py
            System.debug('Value is less than 3');
```

```py
        }
```

```py
    }
```

```py
    private void debugForLoopValues() {
```

```py
        for (Integer i = 0; i < 5; i++) {
```

```py
            debugForSpecificLoopValue(i);
```

```py
        }
```

```py
    }
```

```py
    private void debugForSpecificLoopValue(Integer i) {
```

```py
        if (i == 2) {
```

```py
            System.debug('Loop value is 2');
```

```py
        } else {
```

```py
            System.debug('Loop value is not 2');
```

```py
        }
```

```py
    }
```

```py
    private void handleDebugForValuesFiveToSeven (Integer val) {
```

```py
        switch on val {
```

```py
            when 5 {
```

```py
                System.debug('Value is 5');
```

```py
            }
```

```py
            when 6 {
```

```py
                System.debug('Value is 6');
```

```py
            }
```

```py
            when 7 {
```

```py
                System.debug('Value is 7');
```

```py
            }
```

```py
            when else {
```

```py
                System.debug('Value is not between 5 and 7');
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
}
```

通过将逻辑分解成更小的方法，我们降低了循环复杂度，并使代码更易于阅读。这也使得测试和维护变得更加容易。|

ChatGPT 帮助重构代码以解决诸如循环复杂度等问题。通过在初始开发和常规维护期间利用 ChatGPT，我们可以确保我们的代码保持清晰和可理解，并作为后续项目中最佳实践的指南。

现在我们知道了 ChatGPT 如何帮助我们编写代码，让我们看看它如何帮助调试 Salesforce 代码。

其他 PMD 违规怎么办？

| **提示** | 你能检查以下代码有哪些 PMD 违规吗？

```py
public class HighComplexityClass {
```

```py
    public void complexMethod(Integer val) {
```

```py
        if (val == 1) {
```

```py
            System.debug('Value is 1');
```

```py
        } else if (val == 2) {
```

```py
            System.debug('Value is 2');
```

```py
            if (val < 3) {
```

```py
                System.debug('Value is less than 3');
```

```py
            }
```

```py
        } else if (val == 3) {
```

```py
            System.debug('Value is 3');
```

```py
        } else if (val == 4) {
```

```py
            System.debug('Value is 4');
```

```py
            for (Integer i = 0; i < 5; i++) {
```

```py
                if (i == 2) {
```

```py
                    System.debug('Loop value is 2');
```

```py
                } else {
```

```py
                    System.debug('Loop value is not 2');
```

```py
                }
```

```py
            }
```

```py
        } else {
```

```py
            System.debug('Value is not between 1 and 4');
```

```py
        }
```

|

|  |
| --- |

```py
        switch on val {
```

```py
            when 5 {
```

```py
                System.debug('Value is 5');
```

```py
            }
```

```py
            when 6 {
```

```py
                System.debug('Value is 6');
```

```py
            }
```

```py
            when 7 {
```

```py
                System.debug('Value is 7');
```

```py
            }
```

```py
            when else {
```

```py
                System.debug('Value is not between 5 and 7');
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
}
```

|

| **响应** | 虽然我不能运行 PMD，但我可以手动分析代码片段以识别典型的 PMD 规则违规。以下可能会被标记的内容：**循环复杂度**：complexMethod 有很多条件分支，这可能会增加方法的循环复杂度。如果超过某个阈值，PMD 可能会将其标记为违规。**方法长度过长**：该方法相当长，做了很多事情，可能会被标记为违反单一职责原则。**避免深层嵌套的 if 语句**：尽管不是深层嵌套，但多个 if-else 和内部 if 语句也可能导致复杂度增加，可能会被标记。**过多的调试日志**：尽管这不是严格的 PMD 规则，但使用过多的调试日志可能会在故障排除期间难以识别真实问题，并迅速填满您的日志文件。然而，如果这只是示例代码，那么这个点可能不是那么相关。**缺少注释**：PMD 可能会标记缺少注释来解释方法的功能，尽管这更多的是最佳实践而不是 PMD 违规。**魔法数字**：用于条件中的整数可能被认为是“魔法数字”。将它们替换为命名常量是一种良好的做法，可以使代码更易于阅读和维护。请注意，这些是手动检查，可能无法捕获所有 PMD 规则违规。 |
| --- | --- |

在探讨了 ChatGPT 如何优化 Apex 代码以符合最佳实践并提高性能之后，让我们转向错误处理和调试。编写高效代码的能力与在出现问题时进行故障排除和解决问题的技能相辅相成。ChatGPT 可以是一个不可或缺的工具，用于快速识别错误并提供解决这些错误的解决方案，确保 Apex 代码不仅运行顺畅，而且保持 Salesforce 动态环境所需的鲁棒性。

# Apex 中的错误处理和调试

与前一部分类似，ChatGPT 在调试 Salesforce 代码时可以提供多种帮助。虽然 AI 不能直接替代实际的调试工具或开发环境，但它提供战略建议和指导。以下是 ChatGPT 可以有益的两个重要点：

+   **错误解释**：如果您有一个错误消息或异常，您可以将其提供给 ChatGPT，它可以帮助解释其含义，并根据常见原因提出潜在的修复建议。

+   **逻辑和算法辅助**：如果您试图实现特定的功能或逻辑，但得到了意外的结果，向 ChatGPT 描述问题可以帮助您获得有关如何处理或改进您逻辑的建议。

然而，尽管 ChatGPT 在调试过程中是一个有价值的工具，但将它的指导与实际测试、Salesforce 开发者工具和沙箱相结合是至关重要的。人工智能洞察力和实用工具的结合将产生最有效的调试策略。

让我们通过一个实际例子来看看这个问题。执行新类时得到以下异常：

```py
System.SObjectException: SObject row was retrieved via SOQL without querying the requested field: Service_Contract_Products__c.Status__c
```

ChatGPT 可以帮助理解和修复问题：

| **提示** | 你能帮我理解为什么在执行我的类时会出现以下异常吗？System.SObjectException: SObject 行是通过 SOQL 查询而没有查询请求的字段：Service_Contract_Products__c.Status__c |
| --- | --- |
| **响应** | 当然可以！您看到的异常消息“System.SObjectException: SObject 行是通过 SOQL 查询而没有查询请求的字段：Service_Contract_Products__c.Status__c”清楚地表明了出了什么问题。以下是对问题的分解：**发生了什么？** 您在执行 SOQL 查询后尝试访问 SObject 上的字段（在这种情况下，是 Service_Contract_Products__c 对象上的 Status__c 字段）。**错误发生的原因是什么？** Status__c 字段没有包含在您的 SOQL 查询的 SELECT 子句中。然而，您在代码的后续部分尝试引用或使用该字段。### 如何解决它？**审查 SOQL 查询**：回到您的类中，找到您查询 Service_Contract_Products__c 的地方。确保 Status__c 是您要选择的字段之一。   SELECT Id, Name, Status__c FROM Service_Contract_Products__c WHERE ...**确认访问权限**：同时，确保执行代码的用户/配置文件有读取 Status__c 字段的权限。如果用户没有权限，即使该字段存在于 SOQL 中，也无法访问。**最佳实践**：作为一个最佳实践，始终确保您查询了 SObject 中您计划在 Apex 代码中稍后引用的所有字段。这将有助于防止此类异常。最后，如果您不需要查询太多字段，请谨慎添加。过度查询会使您的代码效率降低，尤其是在处理具有许多自定义字段的对象时。始终追求平衡，只查询您需要的数据。 |

ChatGPT 不仅能够识别问题的根本原因，还提供最佳实践以确保更高效的未来开发。此外，它还提供有关构建更有效查询以实现预期结果的指导。

# 最佳实践和持续学习

ChatGPT 可以作为 Salesforce 开发者的强大助手，尤其是在遵循 Apex 最佳实践和培养持续学习文化方面。作为一个能够分析和生成代码片段的对话式 AI，ChatGPT 可以帮助您了解如何以符合 Salesforce 推荐指南的方式构建 Apex 类、触发器和批量作业。从解释限制器到展示正确的错误处理技术，ChatGPT 可以提供实时见解和建议，这可能为您节省数小时的调试和故障排除时间。

此外，软件开发不断演变的领域需要持续学习；ChatGPT 可以在这里提供帮助。无论您是想要了解 Apex 基础知识的初学者，还是寻求高级优化技术的经验丰富的开发者，ChatGPT 都可以调整其响应以满足您的教育需求。凭借其解析和响应复杂查询的能力，您可以以对话方式探索测试驱动开发、批量处理和 API 集成等主题。这使得学习过程更加互动，不那么令人畏惧。

通过利用 ChatGPT，您不仅可以立即获得与 Apex 相关问题的答案，还可以接触到更广泛的发展实践，这些实践您可能尚未遇到。这促进了一个既适应性强又全面的 学习环境，帮助您跟上最新的 Apex 实践，更广泛地，成为一个更熟练的 Salesforce 开发者。

例如，一个很好的例子是找出 ChatGPT 是否能提供包含 Salesforce 最新功能的响应。

| **提示** | 请您编写一个 Apex 类，查询账户对象的所有字段，同时确保执行字段和对象级权限？ |
| --- | --- |

| **响应** | 当然可以！以下是一个查询账户对象所有字段并强制执行对象和字段级安全性的 Apex 类示例：

```py
public class AccountQueryWithSecurityChecks {
```

```py
    public static List<Account> getAccountsEnforcingSecurity() {
```

```py
        // Describe the Account object
```

```py
        Schema.DescribeSObjectResult objDescribe = Schema.SObjectType.Account;
```

```py
        // Check for object-level permission (readable)
```

```py
        if(!objDescribe.isAccessible()) {
```

```py
            System.debug('User does not have access to read Account records.');
```

```py
            return new List<Account>();
```

```py
        }
```

```py
        // Create a list to hold the names of accessible fields
```

```py
        List<String> accessibleFields = new List<String>();
```

|

|  |
| --- |

```py
        // Get the field map
```

```py
        Map<String, Schema.SObjectField> fieldMap = objDescribe.fields.getMap();
```

```py
        // Iterate over the field map to find accessible fields
```

```py
        for(String fieldName : fieldMap.keySet()) {
```

```py
            Schema.DescribeFieldResult fieldDescribe = fieldMap.get(fieldName).getDescribe();
```

```py
            // Check for field-level security (readable)
```

```py
            if(fieldDescribe.isAccessible()) {
```

```py
                accessibleFields.add(fieldName);
```

```py
            }
```

```py
        }
```

```py
        // If no fields are accessible, return an empty list
```

```py
        if(accessibleFields.isEmpty()) {
```

```py
            System.debug('User does not have access to any Account fields.');
```

```py
            return new List<Account>();
```

```py
        }
```

```py
        // Create a SOQL query dynamically
```

```py
        String query = 'SELECT ' + String.join(accessibleFields, ', ') + ' FROM Account';
```

```py
        // Execute the query
```

```py
        List<Account> accounts = Database.query(query);
```

```py
        return accounts;
```

```py
    }
```

```py
}
```

要使用此 Apex 类，调用 getAccountsEnforcingSecurity() 方法，如下所示：

```py
List<Account> accounts = AccountQueryWithSecurityChecks.getAccountsEnforcingSecurity();
```

此方法将返回一个包含仅当前用户有读取访问权限的字段的账户记录列表。如果用户没有对账户对象或其任何字段的读取访问权限，它将返回一个空列表并写入一个调试语句。|

让我们更具体一些：

| **提示** | 请您使用 WITH SECURITY_ENFORCED 子句重写代码，可以吗？ |
| --- | --- |

| **响应** | 当然！WITH SECURITY_ENFORCED 子句是强制执行 SOQL 查询中字段和对象级安全权限的简单方法。以下是您如何更新之前的 Apex 类以使用此子句的示例：

```py
public class AccountQueryWithSecurityEnforced {
```

```py
    public static List<Account> getAccountsEnforcingSecurity() {
```

```py
        List<Account> accounts;
```

```py
        try {
```

```py
            accounts = [SELECT * FROM Account WITH SECURITY_ENFORCED];
```

```py
        } catch (System.QueryException e) {
```

```py
            // Handle the exception when the user doesn't have the necessary permissions
```

```py
            System.debug('You don\'t have access to one or more fields or the object itself: ' + e.getMessage());
```

```py
            accounts = new List<Account>();
```

```py
        }
```

```py
        return accounts;
```

```py
    }
```

```py
}
```

使用此代码，WITH SECURITY_ENFORCED 子句将自动强制执行对象级和字段级安全。如果当前用户无法访问 Account 对象或查询中的任何字段，Salesforce 将抛出 System.QueryException 错误，这可以被捕获并适当处理。要使用此类，只需调用 getAccountsEnforcingSecurity() 方法：

```py
List<Account> accounts = AccountQueryWithSecurityEnforced.getAccountsEnforcingSecurity();
```

此方法将返回包含当前用户所有可访问字段的账户记录列表。如果任何字段或对象不可访问，它将捕获异常并返回空列表。|

你做到了！ChatGPT 知道 Apex 中可用的 Salesforce 功能，但需要具体信息以确保你的蓝图中有这些功能。

# 摘要

本章介绍了如何将 ChatGPT 应用于 Salesforce Apex 开发。它提供了对 Apex 的基础理解，突出了 ChatGPT 作为辅助工具在编码旅程中的作用。它涵盖了 ChatGPT 如何通过提供建议并确保遵循最佳实践来帮助编写 Apex 代码。本章还涉及了使用 ChatGPT 优化和重构 Apex 代码的策略，旨在提高性能和代码可持续性。*Apex 中的错误处理和调试*部分详细说明了 ChatGPT 如何加快故障排除过程，帮助开发者有效地定位和解决编码问题。最后，本章强调了持续学习的重要性，ChatGPT 作为一种工具，可以帮助开发者跟上最新的 Apex 开发实践。

下一章将讨论 Salesforce 通过 Web 服务和调用与各种平台和系统的交互，说明如何利用 ChatGPT 简化此过程。它将建立对 Web 服务和调用的基础理解，概述它们在 Salesforce 与外部通信能力中的关键作用。随后，我们将回顾如何在 Salesforce 中配置 Web 服务，ChatGPT 将协助最佳实践和实施技术。我们还将涵盖如何使用 ChatGPT 优化和管理调用以提高性能和可靠性。下一章将解决与 Web 服务和调用常见的问题，提供使用 ChatGPT 快速诊断和解决这些问题的策略。
