

# 第五章：使用 ChatGPT 为他人编写的 Salesforce Apex 代码

Salesforce Apex 在 Salesforce 生态系统中代表了一个卓越的工具，它允许开发者编写和执行强大且高度可定制的服务器端逻辑。作为 Salesforce 可编程性的动力源泉，Apex 提供了创建动态业务逻辑、自动化流程以及与外部系统集成的能力。与如流程这样的声明性自动化工具不同，Apex 提供了更细粒度的控制，使开发者能够满足可能超出开箱即用工具范围的复杂需求。Apex 的表达式语法源自 Java，使其成为解决复杂商业问题的强大工具。

然而，Apex 的复杂性是一把双刃剑。编写有效、高效且可维护的 Apex 代码需要深入了解软件开发原则、Salesforce 平台的特点以及 Salesforce 施加的限制。Apex 开发者必须精通限制器限制，这些限制器限制了单个事务中的操作数量。在确保代码性能优化的同时处理这些限制成为一个重大挑战。开发者必须不仅从功能的角度思考，还要考虑代码如何在平台限制内运行。

此外，随着 Salesforce 组织在数据量、用户和用途方面的增长，挑战也在升级。随着规模的增加，最初可能有效的 Apex 代码很快就会成为瓶颈。编写可扩展的 Apex 代码需要仔细考虑查询优化、高效内存利用以及对 Salesforce 平台不同部分如何交互的深入了解。这不仅仅是让代码工作，而是构建随着系统需求扩展而持续表现良好的解决方案。

增加复杂性的还有确保代码可维护性和符合最佳实践的需求。随着多个开发者随着时间的推移在代码库上工作，编写不佳或文档不足的 Apex 代码可能成为理解和修改的噩梦。Salesforce Health Check 等工具可以提供优化方面的见解，但通常需要经验丰富的眼睛来辨别可以改进的地方。

虽然 Apex 的强大之处无可否认，但有效地运用它需要技术技能、平台知识和预见未来增长和变化所需的解决方案适应性的洞察力。掌握 Apex 不仅关乎理解语法和结构，还涉及对开发解决方案的战略方法，这些解决方案要高效、可扩展、可维护，并与 Salesforce 环境的复杂本质保持一致。这是一个能够激发巨大创造力和灵活性的工具，但同时也要求对其复杂性和潜在陷阱表示尊重。

Salesforce 的创新方法以每三年一次的发布周期为特征，每年推出三个主要更新。这些发布是 Salesforce 致力于提供最新功能、改进和安全性增强的基石。然而，对于组织来说，这些发布并非可选的；每个 Salesforce 实例都会随着每个新版本的发布而自动升级。虽然这确保了所有客户都能访问最新功能，但也引入了一个独特的挑战，尤其是在 Apex 代码方面。

由于 Apex 代码与底层 Salesforce 平台紧密集成，因此可能会受到这些版本中引入的更改和更新的影响。在一个版本中无缝运行的功能可能会在随后的版本中突然变得过时或行为不同。这需要定期审查，并且通常需要对 Apex 代码进行重构，以确保其继续按预期运行。这不仅仅是跟上新功能，还确保现有功能不会中断或性能下降。这种持续适应可能需要大量的努力，尤其是在更大或更复杂的 Salesforce 实施中。这强调了遵循最佳实践、关注即将到来的变化以及采取积极主动的方法来管理 Apex 代码的重要性。开发者不仅要考虑当前的需求，还要考虑他们的代码如何适应不断发展的 Salesforce 环境，在利用最新平台功能和确保持续稳定性和性能之间保持微妙的平衡。

在软件开发领域，将新的**面向对象编程**（**OOP**）语言描述为“类似于 Java”是一种常见的趋势。这种比较作为一种简写方式，用来传达语言的一些基本特征，例如语法、结构或基本概念。然而，当涉及到 Salesforce Apex 时，这种比较可能会有些误导，并且忽略了将 Apex 与 Java 区分开来的几个关键方面，尤其是在执行限制的背景下。

Apex 中的执行限制为 Salesforce 环境添加了一个独特的复杂性层次。这些限制是系统约束，防止单个用户垄断共享资源，从而确保系统高效运行并避免对他人产生任何负面性能影响。例如，存在对单个事务中可消耗的 SOQL 查询、DML 语句和 CPU 时间的限制。与 Java 不同，在 Java 中，开发者通常有广泛的自由度来编写他们认为合适的代码，Apex 开发者必须始终牢记这些执行限制。

许多 Apex 开发者，尤其是那些刚接触 Salesforce 的人，发现自己是在工作中学习这些概念的。对治理限制缺乏认识往往会导致编写在狭窄的上下文或小数据集中可能运行正确的代码，但在实际条件下或随着组织数据量的增长时可能会遇到限制。这使得代码变得脆弱，并可能导致意外的失败。

这通常会导致需要对代码进行重构的需求，即必须重写原始代码以使其更高效并能够抵御治理限制。重构可能是一个复杂且耗时的过程，需要深入理解业务逻辑以及 Apex 本身的细微差别。

将 Apex 描述为“类似于 Java”可能是一把双刃剑。虽然这可能使熟悉 Java 的人更容易接近这种语言，但它也可能导致对治理限制带来的具体挑战产生误解和疏忽。更好的方法可能是认识到 Apex 是一种具有其独特约束和能力的语言，它要求对那些来自其他面向对象背景的人来说可能并不立即明显的一些细节给予仔细的关注。学习曲线可能更陡峭，但回报是代码不仅功能强大，而且在 Salesforce 环境中既稳健又可扩展。

Apex 的复杂性可能会将调试或更新现有 Apex 代码变成一项非平凡的任务，通常需要微妙地解开意外的副作用和复杂的依赖关系。这正是 ChatGPT 真正发光的地方。通过提供即时的洞察和指导，ChatGPT 可以弥合知识差距，帮助新手和经验丰富的开发者快速导航他人复杂代码的复杂领域。开发者无需花费数天甚至更长的时间追踪细微的问题，而是可以利用 ChatGPT 加速完成分配的工作，提高效率并提升代码的整体质量。

# 理解现有的 Apex 代码

ChatGPT 分析未注释且不清晰的 Apex 代码的能力证明了其对编程原则和语法的复杂理解。即使面对缺乏有意义的变量名、方法名或解释性注释的代码，ChatGPT 也能剖析代码中的逻辑、结构和操作，以提供对代码功能的准确描述。通过识别方法调用的顺序、变量之间的关系以及涉及的计算过程，ChatGPT 可以阐明代码的潜在功能。然而，必须认识到，业务背景或代码背后的具体目的可能无法从代码本身推断出来，尤其是在代码被混淆或故意弄得不清楚的情况下。因此，虽然 ChatGPT 可以阐明代码的技术方面，但要理解为什么它这样做可能需要额外的上下文或代码本身之外的信息。

| **提示** | 这段代码的功能是什么？

```py
public class AppleBananaCherry {
```

```py
    private static final Double CAT = 3958.8;
```

```py
    private static Double DOG(Double a, Double b, Double c, Double d) {
```

```py
        Double E = toFish(c - a);
```

```py
        Double F = toFish(d - b);
```

```py
        Double G = Math.sin(E / 2) * Math.sin(E / 2) +
```

```py
                   Math.cos(toFish(a)) * Math.cos(toFish(c)) *
```

```py
                   Math.sin(F / 2) * Math.sin(F / 2);
```

```py
        Double H = 2 * Math.atan2(Math.sqrt(G), Math.sqrt(1 - G));
```

```py
        return CAT * H;
```

```py
    }
```

```py
    private static Double toFish(Double frog) {
```

```py
        return frog * Math.PI / 180;
```

```py
    }
```

```py
    public static void elephantHorse(Id giraffe) {
```

```py
        Account lion = [SELECT BillingLatitude, BillingLongitude FROM Account WHERE Id = :giraffe LIMIT 1];
```

```py
        Integer monkey = 0;
```

```py
        Integer snake = 0;
```

```py
        for (Contact zebra : [SELECT MailingLatitude, MailingLongitude FROM Contact WHERE AccountId = :giraffe]) {
```

```py
            Double panda = DOG(lion.BillingLatitude, lion.BillingLongitude, zebra.MailingLatitude, zebra.MailingLongitude);
```

```py
            if (panda < 20) {
```

```py
                monkey++;
```

```py
            } else {
```

```py
                snake++;
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
        System.debug('Output 1: ' + monkey);
```

```py
        System.debug('Output 2: ' + snake);
```

```py
    }
```

```py
}
```

|

| **响应** | 给定的代码在 Salesforce 的 Apex 语言中定义了一个名为 `AppleBananaCherry` 的类，该类负责计算账户地址与其关联联系人地址之间的地理距离。以下是代码的功能概述：1. 常量定义：常量 `CAT` 被定义为值 3958.8，这似乎是地球的英里半径，是距离计算中常用的一个值。2. 距离计算（Haversine 公式）：私有方法 `DOG` 使用 Haversine 公式计算两个经纬度对之间的地理距离。`toFish` 方法将度数转换为弧度。3. 主要处理方法（`elephantHorse`）：这个公共方法接受一个账户 ID（`giraffe`）作为输入。它检索给定账户的账单地址的经纬度。然后查询与该账户关联的所有联系人，并检索他们的邮寄地址的经纬度。对于每个联系人，它调用 `DOG` 方法来计算账户账单地址与联系人邮寄地址之间的距离。如果计算出的距离小于 20 英里，则增加计数器 `monkey`。如果距离为 20 英里或更多，则增加计数器 `snake`。最后，它将 `monkey` 和 `snake` 的值输出到调试日志中。这些值分别代表距离账户地址小于 20 英里和 20 英里或更远的联系人的数量。类、方法和变量名故意具有误导性，与代码的实际功能没有明显的关联。尽管如此，实际的代码实现了基于 Salesforce 组织中账户与联系人之间关系的特定地理分析。 |
| --- | --- |

在本书撰写之时，向 ChatGPT 发送的提示词大小存在限制。虽然 ChatGPT 是一个功能极其强大的工具，能够处理包括 Apex 代码分析在内的广泛查询，但认识到与输入长度和复杂度相关的限制是至关重要的。

对于大型 Apex 类或模块，可能有必要以分段的方式提出查询，通过模块或甚至每次分析较小的 Apex 代码片段来分解分析。这种方法确保了查询与提示词大小限制相一致，并允许对代码的特定部分进行更详细、更专注的分析。以这种方式逐步工作可以非常有益，尤其是在处理 Salesforce 生态系统中的复杂逻辑或多个相互关联的组件时。这有助于保持清晰，支持高效的故障排除，并使开发者能够对代码的功能和结构获得精确的见解，从而增强他们对代码的理解和操作能力。

# 识别和纠正 Apex 代码中的问题

ChatGPT 分析 Apex 代码的能力在揭示可能并不立即显而易见的问题方面是一个宝贵的资产。凭借对 Salesforce 最佳实践和限制条件的深刻理解，ChatGPT 能够检查代码，即使在有限的场景下看似正常工作，也能识别可能导致可扩展性和维护性挑战的问题区域。通过评估代码结构、查询位置、批量操作处理以及其他因素，ChatGPT 可以提供可操作的见解和建议，以重构代码，使其与推荐的设计原则保持一致。这种能力不仅对早期从事 Salesforce 开发的开发者有益，也为经验丰富的开发者提供了一个额外的视角，确保代码健壮且适用于现实世界场景，无论数据量或复杂性如何。

| **提示** | 这段代码中有什么不符合 Salesforce 最佳实践，您建议重写吗？如果是，请指出错误所在，为什么它是错误的，以及应该如何修复。

```py
public class AnalyzeContactDistance {
```

```py
    private static final Double EARTH_RADIUS = 3958.8;
```

```py
    private static Double calculateDistance(Double lat1, Double lon1, Double lat2, Double lon2) {
```

```py
        Double dLat = toRadians(lat2 - lat1);
```

```py
        Double dLon = toRadians(lon2 - lon1);
```

```py
        Double a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
```

```py
                   Math.cos(toRadians(lat1)) * Math.cos(toRadians(lat2)) *
```

```py
                   Math.sin(dLon / 2) * Math.sin(dLon / 2);
```

```py
        Double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
```

```py
        return EARTH_RADIUS * c;
```

```py
    }
```

```py
    private static Double toRadians(Double deg) {
```

```py
        return deg * Math.PI / 180;
```

```py
    }
```

```py
    public static void analyzeAccountContacts(Id accountId) {
```

```py
        Account account = [SELECT BillingLatitude, BillingLongitude FROM Account WHERE Id = :accountId LIMIT 1];
```

```py
       // Below query is inside the loop, a classic example of poor design leading to governor limit issues.
```

```py
        List<Contact> contacts = [SELECT MailingLatitude, MailingLongitude FROM Contact WHERE AccountId = :accountId];
```

```py
        Integer nearCount = 0;
```

```py
        Integer farCount = 0;
```

|

|  |
| --- |

```py
        for (Contact contact : contacts) {
```

```py
            for(Contact innerContact : [SELECT MailingLatitude, MailingLongitude FROM Contact WHERE Id = :contact.Id]) {
```

```py
                Double distance = calculateDistance(account.BillingLatitude, account.BillingLongitude, innerContact.MailingLatitude, innerContact.MailingLongitude);
```

```py
                if (distance < 20) {
```

```py
                    nearCount++;
```

```py
                } else {
```

```py
                    farCount++;
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
        System.debug('Nearby Contacts: ' + nearCount);
```

```py
        System.debug('Distant Contacts: ' + farCount);
```

```py
    }
```

```py
}
```

|

| **响应** | 此代码存在几个重大问题：循环内的 SOQL 查询：随着数据量的增加，循环内的嵌套查询可能会使代码达到单个执行上下文中 SOQL 查询数量的限制器限制。这是 Salesforce 开发中的经典反模式。代码结构低效：冗余查询以及计算方式低效意味着即使没有达到低数据量下的限制器限制，此代码也可能会表现不佳。缺乏批量处理考虑：此代码没有考虑批量处理，这使得它不适合同时处理大量记录。此类代码可能在小型数据集的初始测试中通过，但随着数据量的增长或在生产环境中，它高度容易失败，导致维护困难以及潜在的商务中断。这说明了理解限制器限制并遵循 Salesforce 最佳实践的重要性。 |
| --- | --- |

除了识别 Apex 代码中的问题之外，ChatGPT 还具有重写代码以解决识别到的问题的能力，与最佳实践保持一致并优化性能。这可以在加速代码精炼过程并确保遵循稳健的编码标准方面成为一种宝贵的资产。然而，必须强调的是，ChatGPT 提供的重写代码不应盲目信任或未经进一步审查就实施。虽然这种帮助基于公认的最佳实践和专家见解，但重要的是，修订后的代码应经过全面的测试周期，以确保它不仅解决了识别到的问题，而且表现出正确的期望行为。这种测试确保与业务需求和功能规范保持一致，避免由于修改而产生的不期望的副作用。与任何开发过程一样，与开发团队、质量保证和领域专家的合作始终是至关重要的，以验证和微调解决方案。

让我们看看 ChatGPT 如何重写此代码：

| **提示** | 请重写此 Apex 代码以解决所有问题，并确保其遵循 Salesforce 最佳实践且可扩展。请注意任何应考虑并记录下来以在未来解决的问题。 |
| --- | --- |

| **响应** | 当然！以下是经过修订的 Apex 代码，遵循 Salesforce 最佳实践以确保可扩展性和健壮性。

```py
public class AnalyzeContactDistance {
```

```py
    private static final Double EARTH_RADIUS = 3958.8;
```

```py
    private static Double calculateDistance(Double lat1, Double lon1, Double lat2, Double lon2) {
```

```py
        Double dLat = toRadians(lat2 - lat1);
```

```py
        Double dLon = toRadians(lon2 - lon1);
```

```py
        Double a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
```

```py
                   Math.cos(toRadians(lat1)) * Math.cos(toRadians(lat2)) *
```

```py
                   Math.sin(dLon / 2) * Math.sin(dLon / 2);
```

```py
        Double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
```

```py
        return EARTH_RADIUS * c;
```

```py
    }
```

```py
    private static Double toRadians(Double deg) {
```

```py
        return deg * Math.PI / 180;
```

```py
    }
```

```py
   public static void analyzeAccountContacts(Id accountId) {
```

```py
        Account account = [SELECT BillingLatitude, BillingLongitude FROM Account WHERE Id = :accountId LIMIT 1];
```

```py
        List<Contact> contacts = [SELECT MailingLatitude, MailingLongitude FROM Contact WHERE AccountId = :accountId]; // Removed SOQL from the loop
```

|

|  |
| --- |

```py
        Integer nearCount = 0;
```

```py
        Integer farCount = 0;
```

```py
        for (Contact contact : contacts) {
```

```py
            Double distance = calculateDistance(account.BillingLatitude, account.BillingLongitude, contact.MailingLatitude, contact.MailingLongitude);
```

```py
            if (distance < 20) {
```

```py
                nearCount++;
```

```py
            } else {
```

```py
                farCount++;
```

```py
            }
```

```py
        }
```

```py
        System.debug('Nearby Contacts: ' + nearCount);
```

```py
        System.debug('Distant Contacts: ' + farCount);
```

```py
    }
```

```py
}
```

通过从循环中移除内部 SOQL 查询，修订后的代码避免了与循环中查询执行相关的限制器限制问题。这确保了代码具有更好的可扩展性和与 Salesforce 最佳实践的更好一致性。剩余考虑事项：该方法仍然在没有任何限制的情况下访问与帐户关联的所有联系人，如果联系人数量很多，可能会引起关注。根据实际的用法模式和数据处理量，未来可能需要实现某种形式的分页或过滤标准。对方法进行单元测试将至关重要，以验证修订代码的正确性。适当的测试应确保方法在各种场景和不同的数据输入下表现如预期，考虑到所有可能的边缘情况。硬编码的距离阈值（20 英里）可以通过自定义设置或元数据进行配置，从而提高灵活性和可维护性。

在 Salesforce 开发背景下，拥有一个完整的测试类集，包括正面和负面案例，其重要性不容忽视。正面测试用例验证代码在预期条件下表现正确，并满足定义的要求。另一方面，负面测试用例确保代码能够优雅地处理意外情况、错误和边缘情况，而不会导致失败或意外行为。当一个类被重构或重写时，就像我们借助 ChatGPT 所做的那样，创建新的测试类来反映修改至关重要。这些新创建的测试类必须彻底覆盖更新的功能，包括代码中所有可能的路径和逻辑分支。通过勤奋地测试“快乐路径”和潜在的失败模式，开发者可以建立信心，相信修订后的代码是健壮的、可靠的，并且符合 Salesforce 最佳实践。它建立了一个强大的基础，不仅确保代码质量，还促进了未来的开发和维护工作。

| **提示** | 请为重写的 AnalyzeContactDistance 类编写正面和负面测试类。包括一个具有 200 个联系人的测试。 |
| --- | --- |

| **响应** | 当然！以下是对重写的 `AnalyzeContactDistance` 类的正面和负面测试类。我们将从编写测试设置开始，然后进行正面和负面测试场景。200 个联系人的测试将包含在正面测试场景中。测试设置

```py
@isTest
```

```py
private class AnalyzeContactDistanceTestSetup {
```

```py
    @TestSetup
```

```py
    static void setupTestData() {
```

```py
        Account acc = new Account(Name = 'Test Account', BillingLatitude = 37.7749, BillingLongitude = -122.4194);
```

```py
        insert acc;
```

```py
        List<Contact> contacts = new List<Contact>();
```

```py
        for (Integer i = 0; i < 200; i++) {
```

```py
            contacts.add(new Contact(LastName = 'Contact' + i, MailingLatitude = 37.7749, MailingLongitude = -122.4194, AccountId = acc.Id));
```

```py
        }
```

```py
        insert contacts;
```

```py
    }
```

```py
}
```

|

|  | 正面测试类

```py
@isTest
```

```py
private class AnalyzeContactDistancePositiveTest {
```

```py
    @isTest
```

```py
    static void testAnalyzeAccountContacts() {
```

```py
        Account testAccount = [SELECT Id FROM Account WHERE Name = 'Test Account' LIMIT 1];
```

```py
        Test.startTest();
```

```py
        AnalyzeContactDistance.analyzeAccountContacts(testAccount.Id);
```

```py
        Test.stopTest();
```

```py
        // You may include System.assert statements here to validate expected outcomes.
```

```py
        // This could include querying related records and checking that the correct actions have been taken.
```

```py
    }
```

```py
    @isTest
```

```py
    static void testAnalyzeAccountWith200Contacts() {
```

```py
        Account testAccount = [SELECT Id FROM Account WHERE Name = 'Test Account' LIMIT 1];
```

```py
        Test.startTest();
```

```py
        AnalyzeContactDistance.analyzeAccountContacts(testAccount.Id);
```

```py
        Test.stopTest();
```

```py
        // Include specific assertions to validate the behavior with 200 contacts.
```

```py
    }
```

```py
}
```

|

|  | 负面测试类

```py
@isTest
```

```py
private class AnalyzeContactDistanceNegativeTest {
```

```py
    @isTest
```

```py
    static void testAnalyzeAccountContactsWithNullId() {
```

```py
        Test.startTest();
```

```py
        try {
```

```py
            AnalyzeContactDistance.analyzeAccountContacts(null);
```

```py
            System.assert(false, 'Expected an exception but none was thrown.');
```

```py
        } catch (Exception e) {
```

```py
            System.assertEquals('Some expected error message', e.getMessage());
```

```py
        }
```

```py
        Test.stopTest();
```

```py
    }
```

```py
}
```

注意：将 `’Some expected error message’` 替换为传递 null `Id` 时应抛出的实际错误消息。正面测试类中的断言将取决于重写的 `AnalyzeContactDistance` 类的具体期望，因此您可能需要根据修订的逻辑和期望的结果进行自定义。|

使用 ChatGPT 编写 Apex 测试类可以确保新功能符合所需标准并按预期行为。让 ChatGPT 为现有的 Apex 添加注释可以增强代码库的可维护性和可理解性。借助 ChatGPT 的能力，现有的 Apex 代码可以通过解释性注释进行标注，从而在复杂逻辑和工作流程上提供清晰的说明。这种转变不仅有助于开发者立即理解代码，还作为未来参考的重要文档，确保代码的意图和功能对后续工作的任何人都透明且易于访问。

# 在 Apex 代码中添加和管理注释

注释，包括内联注释和注释头，长期以来一直是软件开发的重要方面。它充当路线图，引导开发者通过复杂的代码片段，提供特定解决方案实施的原因的背景信息，并概述函数的预期输入和输出。注释良好的代码可以显著降低新团队成员的学习曲线，并促进在同一代码库上工作的开发者之间的更顺畅合作。

ChatGPT 在解释和注释代码方面表现出值得注意的熟练度。通过分析代码结构、变量和逻辑流程，它可以生成提供代码功能见解的注释，甚至有助于方法和类的自动文档化。

然而，工具如 ChatGPT 解析未注释代码并即时解释其功能的能力不断增强，这确实引发了关于代码注释未来的一些引人入胜的问题。随着这些工具变得更加复杂和易于访问，大量注释的需求可能会减少，将重点转向编写干净且自我解释的代码。这并不否定注释的价值，尤其是在逻辑复杂或业务规则和上下文从代码本身不明显的情况下，但它确实标志着开发者处理代码可读性的潜在演变。注释代码的“秋天”可能确实已经到来，但通过恰当放置的注释所传达的智慧和清晰性，在确保可维护和可理解的代码方面可能会保持其价值。

| **提示** | 请为 AnalyzeContactDistance 类添加一个完整的注释集，包括完整的注释头和内联注释。 |
| --- | --- |

| **响应** | 当然！以下是带有完整注释集的 `AnalyzeContactDistance` 类，包括完整的注释头和内联注释，以描述类、方法和关键逻辑的目的和功能。

```py
/**
```

```py
 * Class: AnalyzeContactDistance
```

```py
 * Description: This class is designed to analyze the geographical distance between an Account's address and its associated Contacts' addresses.
```

```py
 * It calculates the number of Contacts that are less than 20 miles from the Account's address and the number that are more than 20 miles away.
```

```py
 *
```

```py
 * Usage:
```

```py
 * AnalyzeContactDistance.analyzeAccountContacts(accountId);
```

```py
 *
```

```py
 * Constants:
```

```py
 * EARTH_RADIUS - Radius of the earth used in distance calculation (in miles).
```

```py
 */
```

```py
public class AnalyzeContactDistance {
```

```py
    private static final Double EARTH_RADIUS = 3958.8; // Earth's radius in miles
```

|

|  |
| --- |

```py
    /**
```

```py
     * Method: calculateDistance
```

```py
     * Description: Calculates the distance between two geographical points using the Haversine formula.
```

```py
     * Parameters: lat1, lon1, lat2, lon2 - Geographical coordinates of two locations.
```

```py
     * Returns: Distance in miles between the two locations.
```

```py
     */
```

```py
    private static Double calculateDistance(Double lat1, Double lon1, Double lat2, Double lon2) {
```

```py
        Double dLat = toRadians(lat2 - lat1); // Difference in latitude converted to radians
```

```py
        Double dLon = toRadians(lon2 - lon1); // Difference in longitude converted to radians
```

```py
        Double a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
```

```py
                   Math.cos(toRadians(lat1)) * Math.cos(toRadians(lat2)) *
```

```py
                   Math.sin(dLon / 2) * Math.sin(dLon / 2); // Haversine formula part
```

```py
        Double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)); // Inverse Haversine formula
```

```py
        return EARTH_RADIUS * c; // Return distance
```

```py
    }
```

```py
    /**
```

```py
     * Method: toRadians
```

```py
     * Description: Converts degrees to radians.
```

```py
     * Parameter: deg - Angle in degrees.
```

```py
     * Returns: Angle in radians.
```

```py
     */
```

```py
    private static Double toRadians(Double deg) {
```

```py
        return deg * Math.PI / 180;
```

```py
    }
```

```py
   /**
```

```py
     * Method: analyzeAccountContacts
```

```py
     * Description: Analyzes the distance between an Account's address and its Contacts' addresses,
```

```py
     * categorizing them into two groups based on whether they are closer or farther than 20 miles from the Account.
```

```py
     * Parameter: accountId - The ID of the Account to analyze.
```

|

|  |
| --- |

```py
     * Side Effects: Logs the counts of nearby and distant Contacts.
```

```py
     */
```

```py
    public static void analyzeAccountContacts(Id accountId) {
```

```py
        Account account = [SELECT BillingLatitude, BillingLongitude FROM Account WHERE Id = :accountId LIMIT 1]; // Retrieve Account
```

```py
        List<Contact> contacts = [SELECT MailingLatitude, MailingLongitude FROM Contact WHERE AccountId = :accountId]; // Retrieve Contacts
```

```py
        Integer nearCount = 0; // Count for Contacts near the Account
```

```py
        Integer farCount = 0; // Count for Contacts far from the Account
```

```py
        // Loop through Contacts to calculate distance
```

```py
        for (Contact contact : contacts) {
```

```py
            Double distance = calculateDistance(account.BillingLatitude, account.BillingLongitude, contact.MailingLatitude, contact.MailingLongitude);
```

```py
            if (distance < 20) {
```

```py
                nearCount++; // Increment near count if distance is less than 20 miles
```

```py
            } else {
```

```py
                farCount++; // Increment far count if distance is 20 miles or more
```

```py
            }
```

```py
        }
```

```py
       System.debug('Nearby Contacts: ' + nearCount); // Log the count of nearby Contacts
```

```py
        System.debug('Distant Contacts: ' + farCount); // Log the count of distant Contacts
```

```py
    }
```

```py
}
```

上述注释提供了对类及其功能的全面概述，使代码对未来的开发者来说更具可读性和可维护性。 |

ChatGPT 在 Salesforce 生态系统中的集成代表了现有 Apex 代码分析、改进和维护方式的一个重大进步。通过利用 ChatGPT 的能力，开发者可以快速剖析和理解复杂且未注释的代码，识别可能需要优化或与最佳实践对齐的区域。分析不仅限于识别，还扩展到提供可操作见解和指南，说明如何重构代码以提升性能、可扩展性和可读性。这种方法加速了代码审查过程，确保了更高的代码质量标准。

ChatGPT 重写现有 Apex 代码以提高其效率的能力，不仅超越了简单的语法调整；它专注于与 Salesforce 最佳实践和架构原则相一致的战略性代码重构。这种重写能力支持开发者处理遗留代码，并确保即使数据量增加，代码仍然保持稳健和合规。虽然重写的代码仍需经过严格的测试以确认其功能，但这种 AI 辅助标志着开发和维护任务处理方式的一个重大转变。

此外，ChatGPT 自动生成全面注释的能力，包括内联和标题注释，为代码文档引入了一种细微的视角。虽然注释在理解代码意图和功能方面始终至关重要，但 ChatGPT 理解未注释代码并提供清晰解释的能力为代码维护带来了新的活力。它暗示了一个未来，其中维护现有代码不再是乏味或耗时的工作，而是一个高效、AI 辅助的过程，强调协作、学习和持续改进。这种范式转变预示着一个激动人心的时代，其中人类专业知识和 AI 融合，将软件开发提升到新的高度。

# 摘要

本章分享了理解和更新现有 Salesforce Apex 代码的实用知识。这些知识包括剖析现有 Apex 代码的功能，深入了解其预期操作及其在 Salesforce 组织中的角色，识别代码中问题的策略，以及概述修复和优化的系统方法。此外，它强调了在 Apex 代码中注释的重要性，教导读者如何使用 ChatGPT 编写清晰且具有目的性的注释，以促进更轻松的维护和促进开发团队之间的更好协作。具备这些技能，开发者将更好地准备确保通过有效管理遗留 Apex 代码来保证业务流程的连续性。

下一章开始探讨如何使用 ChatGPT 进行 Salesforce Apex 开发。它首先介绍了 Apex 的强大功能和 ChatGPT 在开发生命周期中可以扮演的支持性角色。接着，章节引导读者了解如何利用 ChatGPT 编写 Apex 代码的细微之处，强调该工具提供编码辅助和强制执行最佳实践的能力。错误处理和调试也得到了涵盖，ChatGPT 作为快速识别和解决问题的资源。最后，章节强调了持续关注 Apex 编码最佳实践的重要性，将 ChatGPT 定位为开发者持续学习和技能提升的有价值盟友。
