# 附录 A：Scala 中的函数式编程

Scala 将函数式编程和面向对象编程结合在一个高级语言中。该附录包含了有关 Scala 中函数式编程原则的参考。

# 函数式编程（FP）

在函数式编程中，函数是第一类公民——这意味着它们像其他值一样被对待，可以作为参数传递给其他函数，或者作为函数的返回结果。在函数式编程中，还可以使用所谓的字面量形式来操作函数，无需为其命名。让我们看一下以下的 Scala 示例：

```py
val integerSeq = Seq(7, 8, 9, 10)
integerSeq.filter(i => i % 2 == 0)
```

`i => i % 2 == 0`是一个没有名称的函数字面量。它检查一个数字是否为偶数。它可以作为另一个函数的参数传递，或者可以作为返回值使用。

# 纯度

函数式编程的支柱之一是纯函数。一个纯函数是类似于数学函数的函数。它仅依赖于其输入参数和内部算法，并且对于给定的输入始终返回预期的结果，因为它不依赖于外部的任何东西。（这与面向对象编程的方法有很大的不同。）你可以很容易理解，这使得函数更容易测试和维护。一个纯函数不依赖外部的任何内容，这意味着它没有副作用。

纯粹的函数式程序在不可变数据上进行操作。与其修改现有的值，不如创建修改后的副本，而原始值则被保留。这意味着它们可以在旧副本和新副本之间共享，因为结构中未改变的部分无法被修改。这样的行为带来的一个结果是显著的内存节省。

在 Scala（以及 Java）中，纯函数的例子包括`List`的`size`方法（[`docs.oracle.com/javase/8/docs/api/java/util/List.html`](https://docs.oracle.com/javase/8/docs/api/java/util/List.html)）或者`String`的`lowercase`方法（[`docs.oracle.com/javase/8/docs/api/java/lang/String.html`](https://docs.oracle.com/javase/8/docs/api/java/lang/String.html)）。`String`和`List`都是不可变的，因此它们的所有方法都像纯函数一样工作。

但并非所有抽象都可以直接通过纯函数实现（例如读取和写入数据库或对象存储，或日志记录等）。FP 提供了两种方法，使开发人员能够以纯粹的方式处理不纯抽象，从而使最终代码更加简洁和可维护。第一种方法在某些其他 FP 语言中使用，但在 Scala 中没有使用，即通过将语言的纯函数核心扩展到副作用来实现。然后，避免在只期望纯函数的情况下使用不纯函数的责任就交给开发人员。第二种方法出现在 Scala 中，它通过引入副作用来模拟纯语言中的副作用，使用*monads*（[`www.haskell.org/tutorial/monads.html`](https://www.haskell.org/tutorial/monads.html)）。这样，虽然编程语言保持纯粹且具有引用透明性，但 monads 可以通过将状态传递到其中来提供隐式状态。编译器不需要了解命令式特性，因为语言本身保持纯粹，而通常实现会出于效率原因了解这些特性。

由于纯计算具有引用透明性，它们可以在任何时间执行，同时仍然产生相同的结果，这使得计算值的时机可以延迟，直到真正需要时再进行（懒惰计算）。这种懒惰求值避免了不必要的计算，并允许定义和使用无限数据结构。

通过像 Scala 一样仅通过 monads 允许副作用，并保持语言的纯粹性，使得懒惰求值成为可能，而不会与不纯代码的副作用冲突。虽然懒惰表达式可以按任何顺序进行求值，但 monad 结构迫使这些副作用按正确的顺序执行。

# 递归

递归在函数式编程（FP）中被广泛使用，因为它是经典的，也是唯一的迭代方式。函数式语言的实现通常会包括基于所谓的**尾递归**（[`alvinalexander.com/scala/fp-book/tail-recursive-algorithms`](https://alvinalexander.com/scala/fp-book/tail-recursive-algorithms)）的优化，以确保重度递归不会对内存消耗产生显著或过度的影响。尾递归是递归的一个特例，其中函数的返回值仅仅是对自身的调用*。* 以下是一个使用 Scala 语言递归计算斐波那契数列的例子。第一段代码表示递归函数的实现：

```py
def fib(prevPrev: Int, prev: Int) {
    val next = prevPrev + prev
    println(next)
    if (next > 1000000) System.exit(0)
    fib(prev, next)
}
```

另一段代码表示同一函数的尾递归实现：

```py
def fib(x: Int): BigInt = {
    @tailrec def fibHelper(x: Int, prev: BigInt = 0, next: BigInt = 1): BigInt = x match {
        case 0 => prev
        case 1 => next
        case _ => fibHelper(x - 1, next, (next + prev))
    }
    fibHelper(x)
}
```

虽然第一个函数的返回行包含对自身的调用，但它还对输出做了一些处理，因此返回值并不完全是递归调用的返回值。第二个实现是一个常规的递归（特别是尾递归）函数。
