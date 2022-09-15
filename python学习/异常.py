import logging

"""
    异常：try···except···else···finally
    try：放在容易出错的代码前
    except：用于捕获异常，遇见异常时的处理代码
        可以有多个except去捕获不同的异常类型
        不但捕获该类型的异常，也会捕获此类异常的子类
    else：try没有发生异常时的处理代码
    finally：无论是否发生异常都要执行的代码
    raise：主动抛出一个异常

    try 语句按照如下方式工作；
        首先，执行 try 子句（在关键字 try 和关键字 except 之间的语句）。
        如果没有异常发生，忽略 except 子句，try 子句执行后结束。
        如果在执行 try 子句的过程中发生了异常，那么 try 子句余下的部分将被忽略。如果异常的类型和 except 之后的名称相符，那么对应的 except 子句将被执行。
        如果一个异常没有与任何的 except 匹配，那么这个异常将会传递给上层的 try 中。

    使用try…except的好处在于可以跨越多层调用，比如函数main()调用foo()，foo()调用bar()，结果bar()出错了，这时，只要main()捕获到了，就可以处理
"""


def _exception():
    def foo(s):
        return 10 / int(s)

    def bar(s):
        return foo(s) * 2

    def main():
        try:    # 检查错误
            bar('0')
        except Exception as e:  # 捕获错误
            print('Error:', e)
        else:
            print("没有出错")   # 没有出错要执行的
        finally:    # 必须要执行的
            print('finally...')


def _thrown_error():
    """
        使用raise抛出一个自定义异常，只有在必要的时候才定义我们自己的错误类型。
        如果可以选择Python已有的内置的错误类型（比如ValueError，TypeError），尽量使用Python内置的错误类型。
    :return:
    """
    class FooError(ValueError):
        pass

    def foo(s):
        n = int(s)
        if n == 0:
            raise FooError(f'invalid value: {s}')
        return 10 / n

    foo('0')


def _logging():
    """
        如果不捕获错误，自然可以让Python解释器来打印出错误堆栈，但程序也被结束了。
        既然我们能捕获错误，就可以把错误堆栈打印出来，然后分析错误原因，同时，让程序继续执行下去。

    """

    def foo(s):
        return 10 / int(s)

    def bar(s):
        return foo(s) * 2

    def main():
        try:
            bar('0')
        except Exception as e:
            logging.exception(e)  # 使用logging模块记录错误信息

    main()
    print('END')


def _assert():
    """
        assert的意思是，前面位置的表达式应该是True，否则，根据程序运行的逻辑，后面的代码肯定会出错。
        如果断言失败，assert语句本身就会抛出AssertionError
    :return:
    """

    def foo(s):
        n = int(s)
        assert n != 0, 'n is zero!'
        return 10 / n

    def main():
        foo('0')

