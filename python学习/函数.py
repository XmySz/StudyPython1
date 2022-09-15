from functools import reduce
import functools


def _params():
    """
    参数类型:
        1.必备参数,必须传入且顺序固定
        2.默认参数,函数的形参里有个默认值
        3.变长参数, *args
        4.关键字参数,**kw
        不定长参数 *args 输入是元祖，输出一般都是元组的结构形式
                  **kw 输入是包含参数名的参数，输出的都是字典形式的结构
    """

    def f1(a, b, c=0, *args, **kw):
        print('a =', a, 'b =', b, 'c =', c, 'args =', args, 'kw =', kw)

    f1(1, 2)
    f1(1, 2, c=3)
    f1(1, 2, 3, 'a', 'b')
    f1(1, 2, 3, 'a', 'b', x=9, y=10)


def _lambda():
    """
    形式: lambda [arg1 [,arg2,.....argn]]:expression
    """
    sum = lambda x, y: x + y


def _advance_function():
    """
        一些函数的高级用法
    :return:
    """
    # 变量可以指向函数
    f = abs
    print(f(-100))
    print(abs)

    # 函数名也可以是变量(不再拥有原本的功能)
    abs = 10

    # 函数还可以作为另外一个函数的参数
    def add(x, y, f):
        return f(x) + f(y)

    # 函数里可以定义函数，外部无法访问函数里定义的函数
    def hi(name="yasoob"):
        print("now you are inside the hi() function")

        def greet():
            return "now you are in the greet() function"

        def welcome():
            return "now you are in the welcome() function"

        print(greet())
        print(welcome())
        print("now you are back in the hi() function")

    # 函数里可以返回函数
    def _hi(name="yasoob"):
        def greet():
            return "now you are in the greet() function"

        def welcome():
            return "now you are in the welcome() function"

        return greet if name == "yasoob" else welcome

    a = hi()
    b = _hi()
    print(a)
    print(b())


def _map():
    """
        map()函数接收两个参数，一个是函数，一个是Iterable，
        map将传入的函数依次作用到序列的每个元素，并把结果作为新的Iterator返回。
        map(func,iterable)
    """

    def f(x):
        return x ** 2

    def f1(x):
        return str(x)

    print(list(map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0])))
    print(list(map(f1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0])))


def _reduce():
    """
        reduce(func,iterable) reduce函数将func作用在iterable序列上,
        这个func必须接收两个参数，reduce把结果继续和序列的下一个元素做累积计算
    """

    def add(x, y):
        return x + y

    print(reduce(add, range(101)))
    print(sum(range(101)))


def _filter():
    """
        filter(func,iterable) filter也接收一个函数和一个序列。
        和map()不同的时，filter()把传入的函数依次作用于每个元素，
        然后根据返回值是True还是False决定保留还是丢弃该元素。

    """

    def is_even(n):
        return n % 2 == 0

    print(list(filter(is_even, range(101))))


def _sorted():
    """
        排序函数，可以接收一个key函数来实现自定义的排序
    :return:
    """
    print(sorted([36, 5, -12, 9, -21]))
    print(sorted([36, 5, -12, 9, -21], key=abs))
    print(sorted([36, 5, -12, 9, -21], key=abs, reverse=True))


def _decorator():
    """
        装饰器:在代码运行期间动态增加功能的方式,本质上是一个返回函数的高阶函数
            Decorator：在不改变原本函数代码的情况下修改其他函数的功能的函数，他们有助于让我们的代码更简短，避免出现太多冗余代码.
            缺点：会丢失原函数的元信息
                解决方法：使用functools.traps(),其本身也是一个装饰器，可以把原函数的元信息拷贝到装饰器里面的func函数中
    """

    def use_logging(func):
        def wrapper():
            print("这是一个装饰器！")
            return func()  # 把 foo 当做参数传递进来时，执行func()就相当于执行foo()

        return wrapper

    # 赋值方式2（语法糖，放在要装饰的函数定义开始处，可以省略方式1中的赋值操作）
    @use_logging
    def foo():
        print('i am foo')

    # 赋值方式1
    foo = use_logging(foo)  # 因为装饰器 use_logging(foo) 返回的时函数对象 wrapper，这条语句相当于  foo = wrapper
    foo()  # 执行foo()就相当于执行 wrapper()


def _partial():
    """
        偏函数functools.partial的作用就是，把一个函数的某些参数给固定住（也就是设置默认值），返回一个新的函数，调用这个新函数会更简单。
    """
    int2 = functools.partial(int, base=2)
    print(int('10010'))
    print(int2('10010'))
