from types import MethodType
from enum import Enum

"""
    类
        封装 继承 多态
"""


# 定义一个类 类名首字母一般大写
class Student(object):  # 括号内代表继承的父类
    """pass"""  # 第一行放对类功能的描述


# 创建类的实例
s = Student()
print(s)
print(Student)


# 在类里强制绑定属性
class Student1(object):
    def __init__(self, name, score):  # self参数必须要有,代表实例对象本身,传入参数时可以忽略它
        self.name = name  # self开头的变量可供类中所有方法和类的所有实例使用
        self.score = score
        self.type = 'human'

    def sleep(self):
        print(f'{self.name}正在睡觉, 请不要打扰！')

    def study(self):
        print(f'{self.name}正在学习, 请不要打扰！')

    def update_score(self, score):  # 方式2
        self.score = score


# 更改类的属性值
s2 = Student1('上条', 59)
s2.score = 60  # 方式1
s2.update_score(61)  # 方式2


def class_data_wrap():
    """
        类的数据封装特性
    :return:
    """

    class Student2(object):

        def __init__(self, name, score):
            self.name = name
            self.score = score

        def print_score(self):
            print('%s: %s' % (self.name, self.score))

    s2 = Student2('赵雨', 100)
    print(s2.print_score())


def private_variable():
    """
        使得类内部定义的变量无法从外部直接访问到
        __name:私有变量，不允许外部直接访问
        __name__:特殊变量，可以从外部直接访问，但是不能随便命名
        _name:“虽然我可以被访问，但是，请把我视为私有变量，不要随意访问”
    """

    class Student3(object):
        sex = '女'  # 类属性

        def __init__(self, name, score):
            self.__name = name
            self.__score = score

        def getName(self): return self.__name

        def getScore(self): return self.__score

        def setName(self, name): self.__name = name

        def setScore(self, score): self.__score = score

        def print_score(self):
            print('%s: %s' % (self.__name, self.__score))

    s3 = Student3('赵宇', 99)
    s3.salary = 3000  # 实例属性
    print(s3.salary)
    # print(s3.__name)      # 会失败
    print(s3.getName())
    # s3.__name='叶叶'      # 会失败
    s3.setName('下雨了')
    print(s3.getName())


def class_inherit():
    """
       创建子类时，必须和父类在同一个文件，并且要在父类后面
    """

    class Animal(object):
        def run(self):
            print('Animal is running...')

    class Dog(Animal):
        def run(self):  # 覆盖父类的同名方法
            print('Dog is running...')

        def eat(self):
            print('Eating meat...')

    class Cat(Animal):
        def run(self):
            print('Cat is running...')

    dog = Dog()
    cat = Cat()
    dog.run()
    cat.run()


def class_slots():
    """
        如果我们想要限制实例的属性怎么办？比如，只允许对Student实例添加name和age属性。
        为了达到限制的目的，Python允许在定义class的时候，
        定义一个特殊的slots变量，来限制该class实例能添加的属性
    """

    class Student(object):
        __slots__ = ('name', 'age')  # 用tuple定义允许绑定的属性名称，仅对当前类有用，对继承的子类无用
        pass

    s = Student()

    s.name = "zyn"  # 动态给实例绑定属性，对别的实例不起作用
    s.salaxy = 3000  # 会失败

    def set_age(self, age):
        self.age = age

    s.set_age = MethodType(set_age, s)  # 给实例绑定方法,对别的实例不起作用
    s.set_age(25)

    def set_score(self, score):
        self.score = score

    Student.set_score = MethodType(set_score, Student)  # 给类直接绑定方法，所有实例都可以使用


def class_property():
    """
        使用@property装饰器来既可以检查参数，又可以用类似属性的方式访问和修改类的变量（把方法变成属性）
    """

    class Student4(object):
        @property
        def score(self):
            return self._score

        @score.setter
        def score(self, value):
            if not isinstance(value, int):
                raise ValueError('score must be an integer!')
            if value < 0 or value > 100:
                raise ValueError('score must between 0 ~ 100!')
            self._score = value

    s4 = Student4()
    s4.score = 96  # 注意到@property,我们在对实例属性操作的时候，就知道该属性很可能不是直接暴露的，而是通过getter和setter方法来实现的。
    print(s4.score)

    class Student(object):
        @property
        def birth(self):
            return self._birth

        @birth.setter
        def birth(self, value):
            self._birth = value

        @property               # 可以定义只读属性，只定义getter方法，不定义setter方法就是一个只读属性
        def age(self):
            return 2015 - self._birth


def class_Multiple_inheritance():
    """
        python 支持多重继承 Mixln
    :return:
    """


def _enum():
    """
        使用枚举类
    :return:
    """
    Month = Enum('Month', ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
    for name, member in Month.__members__.items():
        print(name, '=>', member, ',', member.value)

_enum()