import re
from collections import deque   # 队列
import heapq    # 堆
from collections import Counter


# 1.将序列拆分成单独的变量
def sequence_to_variable():
    """
        任何序列（或可迭代的对象）都可以通过一个简单的赋值操作来分解为单独的变量。
    唯一的要求是变量的总数和结构要与序列相吻合。
    :return:
    """
    a = (4, 5)
    data = ['ACME', 50, 91.1, (2012, 12, 21)]
    s = "abc"
    x, y = a
    name, shares, price, date = data
    b, c, d = s
    print(x, y)
    print(name, shares)
    print(b, c, d)


# 2.分解元素
def decomposition_element():
    def drop_first_last(grades):
        first, *middle, last = grades
        return sum(middle)/len(middle)


# 3.队列
def _deque():
    """
        从队列两端添加或弹出元素的复杂度都是 O(1)。这和列表不同，当从列表的头部插入
    或移除元素时，列表的复杂度为 O(N)。
    :return:
    """
    q = deque(maxlen=5) # 长度为5的队列
    p = deque() # 无限长的队列
    q.append(1) # 从尾部进队
    q.append(2)
    q.append(3)
    q.append(4) # 超过长度就把最早进去的弹出
    q.appendleft(5) # 从头部进队
    q.pop() # 从尾部出队,返回出队的元素
    q.popleft() # 从头部出队,返回出队的元素


# 4.使用堆来寻找某个序列中最大或最小的几个元素
def _heap():
    """
        堆最重要的特性就是 heap[0]总是最小那个的元素。此外，接下来的元素可依次通过
    heapq.heappop()方法轻松找到。
    :return:
    """
    nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]

    # 方式1(适用于总样本个数不那么大)
    heapq.nlargest(3, nums, key=None)   # 最大的几个数
    heapq.nsmallest(3, nums, key=None)   # 最小的几个数

    # 方式2(适用于总样本个数很大)
    heap = list(nums)
    heapq.heapify(heap) # 建立起堆
    heapq.heappop(heap) # 该方法会将第一个元素（最小的）弹出，然后以第二小的元素取而代之（这个操作的复杂度是 O(logN)，N 代表堆的大小）。


# 5.从序列中移除重复项且保持元素间顺序不变
def _remove_elements():
    a = [1, 5, 2, 1, 9, 1, 5, 10]
    print(a)
    print(set(a))   # 简单的使用集合会打乱元素的原始顺序

    def dedupe(items, key=None):
        seen = set()
        for item in items:
            val = item if key is None else key(item)
            if val not in seen:
                yield val
                seen.add(val)

    print(list(dedupe(a)))


# 6.找出序列中出现次数最多的元素
def _frequent_element():
    """
        collections 模块中的 Counter 类正是为此类问题所设计的。它甚至有一个非常方便的
    most_common()方法可以直接告诉我们答案。
    :return:
    """
    words = [
        'look', 'into', 'my', 'eyes', 'look', 'into', 'my', 'eyes',
        'the', 'eyes', 'the', 'eyes', 'the', 'eyes', 'not', 'around', 'the',
        'eyes', "don't", 'look', 'around', 'the', 'eyes', 'look', 'into',
        'my', 'eyes', "you're", 'under'
    ]
    word = Counter(words)
    print(word) # 在底层实现中，Counter 是一个字典，在元素和它们出现的次数间做了映射。
    print(word.most_common(3))  # 参数指定个数,返回一个列表,列表元素的包含值和索引的元组


# 7.针对任意多的字符拆分字符串
def split_str():
    """

    :return:
    """
    line = 'asdf fjdk; afed, fjek,asdf, foo'
    print(re.split(r'[;,\s]\s*', line))

