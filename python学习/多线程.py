"""
    一个进程可以包含多个线程，并且最少要有一个线程，线程是最小的执行单元
    进程是打开notepad，线程是打字，拼音检查，打印·····
    实现多任务的方式：
        1.多进程模式
        2.多线程模式
        3.多进程+多线程

    线程对象:
        守护线程:当一个线程被标记为守护线程时，Python程序会在剩下的线程都是守护线程时退出，即等待所有非守护线程运行完毕；
                守护线程在程序关闭时会突然关闭，可能会导致资源不能被正确释放的的问题
        非守护线程:通常我们创建的线程默认就是非守护线程，Python程序退出时，如果还有非守护线程在运行，
                程序会等待所有非守护线程运行完毕才会退出。
    创建线程对象:
        threading.Thread(group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None)
            group：通常默认即可，作为日后扩展ThreadGroup类实现而保留。
            target：用于run()方法调用的可调用对象，默认为 None。
            name：线程名称，默认是Thread-N格式构成的唯一名称，其中N是十进制数。
            args：用于调用目标函数的参数元组，默认为 ()。
            kwargs：用于调用目标函数的关键字参数字典，默认为 {}。
            daemon：设置线程是否为守护模式，默认为 None。
    锁对象:
        同一变量在多线之间是共享的，任何一个变量都可以被所有线程修改，当多个线程一起修改同一变量时，
        很可能互相冲突得不到正确的结果，造成线程安全问题。
        threading.Lock: 互斥锁,一旦一个线程获得一个锁，会阻塞随后尝试获得锁的线程，直到它被释放，通常称其为互斥锁
        threading.RLock: 可重入锁,一旦线程获得了重入锁，同一个线程再次获取它将不阻塞，重入锁必须由获取它的线程释放。
    条件对象:
        条件对象总是与某种类型的锁对象相关联，锁对象可以通过传入获得，或者在缺省的情况下自动创建。
        使用条件对象的典型场景是将锁用于同步某些共享状态的权限，
        那些关注某些特定状态改变的线程重复调用 wait() 方法，
        直到所期望的改变发生；对于修改状态的线程，
        它们将当前状态改变为可能是等待者所期待的新状态后，调用 notify() 方法或者 notify_all() 方法。
        threading.Condition(lock=None)
    信号量对象:
        和锁机制一样，信号量机制也是一种实现线程同步的机制，不过它比锁多了一个计数器，这个计数器主要用来计算当前剩余的锁的数量。
        threading.Semaphore(value=1)    可选参数 value 赋予内部计数器初始值，默认值为 1
    事件对象:
        一个线程发出事件信号，其他线程等待该信号，这是最简单的线程之间通信机制之一。
        threading.Event()
"""

from threading import *
import time

# threading模块的常见类方法和属性。
# print(enumerate())  # 以列表形式返回当前所有存活的 threading.Thread对象。
# print(active_count())  # 返回当前存活的 threading.Thread对象个数，等于len(threading.enumerate())。
# print(current_thread())  # 返回当前对应调用者控制的 threading.Thread 对象，如果调用者的控制线程不是利用 threading 创建，则会返回一个功能受限的虚拟线程对象。
# print(get_ident())  # 返回当前线程的线程标识符，它是一个非零的整数，其值没有直接含义，它可能会在线程退出，新线程创建时被复用。
# print(main_thread())  # 返回主线程对象，一般情况下，主线程是 Python 解释器开始时创建的线程。
# print(stack_size(32768))  # 返回创建线程时用的堆栈大小，可选参数 size 指定之后新建线程的堆栈大小，size 值需要为 0 或者最小是 32768（32KiB）的一个正整数，如不指定 size，则默认为 0。
# print(get_native_id())  # 返回内核分配给当前线程的原生集成线程 ID，其值是一个非负整数。
# TIMEOUT_MAX = 100  # 指定阻塞函数（如：Lock.acquire()， Condition.wait() ...）中形参 timeout 允许的最大值，传入超过这个值的 timeout 会抛出 OverflowError 异常。


# 线程对象Thread的常见方法和属性def get_directories():
#     dirs.append(tkfilebrowser.askopendirnames())
#     return dirs
#
# b1 = Button(root, text='select directories...', command=get_directories)
def func(tim: int, name: str):
    lock = Lock()
    lock.acquire()
    time.sleep(tim)
    print(1, 2, 3, name)
    time.sleep(tim)
    print(4, 5, 6, name)
    time.sleep(tim)
    print(7, 8, 9, name)
    lock.release()


# t = Thread(target=func, args=(3,))
# t.start()  # 启动线程
# # t.run() # 线程执行具体功能的方法
# t.join(2)  # 当 timeout 为 None 时，会等待至线程结束；当timeout不为None时，会等待至timeout时间结束，单位为秒。
# print(t.is_alive())  # 判断当前线程是否存活
# print(t.getName())  # 返回线程名
# t.setName("Thread-2")  # 设置线程名
# print(t.getName())
# print(t.isDaemon())  # 判断线程是否是守护线程
# # t.setDaemon(False)   # 设置线程是否是守护线程
# print(t.name)  # 线程名字
# print(t.ident)  # 线程标识符
# print(t.daemon)  # 线程是否为守护线程

# 创建线程对象方式一----实例化Thread
t1 = Thread(name='t1', target=func, args=(1, "t1"))
t2 = Thread(name='t2', target=func, args=(2, "t2"))
t3 = Thread(name='t3', target=func, args=(3, "t3"))
t1.start()
t2.start()
t3.start()
print("主线程结束!")


# 创建线程对象方式二----继承Thread
# class MyThread(Thread):
#     def __init__(self, sleep, name):
#         super().__init__()
#         self.sleep = sleep
#         self.name = name
#
#     def run(self):
#         time.sleep(self.sleep)
#         print('name：' + self.name)
#
#
# t3 = MyThread(3, 't3')
# t4 = MyThread(4, 't4')
# t3.start()
# t4.start()

# # 互斥锁
# lock = Lock()
# lock.acquire(blocking=True, timeout=-1)  # 可以阻塞或非阻塞地获得锁，参数 blocking 用来设置是否阻塞，
# # timeout 用来设置阻塞时间，当 blocking 为 False 时 timeout 将被忽略。
# lock.release()  # 释放锁
# print(lock.locked())  # 判断是否获得了锁
#
# # 可重入锁
# rlock = RLock()
# rlock.acquire()
# rlock.release()
#
# # 条件对象
# c = Condition()
# c.acquire()  # 请求底层锁
# c.release()  # 释放底层锁
# c.wait(timeout=20)  # 等待直到被通知或超时
# c.wait_for(predicate=func)  # 等待直到条件计算为 True，predicate 是一个可调用对象且它的返回值可被解释为一个布尔值。
# c.notify(n=1)  # 默认唤醒一个等待该条件的线程
# c.notifyAll()  # 唤醒所有正在等待该条件的线程。
#
# # 信号量对象
# s = Semaphore(10)
# s.acquire()
# s.release()
#
# # 事件对象
# event = Event()
# event.is_set()  # 当内部标志为 True 时返回 True。
# event.set()  # 将内部标志设置为True
# event.clear()  # 将内部标志设置为False
# event.wait()  # 阻塞线程,知道内部标志变为True
