from multiprocessing import *
import time, os

"""
    Python 提供了多进程模块 multiprocessing，该模块同时提供了本地和远程并发，
    使用子进程代替线程，可以有效的避免 GIL 带来的影响，能够充分发挥机器上的多核优势，可以实现真正的并行效果
    multiprocessing 模块通过创建一个 Process 对象然后调用它的 start() 方法来生成进程，Process 与 threading.Thread API 相同。
    多进程对象:
        multiprocessing.Process(group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None)
            group：仅用于兼容 threading.Thread，应该始终是 None。
            target：由 run() 方法调用的可调用对象。
            name：进程名。
            args：目标调用的参数元组。
            kwargs：目标调用的关键字参数字典。
            daemon：设置进程是否为守护进程，如果是默认值 None，则该标志将从创建的进程继承。
    
    进程池对象:
        当进程数量比较多时，我们可以利用进程池方便、高效的对进程进行使用和管理。
        Pool类可以提供指定数量的进程供用户调用，当有新的请求提交到Pool中时，如果进程池还没有满，就会创建一个新的进程来执行请求。
        如果池满，请求就会告知先等待，直到池中有进程结束，才会创建新的进程来执行这些请求。
        multiprocessing.pool.Pool([processes[, initializer[, initargs[, maxtasksperchild[, context]]]]])
            processes：工作进程数目，如果 processes 为 None，则使用 os.cpu_count() 返回的值。
            initializer：如果 initializer 不为 None，则每个工作进程将会在启动时调用 initializer(*initargs)。
            maxtasksperchild：一个工作进程在它退出或被一个新的工作进程代替之前能完成的任务数量，为了释放未使用的资源。
        有如下两种方式向进程池提交任务：
            apply(func[, args[, kwds]])：阻塞方式。
            apply_async(func[, args[, kwds[, callback[, error_callback]]]])：非阻塞方式。
    
    进程间的通信方式:
        管道:
            返回一对 Connection 对象  (conn1, conn2) ， 分别表示管道的两端；
            如果 duplex 被置为 True (默认值)，那么该管道是双向的，否则管道是单向的。
            multiprocessing.Pipe([duplex])     
        队列:
            multiprocessing.Queue([maxsize])   
    
    进程间同步:
        多进程之间不共享数据，但共享同一套文件系统，像访问同一个文件、同一终端打印，如果不进行同步操作，就会出现错乱的现象。
        所有在 threading 存在的同步方式，multiprocessing 中都有类似的等价物，如：锁、信号量等。
        加锁!
    
    进程间共享状态:
        共享内存:
            multiprocessing.Value(typecode_or_type, *args, lock=True)
                返回一个从共享内存上创建的对象。参数说明如下:
                    typecode_or_type：返回的对象类型。
                    *args：传给类的构造函数。
                    lock：如果 lock 值是 True（默认值），将会新建一个递归锁用于同步此值的访问操作；
                          如果 lock 值是 Lock、RLock 对象，那么这个传入的锁将会用于同步这个值的访问操作；
                          如果 lock 是 False，那么对这个对象的访问将没有锁保护，也就是说这个变量不是进程安全的。
            multiprocessing.Array(typecode_or_type, size_or_initializer, *, lock=True)
                从共享内存中申请并返回一个数组对象。
                    typecode_or_type：返回的数组中的元素类型。
                    size_or_initializer：如果参数值是一个整数，则会当做数组的长度；否则参数会被当成一个序列用于初始化数组中的每一个元素，
                                         并且会根据元素个数自动判断数组的长度。
                    lock：说明同上。
        服务进程:
            由 Manager() 返回的管理器对象控制一个服务进程，该进程保存 Python 对象并允许其他进程使用代理操作它们。
            Manager() 返回的管理器支持类型包括：list、dict、Namespace、
                                                Lock、RLock、Semaphore、BoundedSemaphore、
                                                Condition、Event、Barrier、Queue、Value 和 Array。
"""


def target(tim: int):
    time.sleep(tim)
    print('子进程ID：', os.getpid())


# 进程对象的常见方法和属性
p = Process(target=target, args=(2,))
# p.run()   # 进程具体执行的方法
p.start()  # 启动进程
p.join(timeout=-1)  # 果可选参数 timeout 是默认值 None，则将阻塞至调用 join() 方法的进程终止；
# 如果 timeout 是一个正数，则最多会阻塞 timeout 秒。
print(p.name)  # 进程的名称
print(p.is_alive())  # 判断进程是否活着
print(p.daemon)  # 是否是守护进程
print(p.pid)  # 进程的id
print(p.exitcode)  # 子进程的退出代码
print(p.authkey)  # 进程的身份验证密钥
print(p.sentinel)  # 系统对象的数字句柄，当进程结束时将变为 ready。
p.terminate()  # 终止进程
p.kill()  # 同上
p.close()  # 关闭 Process 对象，释放与之关联的所有资源。

# 进程池对象
pool = Pool(processes=8)
pool.apply(target, (2,))  # 阻塞式
pool.apply_async(target, (2,))  # 非阻塞式


# 进程间的通信方式----管道
def setData(conn, data):
    conn.send(data)


def printData(conn):
    print(conn.recv())


con1, con2 = Pipe()
p1 = Process(target=setData, args=(con1, "程序之间",))
p2 = Process(target=printData, args=(con2,))
p1.start()
p2.start()
p1.join()
p2.join()


# 进程间的通信方式----队列
q = Queue()
print(q.qsize())    # 返回队列的大致长度
print(q.empty())    # 判断队列是否为空
print(q.full()) # 判断队列是否为满
q.put(obj=123) # 将obj入队列
q.put_nowait(123)  # 相当于 put(obj, False)。
q.get() # 从队列中取出并返回对象。
q.get_nowait()  # 相当于 get(False)
q.close()   # 指示当前进程将不会再往队列中放入对象。
q.join_thread() # 等待后台线程。
q.cancel_join_thread()  # 防止进程退出时自动等待后台线程退出。

