import calendar  # 日历相关函数
import datetime  # 基本日期和时间类型,扩展了time模块
import time  # 时间的访问和转换


def _time():
    """
    struct_time类代表时间对象,可以通过索引和属性名访问值
    索引  属性       值
    0    tm_year    年
    1    tm_mon     月
    2    tm_mday    日
    3    tm_hour    时
    4    tm_min     分
    5    tm_sec     秒
    6    tm_wday    这个月的第几周
    7    tm_yday    一年内的第几天
    8    tm_isdst   夏令时(-1, 0, 1)
    epoch = 1970年1月1日8:00
    :return:
    """
    t = time.localtime()  # 当地时间,返回struct_time对象,可选参数secs
    t1 = time.localtime(1000)  # 从epoch经过secs后的struct_time对象
    t2 = time.time()  # 时间戳,从epoch累计经过的secs
    t3 = time.gmtime()  # 将时间戳转换为格林威治天文时间下的struct_time,可选参数secs表示从epoch到现在的秒数，默认为当前时间
    t4 = time.asctime(t)  # 接收一个 struct_time 表示的时间，返回形式为：Mon Dec 2 08:53:47 2019 的字符串
    t5 = time.ctime()  # 相当于asctime(localtime(secs))
    t6 = time.strftime("%Y-%m-%d %H:%M:%S", t)  # 格式化日期，接收一个 struct_time 表示的时间，并返回以可读字符串表示的当地时间
    # 格式化参考 https://mp.weixin.qq.com/s/fBGKrjGcBdC5ZXYFQfRF7A
    t7 = time.mktime(t)  # localtime()的反函数

    print(time.time())
    print(time.gmtime())
    print(time.localtime())
    print(time.asctime(time.localtime()))
    print(time.tzname)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))


def _datetime():
    """
    date类表示一个由年、月、日组成的日期，格式为：datetime.date(year, month, day)
    time类表示由时、分、秒、微秒组成的时间，格式为：time(hour=0, minute=0, second=0, microsecond=0, tzinfo=None, *, fold=0)。
    datetime包括了date与time的所有信息，格式为：datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None, *, fold=0)
    :return:
    """
    # date的类方法和属性
    print(datetime.date.today())  # 返回当前的日期 year-month-day
    print(datetime.date.fromtimestamp(time.time()))  # 根据给定的时间戮，返回本地日期
    print(datetime.date.max)  # date所能表示的最大日期
    print(datetime.date.min)  # date所能表示的最小日期

    # date的实例方法和属性
    td = datetime.date.today()
    print(td.replace(1970, 1, 1))  # 生成一个新的日期对象，用参数指定的年，月，日代替原有对象中的属性
    print(td.timetuple())  # 返回日期对应的 struct_time 对象
    print(td.weekday())  # 返回一个整数代表星期几，星期一为 0，星期天为 6
    print(td.isoweekday())  # 返回一个整数代表星期几，星期一为 1，星期天为 7
    print(td.isocalendar())  # 返回格式为 (year，month，day) 的元组
    print(td.isoformat())  # 返回格式如 YYYY-MM-DD 的字符串
    print(td.strftime("%Y--%M--%D"))  # 返回自定义格式的字符串
    print(td.year)
    print(td.month)
    print(td.day)

    # time类的实例方法和属性
    td = datetime.time(10, 10, 10)
    print(td.isoformat())  # 返回 HH:MM:SS 格式的字符串
    print(td.replace(hour=9, minute=9))  # 创建一个新的时间对象，用参数指定的时、分、秒、微秒代替原有对象中的属性
    print(td.strftime(""))  # 同上
    print(td.hour)
    print(td.minute)
    print(td.second)
    print(td.microsecond)
    print(td.tzinfo)  # 时区

    # datetime类的类方法和属性
    print(datetime.datetime.today())  # 返回当地的当前时间
    print(datetime.datetime.now())  # 类似于 today()，可选参数tz可指定时区
    print(datetime.datetime.utcnow())  # 返回当前 UTC 时间
    print(datetime.datetime.fromtimestamp(time.time()))  # 根据时间戳返回对应时间
    print(datetime.datetime.utcfromtimestamp(time.time()))  # 根据时间戳返回对应 UTC 时间
    print(datetime.datetime.combine(datetime.date(2019, 12, 1), datetime.time(10, 10, 10)))  # 根据 date 和 time 返回对应时间
    print(datetime.datetime.min)
    print(datetime.datetime.max)

    # datetime的实例方法和属性
    td = datetime.datetime.today()
    print(td.date())  # 返回具有同样 year,month,day 值的 date 对象
    print(td.time())  # 返回具有同样 hour, minute, second, microsecond 和 fold 值的 time 对象
    # 其他方法高度类似于上文

    x = datetime.datetime(2015, 1, 12, 10, 10, 10)
    y = datetime.datetime(2015, 1, 12, 10, 0, 0)
    print((x - y).days)  # 两个日期对象相差的天数
    print((x - y).seconds)  # 两个日期对象相差的秒数,只算时分秒的差
    print((x - y).total_seconds())  # 两个日期对象相差的总秒数
    print((x - y).microseconds)  # 两个日期对象相差的毫秒数


def _calendar():
    """
    Calendar类
    TextCalendar 为 Calendar子类，用来生成纯文本日历。

    :return:
    """
    # 获取某月的日历
    cal = calendar.month(2021, 10)
    print(cal)

    # 判断某年是否是闰年
    print(calendar.isleap(2021))

    # 返回在两年之间的闰年总数。
    print(calendar.leapdays(2000, 2021))

    c = calendar.Calendar()
    print(list(c.iterweekdays()))  # 返回一个迭代器，迭代器的内容为一星期的数字
    print(list(c.itermonthdays(2022, 5)))  # 返回一个迭代器，迭代器的内容为某年某月的日期

    tc = calendar.TextCalendar()
    print(tc.formatmonth(2019, 12))  # 返回一个多行字符串来表示指定年、月的日历
    print(tc.formatyear(2019))  # 返回一个m列日历，可选参数w,l,和c分别表示日期列数，周的行数，和月之间的间隔


_calendar()
