from turtle import *

# 设置屏幕
screensize(800, 600, bg='yellow')   # 设置画布大小
# setup(0.5, 0.5)      # 前两个参数为大小(小数代表比例)，后两个代表左上角顶点的位置，不加的话则在中间

# 设置画笔属性
pensize(2)      # 画笔的宽度
pencolor('red') # 画笔的颜色
speed(1)        # 画笔的速度

# 绘图命令————运动命令
forward(100)            # 向当前画笔方向移动（单位像素）
backward(100)           # 向当前画笔相反方向移动
right(180)              # 顺时针移动画笔(单位度数)
left(180)               # 逆时针移动画笔
setheading(30)          # 设置当前朝向角度
pendown()               # 移动时绘制图形，缺省时也绘制
# penup()                 # 提起画笔，不绘制图形
goto(100, 100)          # 将画笔移动到某个位置
circle(10, 180)         # 画圆,半径为正,在画笔左边画.
circle(100, steps=4)    # steps (optional) (做半径为radius的圆的内切正多边形，多边形边数为steps)。
setx(200)               # 将当前x轴移动到指定位置
sety(200)               # 将当前y轴移动到指定位置
home()                  # 画笔的位置回到原点，朝向东
dot(10, 'green')        # 画一个指定直径和颜色的圆点

# 绘图命令————画笔控制
fillcolor('black')      # 图形填充颜色
color()                 # 同时填充画笔和填充颜色
filling()               # 返回是否在填充状态
begin_fill()            # 准备开始填充
end_fill()              # 填充完成
hideturtle()            # 隐藏画笔状态（图标）
showturtle()            # 显示画笔状态（图标）

# 绘图命令————全局控制
clear()                 # 清空窗口，但是画笔的位置和状态不会变
reset()                 # 清空窗口，画笔的位置和状态会变
undo()                  # 撤销上一个画笔动作
isvisible()             # 返回画笔状态是否可见
stamp()                 # 复制当前图形
write("月灵儿", font = ("", 20, ""))          # 写文本

# 绘图命令————其他
# mainloop()              # 启动事件循环
# done()                  # 必须是最后一句语句
mode()                    # 设置乌龟的模式
delay(500)                # 单位毫秒的绘图延迟
begin_poly()              # 开始记录多边形的顶点
end_poly()                # 停止记录多边形的顶点
get_poly()                # 返回最后记录的多边形

mainloop()