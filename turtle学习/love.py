from turtle import *


def curvemove():
    for i in range(200):
        right(1)
        forward(1)


pensize(3)
speed(2)

color('red', 'pink')
begin_fill()
left(140)
forward(111.65)
curvemove()
left(120)
curvemove()
forward(111.65)
end_fill()
penup()
goto(-40, 80)
pendown()
write("灵儿", font=('', 30, 'normal'))
hideturtle()


done()