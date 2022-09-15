from turtle import *
from time import *

pensize(5)
pencolor("yellow")
fillcolor("violet")

begin_fill()
for _ in range(5):
    forward(200)
    right(144)
end_fill()
sleep(2)

penup()
goto(-150, -120)
color("violet")
write("Done", font=('Arial', 40, 'normal'))

mainloop()