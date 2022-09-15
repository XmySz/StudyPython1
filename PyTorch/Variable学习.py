import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2], [3,4]])
variable= Variable(tensor, requires_grad=True)

t_out = torch.mean(tensor**2)
v_out = torch.mean(variable**2)

v_out.backward()    # 反向传递
print(variable.grad)

print(variable.data)
print(variable.data.numpy())