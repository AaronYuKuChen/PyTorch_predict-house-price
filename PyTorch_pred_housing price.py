import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = Variable(torch.linspace(0, 100).type(torch.FloatTensor))
rand = Variable(torch.randn(100)) * 10
# print('x: ', x)
# print('rand: ',rand)
y = x + rand
# print('y: ', y)
x_train = x[: -10]
x_test = x[-10:]
y_train = y[:-10]
y_test = y[-10:]

plt.figure(figsize=(10, 8))

plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'o')
plt.xlabel('X')
plt.ylabel('Y')
# plt.show()
print("=========train============")

a = Variable(torch.rand(1), requires_grad=True)
b = Variable(torch.rand(1), requires_grad=True)

print('a:  ',a)
print('b:  ',b)

learning_rate = 0.0001
# 訓練1000次
for i in range(1000):
    predictions = a.expand_as(x_train) * x_train + b.expand_as(x_train)
    loss = torch.mean((predictions - y_train) ** 2)
    #print("Iterations: ", i+1, " and their loss: ", loss)
    loss.backward()
    a.data.add_(- learning_rate * a.grad.data)
    b.data.add_(- learning_rate * b.grad.data)
    a.grad.data.zero_()
    b.grad.data.zero_()

x_data = x_train.data.numpy()

plt.figure(figsize=(10,7))
xplot, = plt.plot(x_data, y_train.data.numpy(), 'o')
yplot, = plt.plot(x_data, a.data.numpy() * x_data + b.data.numpy())

plt.xlabel('X')
plt.ylabel('Y')

str1 = str(a.data.numpy()[0]) + ' x + ' + str(b.data.numpy()[0])

plt.legend([xplot, yplot], ['Data', 'str1'])
# plt.show()

print("=========test============")

predictions = a.expand_as(x_test) *x_test + b.expand_as(x_test)

print("predictions: ",predictions)

x_data = x_train.data.numpy()
x_pred = x_test.data.numpy()
plt.figure(figsize=(10,7))
plt.plot(x_data, y_train.data.numpy(), 'o')
plt.plot(x_pred, y_test.data.numpy(), 's')
x_data = np.r_[x_data, x_test.data.numpy()]

plt.plot(x_data, a.data.numpy() * x_data + x_data +b.data.numpy())
plt.plot(x_pred, a.data.numpy() * x_pred + b.data.numpy(), 'o')

plt.xlabel('X')
plt.ylabel('Y')

str1 = str(a.data.numpy()[0]) + ' x + ' + str(b.data.numpy()[0])
plt.legend([xplot, yplot], ['Data', str1])
plt.show()