from fastai.basics import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc
rc('animation', html='jshtml')

# let us hvae 100 points
n = 100

x = torch.ones(n, 2)
# change the first column to uniform
x[:, 0].uniform_(-1., 1)

# inspect the first five of them
print('x')
print(x[:5])

# these are our parameter, slope/gradient and bias for y=ax+b
a = tensor(3., 2)


# y = a1x1 + a2x2 ----> a1 and a2 are the parameters of our model.
# a1 = slope and a2 = bias. x2 =1. x1 is the only feature we want.

# so generate n points with two columns x1 and x2, but keep x2 =1.

# a1x1 + a2x2 + a3x3 ---> X = [x1, x2, x3] is the feature vector.
# a1, a2 and a3 are the parameters, x3=1

# torch.randn(5) # generates 5 random numbers.

# @ is matrix multiplication in Python.
y = x@a + torch.randn(n)  # 100 random numbers are generated and added to x@a result
print('y')
print(y[:5])

# define MSE loss function: Mean Squared Error


def mse(y_hat, y):
    """mean squared error of y_hat """
    return ((y_hat -y)**2).mean()

# let us assume that we don't know a, i.e. that 3. and 2 are the parameters
# which create our line. We need to find slope 3. and y-intercept 2.

# we randomly guess slope and intercept


# print('y_hat')
#
# a = tensor(1., 1)
#
# y_hat = x@a
# print(y_hat[:5])
#
# print('loss')
# loss = mse(y_hat, y)
# print(loss)
#
#
# plt.scatter(x[:, 0], y)
# plt.scatter(x[:, 0], y_hat)
# plt.show()

# the model y_hat = x@a has been visualized and our loss function MSE has been specified.

# we must now optimize our parameters so that we find best fit.

# Enter Gradient Descent

a = tensor(-1., 1)

# convert a to a parameter
a = nn.Parameter(a)

# set learning rate

lr = 1e-1  # 0.1


def update():
    y_hat = x@a
    loss = mse(y_hat, y)
    if t%10 == 0:
        print(loss)
    loss.backward()
    with torch.no_grad():
        # subtract our parameters from the gradient of the loss function w.r.t parameters slope and intercept in a
        a.sub_(lr * a.grad)
        a.grad.zero_()  # zero out gradients so they don't accumulate


for t in range(100):
    update()


# plot again
plt.scatter(x[:, 0], y)
plt.plot(x[:, 0], x@a, c='orange')
plt.show()

# aimate this

a = nn.Parameter(-tensor(1., 1))

fig = plt.figure()
plt.scatter(x[:, 0], y, c='orange')
line, = plt.plot(x[:, 0], x@a)
plt.close()


def animate():
    update()
    line.set_ydata(x@a)
    return a


animation.FuncAnimation(fig, animate, np.arange(0, 100), interval=20)


