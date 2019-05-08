import torch
import numpy as np
from torchvision import datasets
torch.manual_seed(3)
import argparse

# A bit of numpy, creating compicated miulti-dimensional arrays
print('be good at numpy')

p = np.array([[[1, 2], [2, 3],[2, 3]], [[1, 3], [4, 1], [7, 9]]])
s = np.array([[[1, 0, 9, 1], [1, 0, 2,3]], [[1, 0, 11, 3], [0, 1, 9, 10]], [[1, 2, 3, 1], [90, 91, 92, 93]]])
print(s[2, 1, 0])

print(s)
print(s.shape)
print(p)
print(p.shape)


# Constructing a tensor directly from data

print()
print('Constructing a tensor directly from data')
print()

# initialized it directly

v = torch.tensor([2, 3])  # directly passed in 2 and 3.3 here.


print(f'v: {v}')
print(f'v size: {v.size()}')
print(f'v shape: {v.shape}')
print(f'type of size and shape {type(v.size())}, {type(v.shape)}')
print()
print()

print('check if tensor, returns a boolean')
print(torch.is_tensor(v))

print('rank, axes, shape')

# all fundamentally connected to the concept of indices

# rank of a tensor is # of dims present in that tensor.

KL = torch.rand(5, 3, 8, 2)  # this tensor has rank 4.
print(KL)
print(KL[3, 2, 4, 1])
print('use tensor v to construct new tensors')

# axes of a tensor is a specific dimension of a tensor
print(KL.size(2))  # gives number of dimensions in the third axes of my tensor

# rank of a tensor == len(shape or size)

print(f'rank: {len(KL.size())}')
print(f'numel: {KL.numel()}')

print()
print('###########################')

v = v.new_ones(3, 3, dtype=torch.double)  # creates a 3 * 3 matrix of ones.
print(f'v.new_ones(3, 3) : {v}')  # dtype is now double.
print(f'size v : {v.size()}')
print(f'dtype : {v.dtype}')

print()
wow = torch.randn_like(v, dtype=torch.float)  # random numbers but same shape of v, dtype changes to float from double
# dtype has also changed
print(f'wow = torch.randn_like(v): {wow}')
print(f'wow size : {wow.size()}')
print(f'the type of size: {type(wow.size())}')
print(f'wow.size()[0] : {wow.size()[0]}')  # is a tuple, accessing the first index

print()
print('##############################################################')
print()

print(f'to create a scalar torch.tensor(2): {torch.tensor(2)}')
print(f'type: {type(torch.tensor(2))}')
print(f'size: {torch.tensor(2).size()}')
print()

print(f'torch.Tensor(3) : {torch.Tensor(3)}')
print(f'size : {torch.Tensor(3).size()}')  # size : torch.Size([])
print(f'shape : {torch.Tensor(3).shape}')  # shape : torch.Size([])
print(f'type : {type(torch.Tensor(3))}')

print()

print(f'torch.Tensor([3]) : {torch.Tensor([3])}')
print(f'size for [3]: {torch.Tensor([3]).size()}')  # shape : torch.Size([])
print(f'shape for [3] : {torch.Tensor([3]).shape}')  # shape : torch.Size([])
print(f'type : {type(torch.Tensor([3]))}')
print()

print('######################################')
print()

print('an uninitialized tensor with 2 rows and 3 columns')
print()

# using dtype=torch.long gives RunTimeError: _normal_ is not implemented for it.

e = torch.empty(2, 3)
print(f'e : {e}')

y = e.new_ones(4, 3, dtype=torch.double)  # constructing new_ones tensor from e and changing dtype

print(f'y = e.new_ones(4, 3): {y}')
print(f'dtype {y.dtype}')
print(f'device {y.device}')
print(f'layout {y.layout}')

y = torch.randn_like(y, dtype=torch.float)  # 4 by 3 shape shape but randon numbers
print(f'torch.randn_like(y): {y}')  # create randomized entries with the same shape as y.

print()
print('Comparison of list, numpy and tensor')
print()

# list, numpy, tensor
print(list(range(5)))
n = np.arange(5)
print(n)
# numpy --> torch
print(torch.from_numpy(n))
print(torch.as_tensor(n))  # also factory
print(torch.tensor(n))  # factory function
print(torch.Tensor(n))

# the class constructor uses the global default dtype which is float as oppose to the factory functions which infer
# the dtype, also called type inference.

# as_tensor and from_numpy share memory.
# so changing the numpy array will change the pytorch tensor and vice vicersa.

print(torch.get_default_dtype())

# Tensor Addition

print()
print('Tensor Addition')
print()

g = torch.rand(2, 3, dtype=torch.float)
print(f'g: {g}')
h = torch.rand(2, 3, dtype=torch.float)
print(f'h: {h}')
print(f'g + h: {g+h}')  # can add only if dimensions match.


print('torch.add(g + h)')
print(torch.add(g, h))  # another way
print('inplace operation addition: g.add_(h)') # g = g + h
print(g.add_(h))
print('sub(g + h)')
print(torch.sub(g, h))  # same as g-h and torch.sub_(g, h)---.inplace


print()
x = torch.empty(5, 7, dtype=torch.long)  # long, double, float, int
print(f'empty 5 by 7: {x}')

x = x.new_ones(5, 7, dtype=torch.double)
print(f'x: {x}')

d = torch.ones(5, 7, dtype=torch.double)
print(f'd: {d}')

print('check eqaulity between x and d using eq')
print()
print(torch.eq(x, d))  # returns 1 at each place of the tensnor if true
# or 0 if false
# print(x.item()) --> error  # only one element
# only one element tensors can be converted to Python scalars, torch([3]).item() --> 3

print()
print('reshaping your tensors, good for conv to full and full to conv')

# reshaping and resizing tensors
# use torch.view to resize your data, but this is not inplace.

A = torch.rand(3,  5, dtype=torch.double)
print()
print(f'A: {A}', f'Size of A: {A.size()}')
print()
# print(f'A.view(), without any parameters :
# {A.view()}') TypeError: view() missing 1 required positional arguments: "size"
print(f'A.view(15) : {A.view(15)}', f' : {A.view(15).size()}')  #  this is flattening the array
print()
print(f'Resized view with dim  5 by 3 A : {A.view(5, 3)}')
print()
print(f'using -1 with view :{A.view(-1, 15)}', f' Size of this thing : {A.view(-1, 15).size()}')  # this would be 1, 15
print()
print(f'using -1 with view', f'{A.view(-1, 5)}', f'the size {A.view(-1, 5).size()}')
# this sets the 1st dimension automatically, if the second dimension is specified as 5.
# The first will be 5 to satisfy 15. If A.view(-1, 3) ---> then 5 will be the first dimension.


print()
print('##################################################################')

print()
print('resize_: you resize it in place, the leading underscore signifies in-place')
d = torch.ones(3, 2)
print(f'd: {d}')
d.resize_(2, 3)
print('after resize_')
print()
print(f'd: {d}')

print()
print('calling item to get python object from tensor object')
print()
s = torch.rand([1])
print(f's: {s}')
print(f's.item() : {s.item()}')

print()
print('inplace add_ operation')
S = torch.rand(3, 5)
print(f'S: {S}')
print('add 5')
S.add_(5)
print('after adding 5')
print(f'S: {S}')

E = torch.ones(2, 2)
print(f'E : {E}')
E.add(5)  # this is not inplace, creates a new tensor,
# this new tesnor is 6.
print(f'E after E.add(5): {E}')
E.add_(5)  # this is inplace, changes tensor 6 to 11 by adding 5.
print(f'E after E.add_(5): {E.add_(5)}')

print()
print('Indexing and slicing')
print('##################################')
print()

# indexing and slicing
mytensor = torch.Tensor(3, 2)
print(mytensor)
print(mytensor[:, 1])

thr_d = torch.Tensor(3, 3, 2, 2)
print(thr_d)
print(thr_d[2, 1, :, 1])
print(thr_d.numel())

# Torch and Numpy share memory
# If you convert a numpy array to tensor or vice versa
# and perform an operation on one, the other element also changes.
# torch to numpy

print('torch and numpy conversions')
print()
print('torch to numpy')
print()


t = torch.ones(2, 2)
print(f't torch.ones(2, 2): {t}')

n = t.numpy()
print(f'n = t.numpy(): {n}')
print(f'type of n {type(n)}')

# n and t both point to the same memory location.
print('t.add_5')
t.add_(5)
print(f'n : {n}')  # 2 by 2 6
print(f't: {t}')  # 2 by 2 6

print(f'n + 2 : {n + 2}')
print(f't : {t}')  # t is still 6 though

# numpy to torch

print('numpy to torch')
print()

z = np.ones([3, 2])  # 3 by 2 numpy matrix of ones.
print(f'z numpy: {z}')
trch = torch.from_numpy(z)
print(f'trch = torch.from_numpy(z): {trch}')
print('add three to trch inplace')
trch.add_(3)
print(f'trch: {trch}')
print(f'z : {z}')
print()
print('now add three to numpy and see if torch changes')
z += 3
print(f'z += 3 : {z}')
print(f'trch : {trch}')

# change numpy, torch changes and vice versa

print()

# Autograd module
# Automatic differentiation

# torch.tensor is the fundamental structure on which we
# set the boolean flag requires_grad = True.
# This tracks all the operations on our tensor.
# After performing desired operations, which in the case of
# deep learning is performing the forward pass.
# we call tensor.backward() function which computes
# the gradient or derivative of our tensor.
# only for the leaf nodes though

# the gradeints which are computed
# are stored in
print()
print('#######################################################################################')
print()
print('AUTOGRAD PRACTICE')
print()

a = torch.tensor([2.0], requires_grad=True)
b = a ** 2
c = b + 2
d = c * 5
print(f'a: {a}')
print(f'b = a ** 2: {b}')
print(f'c = b + 2 = {c}')
print(f'a = 2.0; b = 4.0; c= 6.0; then 6.0 * 5 makes d = {d}')
print(f'grad_fn a: {a.grad_fn}, no grad_fn because a was defined by us')
print(f'grad_fn  b: {b.grad_fn}')
print(f'grad_fn c: {c.grad_fn}')
print(f'grad_fn d: {d.grad_fn}')

print('CALLING BACKWARD on d : d.backward(), note that d here is a scalar value')
d.backward()
print(f'a.grad : {a.grad}, became 20 from 2')
print(f'b.grad : {b.grad}')
print(f'c.grad : {c.grad}')
print(f'd.grad : {d.grad}')

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y.mean()  # add all values and divided by, if axis=0 mean is for the rows, if axis=1 it is for the columns.
print(f'x: {x}')
print(f'x.grad before calling backward : {x.grad}')
print(f'y = x + 2: {y}')
print(f'y.grad before calling backward : {y.grad}')
print(f'z: {z}')
print(f'z.grad before calling backward : {z.grad}')
print(f'calling backward on z: z.backward(), scalar value')

# z is being called on a scalar, as such the grad_variables arguments is torch.tensor([1]) by default.

z.backward()
print(f'x.grad.data: {x.grad}')
print(f'y.grad.data: {y.grad}')
print(f'z.grad.data : {z.grad}')

print()
print('calling backward on vector value now')
print()

l = torch.tensor([[2, 1]], dtype=torch.float, requires_grad=True)
print(f'l: {l}')
print(f'l.grad_fn: {l.grad_fn}')

M = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
print()
print(f'M: {M}')
print(f'M.grad_fn: {M.grad_fn}')
print()

p = torch.mm(l, M)
print(f'p = torch.mm(l, M): {p}')
print(f'p.grad_fn: {p.grad_fn}')

print()
print('calling backwards with various grad_variables')
print()

p.backward(torch.tensor([[1, 0]], dtype=torch.float), retain_graph=True)
print()
print(f'l.grad.data grad_variables =[1, 0]: {l.grad.data}')

l.grad.data.zero_()  # zero out the gradeints so that they don't accumulate.

p.backward(torch.tensor([[0, 1]], dtype=torch.float), retain_graph=True)
print()
print(f'l.grad.data grad_variables =[0, 1]: {l.grad.data}')

l.grad.data.zero_()
p.backward(torch.tensor([[1, 1]], dtype=torch.float), retain_graph=True)
print()
print(f'l.grad.data grad_variables = [1, 1] : {l.grad.data}')

# jacobian = torch.tensor(2, 2).zero_()
# print(jacobian)

print('Matrix multiplication in pytorch, this is what nn.Linear() does')
# does a torch.mm(x, weights initliazed by the nn.Linear() method and then adds a bias.)

d = torch.tensor([[3, 4, 7], [6, 10, 11]])
print(d)
print(d.size())

q = torch.tensor([[5, 7], [6, 10], [11, 15]])
print(q)
print(q.size())

E = torch.mm(d, q)
print(E)

print(torch.mm(q, d))

print('#####################################')

# Building Neural Networks with Pytorch

x = torch.ones(3, requires_grad=True)
print(f'x : {x}')

y = x * 2
while y.data.norm() < 1000:  # torch.data.norm() computes the l-2 or l-1 norm.
    y = y * 2

print(f'y : {y}')

# pass in v in order to find x.grad.
v = torch.tensor([2.0, 1.0, 1.0], dtype=torch.float)  # we passed in these vector because y is not scalar.
print(f'v : {v}')
# y isn't a scalar, else pytorch sets grad_varaibles param to torch.tensor([1]) b default.
y.backward(v)

print(f'x.grad : {x.grad}')

print('###################################################')
print('checking for device')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Create tensor and pass in device as parameter

s = torch.ones(2, 3, dtype=torch.float)

print(f's:{s}')

print(f'device : {s.device}')

s.to(device)

print(s.device)  # cpu, cause I don't have cuda.
print()
print('########################################')
print()

print('Autograd and Dynamic Computation Graphs')
print()

# Define Leaf Nodes of Dynamic Computation Graph, i.e. Nodes with no children.
a = torch.tensor([4], dtype=torch.float, requires_grad=True)
print(f'a: {a}')  # Variable.FloatTensor([4], requires_grad = True) v <= 0.3

# define 4 weights for our graph
weights = [torch.tensor([i], dtype=torch.float, requires_grad=True) for i in (2, 5, 9, 7)]


# unpack these weights
w1, w2, w3, w4,  = weights
print(f'w1:{w1}', f'w2:{w2}', f'w3:{w3}', f'w4:{w4}')
print(f'w1 grad_fn: {w1.grad_fn}')

# following is the forward pass of our model.
# inputs * weights + bias and then activation. All of these
# operations are performed when we call .forward() on our model.
# but here we are doing it separately to get a clear idea of what's going on.

b = w1 * a   # pytorch has already started to build our graph. ON THE FLY.
print(f'b : {b}')   # has grad_fn=<MulBackward0>

c = w2 * a
print(f'c: {c}')

d = w3 * b + w4 * c  # 42 + 135 =>
print(f'd: {d}')  # d is y_hat, i.e. the prediction of our model

L = (10 - d)  # L here is Loss, thus 10 is the correct value we are trying predict.
print(f'L : {L}')
print(f'L.grad_fn : {L.grad_fn}')  # grad_fn=<RsubBackward1>

L.backward()  # take derivative of L w.r.t each node(i.e. d, c, b)
# but also the weights w3, w4, w2 and w1 that d, b and c depend on.
# here d, c, b change as w1, w2, w3, w4 change.

for index, weight in enumerate(weights, start=1):
    # print(weight.grad.data)
    gradient, *_ = weight.grad.data
    print(f"Gradient of w{index} w.r.t to L: {gradient}")


print()
print('###################################################')
print()

a = torch.tensor([2], dtype=torch.float, requires_grad=True)
print(f'a: {a}')

b = torch.tensor([1], dtype=torch.float, requires_grad=True)
print(f'b : {b}')

c = a + b
print(f'c: {c}')
# c.retain_graph() # stores the gradient for this non-leaf node

d = b + 1
print(f'd : {d}')
# d.retain_graph()

e = c * d
print(f'e : {e}')
# e.retain_graph()

e.backward()


print(a.grad)  # de/da --> this is computing how e changes as b does
print(b.grad)  # de/ab --> this is computing how b changes as e does

# Here and and b are the leaf nodes of our system.

print('#################################')
print('sum product, flaten, reshape and brodcasting on tensors. ')
# flatten, reshape

a = torch.tensor([
                  [1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]
                  ])

print('add 5 to a')
print(a + 5)  # create a 3*3 tensor of 5's. adds that to a.


b = torch.tensor([2, 2, 2])
print(f'this is b: {b}')

print('add a and b')

print(a + b)  # brodacasts b to match shape (3,3) for each element of our tensor.

print('define c')
c = torch.tensor([[3, 3, 3]])
print(c)
print(c.size())

print('add a and c')
print(a + c)

# scalar value tensor is being brodcasted to the shape of our tensor

print('broadcast any scalar to take the shape of a tensor')

d = np.broadcast_to(5, a.shape)
print(d)

print('comparison with broacasting')
print(a.ge(0))  # 0 here is being broadcasted.
print(a.le(5))  # 5 is being broadcasted. Can also do <= or >=

ada = torch.tensor([
                  [1, 3, 1],
                  [1, 0, 5],
                  [7, 1, 1]]
)

print(f'ada : {ada}')

print('check for equality')

print(a.eq(ada))
print('')
print(a.eq(ada).sum().item())
# the above is what is done in a loop while testing Neural Network accuracy.

print('########################################################################')

print('Argmax and getting max and min along dimensions')

# ArgMax and reduction operations

WQ = torch.tensor([1, 2, 4, 3])
print(WQ.max())

print('sum of product and value in WQ: 1, 2, 4, 3')
print(f'WQ.sum(): {WQ.sum()}')
print(f'WQ.prod(): {WQ.prod()}')

print('this is tensor R')
R = torch.tensor([[1, 0, 0, 2],
                  [0, 3, 3, 0],
                  [4, 2, 0, 5]])

print('indexing for R')
print(R[2][3])
print()
print('max')
print(R.argmax())  # returns index location of max value of the flattened tensor.
print('min')
print(R.argmin())

print('R.max() for dimesnion 0 and then dimension 1')
print(R.max(dim=0))
print(R.max(dim=1))

print('flattening R into a row tensor')
e = R.flatten()
print(f'e: {e}')
print('e.max(dim=0)')
print(e.max(dim=0))  # only dimension zero makes sense here

print('concatenation')
foo = torch.ones(2, 2)
bar = torch.empty(2, 2).fill_(3)

print('foo')
print(foo)
print('bar')
print(bar)

print('concatenation')

FooBar_0 = torch.cat((foo, bar), dim=0)  # along the row axis. verticaly concatenates
print(FooBar_0)

FooBar_1 = torch.cat((foo, bar), dim=1)  # along the row axis. horizontally concatenates
print(FooBar_1)

print('Matrix Multiplication, four ways: @ operator,  ')

s = torch.randn(2, 3).fill_(1)
f = torch.randn(3, 5).fill_(7)

print(s)
print(f)

pol = s @ f  # @ means matrix multiplication

print(pol)

print(pol.size())

prt = torch.mm(s, f)
print(prt)
print(prt.size())

qs = torch.nn.Linear(2, 3)

a = torch.rand(5, 2)
print(qs.forward(a))


