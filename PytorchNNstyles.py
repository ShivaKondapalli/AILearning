import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import string
import json
import torch
import unicodedata
from collections import OrderedDict
import matplotlib.pyplot as plt
import string
import os
import glob
torch.manual_seed(3)


print('With Module list for hidden layers')
# using ModuleList so that one can pass on any number of hidden layers
# [250, 500, 100]
# if we want more that one hidden layer, one has to use a list,
# but not any list, but rather a ModuleList, because the parameters of each layers are weights learned
# through backpropogation.
# the models needs track all of these.
# it will not be able to do so if we use a vanilla python list.


class Neural(nn.Module):

    def __init__(self, input_size, hidden_layers, output_size, drop=0.5):

        super(Neural, self).__init__()  # inherits the __init__ method of the parent class Module.

        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])

        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop)

    def forward(self, x):

        for linear in self.hidden_layers:
            x = F.relu(linear.forward(x))
            x = self.dropout(x)

        x = self.output(x)

        # Log softmax comes after the output layer.
        # i.e the input to log softmax (or softmax) is the final layer of our network
        # the units of which must equal the number of classes we have.

        return F.log_softmax(x)


print()
# instantiating an object of the Neural class
model = Neural(25088, [5100, 2500, 33], 102)
print()
print(model)
print()

# modulelist will create [nn.Linear([5100, 2500]), nn.Linear([2500, 33])]

print('manual excersize with a list')

print()
s = [1200, 600, 300]
print()

print('hidden layers list')
print(s)
print()

layer_sizes = zip(s[:-1], s[1:])

in_features = 2000
hl = nn.ModuleList([nn.Linear(in_features, s[0])])  # passing in a list because nn.ModuleList takes
#  an iterable

print('before extending with hidden')

print(f'h1:{hl}')
print(f'type: {type(hl)} ')

print('after extending with hidden layers')

# Here, we are extending the module list itself.
hl.extend([nn.Linear(x, y) for x, y in layer_sizes])

print(f'hl:{hl}')
print(f'hl: {type(hl)}')

print('after extending with out_features')

out_features = 20

hl.extend([nn.Linear(s[-1], out_features)])

print(hl)
print(type(hl))

print('iterating through the list')
for linear in hl:
    print(linear)  # give us ---> Linear(in_features =1200, out_features =600),

#  Linear(600, 300) # instances of Linear class

print()
print('nn.Linear exploration')
print()
print('instantiating nn.Linear() creates weight matrix of shape passed into the parameters')
print()

# 'IMPORTANT: A layer is a tensor of weights, be it conv, linear or recurrent'
# passing in data will be a linear transformation of said feature vector with the tensor/matrix of weights
# 'this is followed by activation function'

m = nn.Linear(2, 3)  # this defines a linear map, i.e. y = mx + b, creates a 3*2 weight tensor
# and a 1*3 bias tensor (row tensor)

# nn.Linear(m,n) --> this gives a n*m weight tensor and a 1 * n bias tensor, row tensor.

print(f'm: {m}')
print(f'dir(m):{dir(m)}')
print(f'in_features: {m.in_features}')
print(f'out_features: {m.out_features}')
print(f'parameters of our Linear transformation {m.parameters()}')  # this is a generator object

print('calling next on the above generator object gives us')
print(next(m.parameters()))  # same as calling m.weight

print('the parameters can be filled with whatever one chooses')
print('this is the weight')
print(m.weight.data.fill_(2))

print('this the bias')
print(m.bias.data.fill_(3))

print('a 4 * 2 torch tensor of ones')
inp = torch.empty(4, 2).fill_(1)  # could also do torch.ones()
print(inp)
print(inp.size())
p = m.forward(inp)

# here the 3 * 2 tensor is multiplied by 2 * 4 random tensor and gives a 3*4 tensor,
# add 1*4 bias and transpose that to get a 4 * 3 tensor.
# This is the forward method of the Linear class, which computes the affine transformation between
# the tensor passed as input matrix defined in it.
# we do this inside the forward() method of our Neural network class. The base class has this
# method which is not implemented.

# Thus an (m*n)^t * (d*m)^t + 1 * n bias = (n*d)^t is the output of m.forward(p)
print()

print(p)
print(type(p))
print(p.size())

print()

print('without module list, plain vanilla way of doing things')

# WITHOUT MODULE LIST.


class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.fc1 = nn.Linear(748, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.softmax(x, dim=1)
        return F.log_softmax(x)


model = Model()


class NewModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(NewModel, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


M = NewModel(500, 200, 10)
print('Here I am printing M')
print(M)

# extract weights in our network by accessing them as attributes

print()
print('weights and weights size for first layer: fc1')
print(model.fc1.weight)
print(model.fc1.weight.size())  # 500 * 784, in_vector must be 1*784 which will be transformed to 784 *1
print()

print()
print('weights and weights size for second layer: fc2')
print(model.fc2.weight)
print(model.fc2.weight.size())
print()

print()
print('fc1.bias and fc1.bias.size(), this is bias and bias size for fc1')
print(model.fc1.bias)
print(model.fc1.bias.size())  # no of units in succesive layers * no of units in present layer.
print()

# fill bias with zeros
print()
p = model.fc1.bias.data.fill_(0)
print(p)
print(p.size())
print()

# fill al weights with normal distribution
print()
print('fc1 layer weight matrix can be initialized with mean and standard deviation from the normal distribution')
b = model.fc1.weight.data.normal_(mean=0, std=0.01)
print(b.size())
print()

# Using nn.Sequrntial.
# each layer will be indexed, so nn.Linear(784, 500) is 0, nn.ReLu() is 1
new_model = nn.Sequential(nn.Linear(784, 500), nn.ReLU(), nn.Linear(500, 200), nn.ReLU(),
                          nn.Linear(200, 100), nn.ReLU(), nn.Linear(100, 10), nn.Softmax())

print()
print('using nn.Sequential only')
print(new_model)
print()

# Using OrderedDict with nn.Sequential

print('using nn.Sequential with OrderedDict')
seq_model = nn.Sequential(OrderedDict([('fc1', nn.Linear(1000, 500)), ('relu1', nn.ReLU()), ('fc2', nn.Linear(500, 200)),
                                       ('relu3', nn.ReLU()), ('dropout', nn.Dropout()), ('fc3', nn.Linear(200, 100)),
                                       ('relu4', nn.ReLU()), ('fc4', nn.Linear(100, 10))]))
print()
print('using sequential model')
print(seq_model)
print()

# a convolution model


class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        # ENCODING PART
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        # DECODING PART
        self.fc1 = nn.Linear(in_features=256 * 29 * 29, out_features=100)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=3, stride=2, padding=1))

        x = x.flatten(start_dim=1)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.out(x)

        return F.log_softmax(x, dim=1)


def get_data_target(batch_size):
    input_ = torch.rand([batch_size, 3, 228, 228])
    target = torch.tensor([1, 2, 9])
    return input_, target


def get_criterion_optimizer(model, lr=0.001):

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return criterion, optimizer


def train(model, n_epochs, batch_size, lr):

    print('################HYPERPARAMETERS#####################')
    print(f'number of epochs:{n_epochs}')
    print(f'the batch size:{batch_size}')
    print(f'learning rate:{lr}')

    input_, target = get_data_target(batch_size=batch_size)

    criterion, optimizer = get_criterion_optimizer(model=model, lr=lr)

    for epoch in range(n_epochs):

        running_loss = 0.0

        optimizer.zero_grad()

        out = model.forward(input_)
        loss = criterion(out, target)

        loss.backward()
        optimizer.step()
        running_loss += loss


        print(f'epoch:{epoch}, loss: {loss}')


model = CNN()


# train(model=model, n_epochs=3, batch_size=3, lr=0.001)


# print('loss.grad_fn')
# print('next function on grad_fn')
# print(loss.grad_fn.next_functions[0])
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0])
#
#
# model.eval()
# with torch.no_grad():
#     s = model.forward(r)
#
# print('size is s')
# print(s)
# print('here is the size')
# print(s.size())  # batch_size * # of classes


# this is good for weight initilization
params = list(model.parameters())
print('parameters of our model')
print(params)
print('length')
print(len(params))
print('')
print('')
print('the first conv layer parameters')
print(params[0])
print('parameter size')
print(params[0].size())
print('the first filter')
print(params[0][0])
print('size of this filter')
print(params[0][0].size())

#
# class LR(nn.Module):
#
#     def __init__(self, input_size, output_size):
#
#         super().__init__()
#
#         self.linear = nn.Linear(in_features=input_size, out_features=output_size)
#
#     def forward(self, x):
#         y = self.linear(x)
#         return y
#
#
# model = LR(1, 1)
#
# # print(model)
#
# X = torch.rand(100, 1) * 10
# Y = X + 2*torch.randn(100, 1)
#
#
# def get_prams():
#     w, b = list((model.parameters()))
#     w = w[0][0].item()
#     b = b.item()
#     return (w,b)
#
#
# def plot_func(title):
#
#     plt.title(title)
#
#     slope, y_intcpt = get_prams()
#
#     x = np.array([-1, 10])  # the x here is same as the x there.
#
#     y_pred = slope * x + y_intcpt  # defining predicted y as a function of x.
#     plt.plot(x, y_pred, 'r')  # this will plot the predicted for featiure x and
#     # draw a line through it.
#     plt.scatter(X.numpy(), Y.numpy())  # also plotting the original X and Y as plots for us.
#     plt.show()
#
# # training
#
#
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.1, nesterov=True)
#
# epochs = 200
# losses = []
#
# for epoch in range(epochs):
#
#     y_pred = model.forward(X)
#
#     loss = criterion(y_pred, Y)
#     losses.append(loss)
#
#     # print(f'epoch: {epoch+1} loss is: {loss.item()}')
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#
# plt.plot(range(epochs), losses)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.show()

# plot_func('Trained Model')

print('Encoder and Decoder networks')


# That convolutional part of a network is called the encoding part.
# the fully connected part is called the decoding part. It flattens
# or decodes the patterns/features back.


class Neural(nn.Module):

    def __init__(self, in_features, n_classes):

        super().__init__()

        self.conv_block1 = nn.Sequential(nn.Conv2d(in_features, 6, kernel_size=3),
                                         nn.BatchNorm2d(6),
                                         nn.ReLU()

                                         )

        self.conv_block2 = nn.Sequential(nn.Conv2d(in_features, 12, kernel_size=3),
                                         nn.BatchNorm2d(12),
                                         nn.ReLU()

                                         )

        self.decoder = nn.Sequential(
            nn.Linear(12 * 4 * 4, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, n_classes)

        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        x = x.view(x.size(0), -1)

        x = self.decoder(x)

        return x


new_model = Neural(1, 10)
print(f'new_model: {new_model}')


def encoder(in_features, out_features, *args, **kwargs):
    stack = nn.Sequential(nn.Conv2d(in_features, out_features, *args, **kwargs),
                          nn.BatchNorm2d(out_features),
                          nn.ReLU())
    return stack


class Neural1(nn.Module):

    def __init__(self, in_features, n_classes):

        super().__init__()

        self.conv_block1 = encoder(in_features=in_features, out_features=32, kernel_size=3, padding=1)

        self.conv_block2 = encoder(in_features=32, out_features=64, kernel_size=7, padding=1)

        self.decoder = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, n_classes)

        )

    def forward(self, x):

        x = self.conv_block1(x)

        x = self.conv_block2(x)

        x = x.view(x.size(0), -1)

        x = self.decoder(x)

        return x

print()
print()
print('##########')


print('New Neural Model')

M = Neural1(3, 15)

print(M)

print('nn.Sequential() can have sequential stacks within itself as well')

j = nn.Sequential(nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2)
                                ),

              nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
                            nn.BatchNorm2d(128),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2))
                  )

print(j)

print('Encoder decoder style nn.Sequentrial')


class Neural3(nn.Module):

    def __init__(self, in_channels, n_classes):

        super().__init__()

        self.encoder = nn.Sequential(encoder(in_channels,
                                             out_features=32,
                                            kernel_size=5),

                                     encoder(in_features=32,
                                             out_features=64,
                                             kernel_size=7)

                                     )

        self.decoder = nn.Sequential(nn.Linear(64*7*7, 1024),
                                     nn.Sigmoid(),
                                     nn.Linear(1024, n_classes)

                                     )

    def forward(self, x):

        x = self.encoder(x)

        x = x.view(x.size(0), -1)

        x = self.decoder(x)

        return x


n_model = Neural3(1, 15)
print(n_model)

# Create multiple layers in conv black all at once


class NeuralCNN(nn.Module):

    def __init__(self, in_c, enc_sizes,  n_classes):

        super().__init__()

        self.enc_sizes = [in_c, *enc_sizes]

        conv_stacks = [encoder(in_f, out_f, kernel_size=3) for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]

        self.encoder = nn.Sequential(*conv_stacks)

        self.decoder = nn.Sequential(nn.Linear(12 * 5*5, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, n_classes))

    def forward(self, x):

        x = self.encoder(x)

        x = x.view(-1, self.num_flat_feaures(x))

        x = self.decoder(x)

        return x

    def num_flat_feaures(self, x):

        x = x.size()[1:]

        dims = 1

        for num in x:

            dims *= num

        return dims


print('using conv_blocks in a list')


u = NeuralCNN(3, [32, 64, 128], 100)

print(u)

print('Use a decoding block as well')


def dec_block(inf, outf):

    return nn.Sequential(nn.Linear(inf, outf),
                         nn.Sigmoid())


class Network(nn.Module):

    def __init__(self, in_c, enc_sizes, dec_sizes, out_c):

        super().__init__()

        self.enc_sizes = [in_c, *enc_sizes]
        self.dec_sizes = [32 * 28 * 28, *dec_sizes]

        conv_blocks = [encoder(in_f, out_f) for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]

        self.encoder = nn.Sequential(*conv_blocks)  # star unpacks each encoding block into

        dec_blocks = [dec_block(inf, out) for inf, out in zip(self.dec_sizes, self.dec_sizes[1:])]

        self.decoder = nn.Sequential(*dec_blocks)

        self.last = nn.Linear(self.dec_sizes[-1], out_c)

    def forward(self, x):

        x = self.encoder(x)

        x = x.view(x.size(0), -1)

        x = self.decoder(x)

        return x


print('Recurrent Nueral Networks')

# Recurrent Neural Networks: Super awesome!

# They operate over sequences.
# have multiple inputs and multiple outputs: many-to-many: language translation
# multiple inputs and one output: many-to-one: sentiment classification
# also one-to-one: plain feed-forward or CNN,
# one-to-many: Image Captioning: image---> CNN--->RNN: sequence of data

# Unlike feed-forward neural networks,
# RNN's loop the output (called hidden state/activation) at each time step back into the input.
# each time step has current input and a previous input.

# Unrolling an RNN at each time step gives us an rrn of t layers.
# thus it maintains the state of each word.

# Used for Speech-to-text. Image captioning and text generation.

# RNN's suffer from vanishing gradient problem and short-term memory.

# LSTM AND GRU networks solve the vanishing gradient and exploding gradient problem by having internal
# gates at every cell, which decides what information to forget and what to pass on.

# these gates also solve the vanishing gradient problem by
# letting the gradient never collapse no matter how small they become.

# The recurrent steps are not fixed and vary depending on the input being processed.

# These language models output a probability distribution over data.
# During test time we predcit the probability for each letter/word depending on weather
# we are operating over characters/letters or over data.
# pick the one with the highest probability and feed that as input in the next time step.


all_letters = string.ascii_letters + " .,;'"  # A string of the Egnlish Alphabet: ABCD......abcd.......,.;'"
n_letters = len(all_letters)


def unicodetoascii(s):
    """converts string from unicode to ascii, """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'
            and c in all_letters)


def lettertoindex(l):
    return all_letters.find(l)


def nametotensor(line):
    """convert name to a tensor, insert 1 at each lth position for each tensor"""
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][lettertoindex(letter)] = 1
        # tensor[0][0][li] or tensor[1][0][next_li] and so on. For each index position of the alphabet.
    return tensor


# Creating the network
class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # Weight tensor for input to hidden
        self.i2o = nn.Linear(input_size + hidden_size, output_size)  # Weight tensor for input to output
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,  input, hidden):
        # combined.size() = torch.Size([1, input_size + hidden_size]) since dim=1
        combined = torch.cat((input, hidden), dim=1)
        hidden = F.tanh(self.i2h(combined))  # tanh(the linear transformation inside)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def inithidden(self):
        return torch.zeros(1, self.hidden_size)


n_categories = 3
n_hidden = 128
hidden = torch.zeros(1, n_hidden)
name = nametotensor(unicodetoascii('Shiva'))

rnn = RNN(n_letters, n_hidden, n_categories)

# the hidden state at index 0 of the word is fed as
# input to index 1 of the word, this process
# is repeated many times till the last index of the word.

# the number of outputs and hidden states
# will be equal to the length of the word.
# thus here we have a total of 5 outputs and 5 hidden states

for i in range(name.size()[0]):
    output, hidden = rnn.forward(name[i], hidden)  # (name[0], hidden), (name[1], hidden), hidden for name[1] is the hiddem that came
    # out for name[0] and hidden for name[2] is the hidden that came out for name[1].
    print(output)  # 1 * output_size
    print(hidden)  # 1 * n_hidden
    print(hidden.size())


print('doing it using nn.RNN()')
# nn.RNN(input_size, hidden_size, num_layers, non_linearity, bias)
r = nn.RNN(57, 128, 2)  # bias is True and non_linearity is 'tanh' by default, can change to relu or sigmoid.
print('r')
print(r)
# by default the number if layers is 1


print('the name')
print('input size must be [seq_len, batch_size, input_size]')
print(name)
print(name.size())  # [5, 1, 57 ] 5: seq_len, batch_size: 1, input_size: 57


print('initialzing the first hidden state tensor')
h0 = torch.zeros(2, 1, 128)  # num_layers * num_directions, batch_size,  hidden_size
print('ho')
print(h0)

# our network has 128 units in the hidden layers.

# important to distinguish between number of layers in the network and number of hidden neurons/units in each layer.

output_n, hn = r.forward(name, h0)  # each  hidden state will be input to the next time step.
print('output_n')
print(output_n)
print(output_n.size())  # seq_len * batch_size * n_hidden
print('hn')
print(hn)
print(hn.size())  # num_layers, batch_size, n_hidden

print()
print('Try it this time with batch_first =True')
print()

cell = nn.RNN(input_size=4, hidden_size=2, batch_first=True)  # num_layers = 1 by default.
inpt = torch.tensor([[[1, 0, 0, 0]], [[0, 1, 0, 0]], [[0, 0, 1, 0]]], dtype=torch.float)
# batch_size:3, seq_len: 1, input_size:4; seq_len is first if batch_first = False.
print(inpt)
print(inpt.size())

h = torch.zeros(1, 3, 2)  # num_layers * num_directions, batch_size, hidden_size;  1, 3, 2
print(h)
print(h.size())


o, h = cell.forward(inpt, h)
print(f'o: {o}, size: {o.size()}')  # 3, 1, 2: batch, seq, hidden
print(f'h: {h}, size: {h.size()}')  # 1, 3, 2

# With batch_first = True

cell1 = nn.RNN(input_size=10, hidden_size=3, num_layers=2, batch_first=True)
print(f'cell1: {cell1}')
ins = torch.zeros(5, 4, 10)  # batch_size, seq_len, input_size
print(f'ins: {ins}')
hid = torch.zeros(2, 5, 3)  # num_layers * num_directions, batch_size, hidden_size
print(f'hid : {hid}')

print(f'ins: {ins.size()}')
print(f'hid: {hid.size()}')

o1, h1 = cell1.forward(ins, hid)
print(f'o1: {o1}, size: {o1.size()}')  # batch_size * seq_len * hidden_size, 5, 4, 3
print(f'h1: {h1}, size: {h1.size()}')  # num_layers, batch_size, hidden_size
num_classes = 2
h1 = h1.view(-1, num_classes)
print(f'h1: {h1}, size: {h1.size()}')

# num_layers = 1
# batch_size = 1
# num_classes = 18


class Rec(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,  num_layers=1):

        super(Rec, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # input and output aren't hyper-parameters, it doesn't require initialization.

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(1)
        hidden = self.init_hidden(batch_size)
        output, hidden = self.gru(x, hidden)
        fc_out = self.fc(hidden)
        return fc_out

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


rnn_new = Rec(input_size=57, hidden_size=128, output_size=18)
print(rnn_new)


another_out = rnn_new.forward(name)
print(f'another_out: {another_out}, size:{another_out.size()}')

# long short term memory networks: LSTM


class OneLayerRnn(nn.Module):

    def __init__(self, n_inputs: int, n_neurons: int):

        super(OneLayerRnn, self).__init__()

        self.Wx = torch.randn(n_inputs, n_neurons)
        self.Wy = torch.randn(n_neurons, n_neurons)

        self.b = torch.zeros(1, n_neurons)

    def forward(self, X0, X1):

        self.Y0 = torch.tanh(torch.mm(self.Wx, self.X0) + self.b)

        self.Y1 = torch.tanh(torch.mm(self.Wy, self.Y0) + self.torch.mm(self.Wx, self.X1) + self.b)

        return self.Y0, self.Y1


n_inputs = 4
n_neurons = 1

single_rnn = OneLayerRnn(n_inputs, n_neurons)
print(single_rnn)


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers):

        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(1)
        hidden = self.init_hidden(batch_size)
        output, hidden = self.gru(x, hidden)
        fc_out = self.fc(hidden)
        return fc_out

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


lstm = LSTM(input_size=57, hidden_size=128, output_size=18, num_layers=2)
print(lstm)






