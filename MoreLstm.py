import torch.nn as nn
import torch
torch.manual_seed(2)
import string
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# RNN, LSTM, GRU

inputs = [torch.rand((4, 3)) for _ in range(3)]
print(f'inputs: {inputs}')


inp_1 = inputs[0]
print(f'before unsqueeze: {inp_1}')
inp_1.unsqueeze_(1)
print(f'afetr unsqueeze: {inp_1}')
print(f'size: {inp_1.size()}')  # seq_len, batch_size, input_size


# Initialize the initial hidden state and cell state randomly, mostly it is initialized to zero.
h_0, c_0 = (torch.rand(2, 1, 4),
          torch.randn(2, 1, 4))  # num_layers* num_directions, batch, hidden

print(f'h_0: {h_0}')  # h_0 only for rnn and gru
print(f'c_0: {c_0}')  # h_0 and c_0 for for lstm

h = h_0  # only require h_0 for rnn and gru as mentioned earlier.
print(h)
print(h.size())

rnn = nn.RNN(input_size=3, hidden_size=4, num_layers=2, bidirectional=False)
print(rnn)


o, h_rnn = rnn(inp_1, h)
print(f'out_gru: {o}')
print(f'h_gru: {h_rnn}')
print(h.size())

print('')
print('LSTM')

lstm = nn.LSTM(input_size=3, hidden_size=4, num_layers=2, bidirectional=False)
print(lstm)

inp_2 = inputs[1]
print(f'before unsqueeze: {inp_2}')
inp_2.unsqueeze_(1)
print(f'afetr unsqueeze: {inp_2}')
print(f'size: {inp_2.size()}')

out, hidden = lstm(inp_2, (h_0, c_0))
print(f'out: {out}, size: {out.size()}')  # seq_len, batch_size, num_directions * hidden_size: 4, 1, 1*4 --> 4, 1, 4.
print(f'hidden: {hidden}, \n'
      f''f'size, h_4: {hidden[0].size()}, size: c_4:{hidden[1].size()}')  # num_layers*num_directions, batch, hidden

# in an LSTM with layers >=2, h_0 and c_0 will contain the hidden state of the last time step for each layer.
# so a 5 layer lstm operating on  7 seq_len will have 5 h_0's and 5 c_0's.
# the last output will have the h_0 of the last layer.

print('')
print('GRU')

gru = nn.GRU(input_size=3, hidden_size=4, num_layers=2, bidirectional=False)
print(gru)

inp_3 = inputs[2]
print(f'before unsqueeze: {inp_3}')
inp_3.unsqueeze_(1)
print(f'afetr unsqueeze: {inp_3}')
print(f'size: {inp_3.size()}')

out_gru, hidden_gru = gru(inp_3, h)
print(f'out_gru: {out_gru}')
print(f'h_gru: {hidden_gru}')
print(hidden_gru.size())

print('')
print('Bidirectional')

bi_rnn = nn.RNN(input_size=5, hidden_size=6, num_layers=3, bidirectional=True)

in_ = torch.rand(3, 1, 5)

hid = torch.rand(6, 1, 6)

output, hi = bi_rnn(in_, hid)
print(output)  # 3, 1, 6*2, input, batch, num_dir * batch_size
print(hi)  # 6, 1, 6

print('')
print('WORD EMEBEDDINGS #########################')
print('')

# Word Embeddings : Finds relationship between words.

e = nn.Embedding(num_embeddings=2, embedding_dim=5)  # 2 * 5 embedding matrix.

# 2: num_embeddings, this is the "vocabulary size": i.e. number of words in pur vocab.
# 5: embedding dimension, the dimensionality of the vector which is to represent each word.

# this means embed each word in my vocab into a 5 dimensional space.
# The tensor e is an embedding matrix and is randomly initialized.

print(e)
print(type(e))
print(e.weight)
print(e.weight.size())

word_to_idx = {'Being': 0, 'Becoming': 1}

lookup1 = torch.tensor(word_to_idx['Becoming'], dtype=torch.long)
print('lookup1')
print(lookup1)

lookup2 = torch.tensor(word_to_idx['Being'], dtype=torch.long)
print('lookup2')
print(lookup2)

print('for lookup1')
# Becoming has index 1
embed_becoming2 = e(lookup1)
print(embed_becoming2)  # this returns the row with index 1 from our embedding matrix

print('for lookup2')
embed_becoming2 = e(lookup2)
print(embed_becoming2)  # returns oth index

# When an word that maps to index i is passed into the embedding layer,
# the embedding layer returns the ith row of the V*N matrix. This integer maps to a word.
# thus [-0.8140, -0.0086, -0.4885, -0.5024, -1.2709] is the vector for 'Becoming'. Index 1
# [-1.3469, -1.2861,  0.7679, -1.6956,  0.9584] is the vector for 'Being'. Index 0.


print('')
print('Let us see this on some training data')

# start with some data.
sentence = "I want a glass of orange juice to go along with my cereal"

# get data into a list of words
train_data = sentence.split()

# Load the vocabulary
with open("data/words") as f:
    Vocabulary = f.read().strip().split('\n')

# tokenize: word in vocab to index
vocab_indx = {l: i for i, l in enumerate(Vocabulary)}


# one hot encoding of words in our sentence based on the indices of these words in the dictionary.
def one_hot(wrd):
    tensor = torch.zeros(len(Vocabulary), 1).type(torch.LongTensor)
    tensor[vocab_indx[wrd]][0] = 1
    return tensor


# list of one-hot vectors
one_hot_vec = [one_hot(wrd=wrd) for wrd in train_data]

# vocab_size * num_embedding dimensions
Embedding = nn.Embedding(num_embeddings=1008, embedding_dim=500)

# print('')
# print('iterating over our one hot list')

# Not sure about this, don't know if I am mistaken in converting to one hot befreo apssing into embedding layer.
# for vector in one_hot_vec:
#     embedding = Embedding(vector.view(-1, 1008))
#     print(embedding)
#     print(embedding.size())
#     print('')

print('')
print('This is for word orange')
print('')


new_vocab = set(sentence.split())
print(f'new_vocab: {new_vocab}')
print(f'len(new_vocab): {len(new_vocab)}')

new_idx = {w: i for i, w in enumerate(new_vocab)}
print(f'new_idx: {new_idx}')

vec = torch.tensor([new_idx['orange']], dtype=torch.long)
print(f'vec: {vec}')

e1 = nn.Embedding(num_embeddings=len(new_vocab), embedding_dim=10)  # 13 * 10
print(f'e1:{e1}')
print(f' embedding tensor weights : {e1.weight}')

word_embedding = e1(vec)
print(f'word_embedding: {word_embedding}')
print(f'word_embedding.size(): {word_embedding.size()}')  # word embedding for orange

linear1 = nn.Linear(10, 100)  # the vector obtained from the embedding layer is passed into
# linear layer. Comes out as 10 * 100 matrix

# These are the weights between hidden layer and output layer
linear2 = nn.Linear(100, 13)

# pass in vector through the single layer neural network
i_to_h = linear1(word_embedding)
h_to_o = linear2(i_to_h)

softmax = F.softmax(h_to_o, dim=1)

# *****AFTER TRAINING, WE ARE INTERESTED IN THE WEIGHTS BETWEN HIDDEN TO OUTPUT****.
print('hidden to out wieghts')
print(linear2.weight)
print(linear2.weight.size())  # these weights learnt in order to minimize our loss function are the word vectors.



print('')
# Not sure if this is correct, should we one hot encode the vectors before and
# then pass it to embedding layer, or is passing in the index enough and the matrix does it for you?
# FIGURE THIS OUT!

# want = one_hot('want')
# print(want)

# print(vocab_indx['want'])

# print(want[944][0])

# print(Embedding)
# m = Embedding.weight
# row = m[944]
# print(row)
# print(row.size())

# print('embeding ')
# embedding = Embedding(want)
# print(embedding)
# print(embedding[944])
# print(embedding[944].size())


print('Tutorail: official docs #########################')
print('')

# Create test sentence

# Following the above tutorial to understand word embeddings

# https://pytorch.org/tutorials/beginner/nlp/
# word_embeddings_tutorial.html#word-embeddings-in-pytorch

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# Create a set of unique words which form our vocabulary
vocab = set(test_sentence)

w_idx = {w: i for i, w in enumerate(vocab)}

trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]

# 115- 2= 113, range(113)= 0, 1, 2, 3..............112
# ===> test_sentence[0], test_sentence[0+1] = ([when, forty], winters)
# thus each word, the next word, are the context, the words comming after that is the target
# ([forty, winters], shall), shall would be the target: output

print(trigrams[:3])

bigrams = [([test_sentence[i]], test_sentence[i+1]) for i in range(len(test_sentence) -2)]
print(bigrams[:3])


# one word, then the next, then that word, and the word after that.
# This would be useful if we are using single context ----> single pairs.


context_size = 2
embedding_size = 10


class NgramLanguageModeler(nn.Module):
    """ n-gram model to predict a target given context.
    More mathematicaly: Estimate P(T|Context word(s))
    Skip-gram because  predicts context given words.
    """

    def __init__(self, vocab_size, embeddeing_size, context_size):
        super(NgramLanguageModeler, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embeddeing_size)  # 97, 10
        self.fc1 = nn.Linear(context_size * embeddeing_size, 128)  #
        self.fc2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))  # make it a row vector.
        out = F.relu(self.fc1(embeds))
        out = self.fc2(out)  # the weights of this layer is what we are interested in.
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
NgramModel = NgramLanguageModeler(len(vocab), embedding_size, context_size)
optimizer = optim.SGD(NgramModel.parameters(), lr=0.001)

print(NgramModel)

for epoch in range(10):
    total_loss = 0
    for context, target in trigrams:

        context_ids = torch.tensor([w_idx[w] for w in context])

        NgramModel.zero_grad()

        log_probs = NgramModel(context_ids)  # indices of the context words
        loss = loss_function(log_probs, torch.tensor([w_idx[target]], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)

print(losses)

print(NgramModel.fc2.weight)
print(NgramModel.fc2.weight.size())  # each row in this one are are word vectors for all 97 words in the vocabulary.

print('')
print('Trying with just a single example and only for embedding layer')

x = trigrams[0]
print(x)
print(x[0])


g = nn.Embedding(97, 10)

inp = torch.tensor([w_idx[wrd] for wrd in x[0]])

print(g)
print(g.weight)

print(g.weight[inp[0]])
print(g.weight[inp[1]])

print(inp)


# resize
new_inp = inp.view(1, -1)
print(new_inp)

print('perform embedding')
s = g(new_inp)  # gets the ith rows in g that correspond tot he indx in inp
print(s)
print(s.size())


sent = "Why are you a baby"
sent2 = "are you a cat"
sent3 = "I am awesome"

the_vocab = set(sent.split() + sent2.split() + sent3.split())
print(the_vocab)

dict_ = {w: i for i, w in enumerate(the_vocab)}
print(dict_)


def enocde(word):
    """plain stupid encode function"""
    vector = []

    for w in the_vocab:
        if word == w:
            vector.append(1)
        else:
            vector.append(0)
    return vector


u = enocde('cat')
print(u)


def encode_numpy(word):
    """much faster, no loops!"""

    vect = np.zeros(len(the_vocab), dtype=np.int)

    vect[dict_[word]] = 1

    return vect


s = encode_numpy('awesome')
print(s)

# MANUALLY COMPUTING WORD-2 VECTOR EMBEDDINGS

embedding_matrix = torch.tensor([[1, 2, 5, 7],
                                 [4, 5, 3, 8],
                                 [1, 7, 3, 3],
                                 [4, 9, 1, 0],
                                 [2, 0, 1, 2]])
print(embedding_matrix)
print(embedding_matrix.size())

one_hot = torch.tensor([[0, 0, 0, 1, 0]], dtype=torch.long)  # Vocab size = 5
print(one_hot)
print(one_hot.size())

# manually computing hidden activation
hidden_activation_manual = torch.matmul(one_hot, embedding_matrix)
print(hidden_activation_manual)

# computing with nn.Embedding
Emb = nn.Embedding(num_embeddings=5, embedding_dim=4)
print(Emb.weight)

# indx of the word in the vocab.
indx = torch.tensor([3], dtype=torch.long)
print(indx)

# Passing in one-hot vector instead of indx.
# Not really correct I think.
# The resulting matrix keeps everything constant
# while changing just the ith row in question.

hidden_activation_nn = Emb(indx)
print(hidden_activation_nn)

# pass many indices
indices = torch.tensor([1, 3, 2], dtype=torch.long)
print(indices)

hidden_activation_mutliple = Emb(indices)
print(hidden_activation_mutliple)

avg = hidden_activation_mutliple.mean(0)
print(avg)
