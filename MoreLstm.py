import torch.nn as nn
import torch
torch.manual_seed(2)
import string
import torch.nn.functional as F
import torch.optim as optim

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

# Word Embeddings : Finds relationship between data.

e = nn.Embedding(num_embeddings=26, embedding_dim=100)  # 26 * 100 embedding matrix.

# 20: num_embeddings, this is the vocabulary size, we would like to embed each word in our vocabulary.
# 100: embedding dimension,

# this means embed 20 data/characters into a 100 dimensional space.

# This matrix is called embedding matrix and is randomly initialized.
# We multiply this with one-hot vector for each word giving the word-embedding for
# that particular word.

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

# voncab_size * num_embedding dimensions
Embedding = nn.Embedding(num_embeddings=1008, embedding_dim=500)
print(Embedding)

print('iterating over our one hot list')

for vector in one_hot_vec:
    embedding = Embedding(vector.view(-1, 1008))
    print(embedding)
    print(embedding.size())
    print('')

print('without one-hot-encoding')


new_vocab = set(sentence.split())

new_idx = {w : i for i, w in enumerate(new_vocab)}

print(new_idx)

vec = torch.tensor([new_idx['orange']], dtype=torch.long)
print(vec)

e1 = nn.Embedding(num_embeddings=len(train_data), embedding_dim=10)

word_embedding = e1(vec)
print(word_embedding)
print(word_embedding.size())  # word embedding for orange

print('')
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
print(w_idx)

trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]

# 115- 2= 113, range(113)= 0, 1, 2, 3
# ===> test_sentence[0], test_sentence[0+1] = ([when, forty], winters)
# thus each word, the next word, are the context, the words comming after that is the target
# ([forty, winters], shall), shall would be the
print(trigrams)  # these are three sets of words, thus trigrams, bigrams.

bigrams = [([test_sentence[i]], test_sentence[i+1]) for i in range(len(test_sentence) -2)]


# one word, then the next, then that word, and the word after that.
# This would be useful if we are using single context ----> single pairs.


context_size = 2
embedding_size = 10


name = 'shiva'

str_ascci = [ord(c) for c in name]


class NgramLanguageModeler(nn.Module):
    """these are also called skip-gram models, they predict target given context,
    more mathematicaly the estimate P(C|T) ---> Context|Target.
    """

    def __init__(self, vocab_size, embeddeing_size, context_size):
        super(NgramLanguageModeler, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embeddeing_size)  # 97, 10
        self.fc1 = nn.Linear(context_size * embeddeing_size, 128)  #
        self.fc2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))  # make it a row vector.
        out = F.relu(self.fc1(embeds))
        out = self.fc2(out)
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


x = trigrams[0]
print(x)
print(x[0])


g = nn.Embedding(97, 10)

inp = torch.tensor([w_idx[wrd] for wrd in x[0]])
print(inp)

new_inp = inp.view(1, -1)
print(new_inp)

s = g(new_inp)
print(s)
print(s.size())