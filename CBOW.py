import torch.nn as nn
import torch
torch.manual_seed(2)
import torch.nn.functional as F
import torch.optim as optim

# https://pytorch.org/tutorials/beginner/nlp/
# word_embeddings_tutorial.html#word-embeddings-in-pytorch

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)
print('size of vocab')
print(vocab_size)
print('')


word_to_ix = {word: i for i, word in enumerate(vocab)}
# print(word_to_ix)

data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))

print('context, target')
print(data[:1])


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding):

        super(CBOW, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding)
        self.linear = nn.Linear(embedding, vocab_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs.view(1, -1))
        out = self.linear(embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


# create your model and train.  here are some functions to help you make
# the data ready for use by your module

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


# instantiate model
model = CBOW(49, 70)
print(model)

# loss and optimizer
losses = []
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# input
context_vec = make_context_vector(data[0][0], word_to_ix)

# context vector
print(context_vec)

# target
t = torch.tensor(word_to_ix[data[0][1]], dtype=torch.long)
print(data[0][1])
print(t)


out = model(context_vec)
print(out)
print(out.size())





