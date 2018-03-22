import torch
from models.rnn import TextClassifier

def test():
    vocab_size = 10000
    model = TextClassifier(vocab_size)
    x = torch.Tensor(32, 500).uniform_(0, vocab_size)
    x = x.type(torch.LongTensor)
    yhat = model(x)

