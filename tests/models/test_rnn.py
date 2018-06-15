import torch

from models.rnn import LstmClassifier

def test_forward():
    vocab_size = 10000
    model = LstmClassifier(vocab_size)
    x = torch.Tensor(32, 500).uniform_(0, vocab_size)
    x = x.type(torch.LongTensor)
    yhat = model(x)

