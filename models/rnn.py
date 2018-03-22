import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

class TextClassifier(nn.Module):

    """
    Three-layer text classifier. The three layers are:

    1) Embedding
    2) LSTM
    3) FC

    # Arguments:
        vocab_size: the size of the vocabulary
        embedding_size: the size of the embeddings
        lstm_size: the output size of the LSTM layer
        output_size: the final output size
    """
    def __init__(self, vocab_size, embedding_size=32, lstm_size=32,
            output_size=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, lstm_size)
        self.linear = nn.Linear(lstm_size, output_size)
        self.lstm_size = lstm_size

    def forward(self, x):
        """
        # Arguments:
            x: sequence tensor, each sequence is a sequence of number
                tokens, number tokens can be generated using the
                hashing trick for example; the tensor is of shape
                (batch_size, sequence_length) where the 1st axis is
                the mini-batch and the 2nd axis is the sequence; if
                the batch size is > 1, the sequence can be padded
                to the same length using 0s
        """
        batch_size = x.shape[0]

        # Step 1 Embedding
        # (mini-batch, sequence) -> (mini-batch, sequence, embedding)
        x = autograd.Variable(x)
        embeddings = self.embedding(x)

        # Step 2 LSTM
        # 1) Memory is reset on each call -- stateless LSTM
        # 2) For multiple layers and bidirectional, the first axis of
        #    h and c would be num_layers * num_directions
        h = autograd.Variable(torch.zeros(1, batch_size, self.lstm_size))
        c = autograd.Variable(torch.zeros(1, batch_size, self.lstm_size))
        # (sequence, mini-batch, embedding) -> (1, mini-batch, lstm)
        lstm_out, (h, c) = self.lstm(embeddings.transpose(0, 1), (h, c))

        # Step 3 FC
        # Ignore the sequence out, pass the hidden state at t = seq_len
        # Why not:
        # 1) y<t> --> what is y like?
        # 2) c --> c is already used to compute h
        print(lstm_out.shape)
        print(lstm_out[:-1])
        y = self.linear(torch.squeeze(h))
        return F.log_softmax(y)

    def train(self, epoches, batch_size=1, lr=0.01):
        # http://pytorch.org/docs/master/nn.html#embedding
        # Keep in mind that only a limited number of optimizers
        # support sparse gradients: currently it’s optim.SGD
        # (cuda and cpu), optim.SparseAdam (cuda and cpu) and
        # optim.Adagrad (cpu)
        optimizer = optim.SparseAdam(self.parameters(), lr=lr)
        for _ in epoches:
            self.zero_grad()
            x_grad = autograd.Variable(x)
            y_grad = autograd.Variable(y)
            y_hat = self(x_grad)
            # Sigmoid + BCELoss = multiclass, multilabel
            loss = nn.CrossEntropyLoss(y_hat, y)
            loss.backward()
            self.optimizer.step()
