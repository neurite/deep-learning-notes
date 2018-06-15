import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LstmClassifier(nn.Module):
    """
    Three-layer LSTM classifier.

    The three layers are:

    1) Embedding
    2) LSTM
    3) FC

    # Arguments:
        vocab_size: The size of the vocabulary
        embed_size: The size of the embeddings, the number of
            embedding dimensions
        lstm_size: The output size of the LSTM layer, the size of
            the LSTM hidden state
        output_size: The final output size
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
            x: Sequence tensor. The tensor has shape (batch_size,
                sequence_length) where the 1st axis is the mini-batch
                and the 2nd axis is the sequence. Each sequence is a
                sequence of number tokens. Number tokens can be
                generated using the hashing trick for example. When
                x is passed in as batch size > 1, the sequences in
                the same batch should be padded to the same length.
        # Return:
            yhat of (batch_size, output_size)
        """
        batch_size = x.shape[0]
        x = autograd.Variable(x)

        # Step 1 Embedding
        # (mini-batch, sequence) -> (mini-batch, sequence, embedding)
        embeddings = self.embedding(x)

        # Step 2 LSTM
        # 1) Memory is reset on each call -- stateless LSTM
        # 2) For multiple layers and bidirectional, the first axis of
        #    h and c would be num_layers * num_directions, our case
        #    here is 1 layer, 1 direction
        h = torch.zeros(1, batch_size, self.lstm_size)
        c = torch.zeros(1, batch_size, self.lstm_size)
        h = autograd.Variable(h)
        c = autograd.Variable(c)
        # (sequence, mini-batch, embedding) -> (1, mini-batch, lstm)
        output, (h, c) = self.lstm(embeddings.transpose(0, 1), (h, c))

        # Step 3 FC
        # Ignore the sequence out, pass the hidden state at t = seq_len
        # Note the LSTM out is essentially all the hidden states at the
        # sequence steps stacked together
        # (mini-batch, lstm) -> (mini-batch, output)
        y = self.linear(torch.squeeze(h))
        return F.log_softmax(y, dim=1)

    def train(self, x, y, epoches, batch_size=32, lr=0.001):
        """
        # Arguments:
            x: Tensor of shape (N, seqence-length). Each sequence is a
                list of number tokens. All the sequences need to be
                padded or clipped to the same length.
            y: Labeled output tensor of shape (N, output-size).
            epoches: The number of epoches to run.
        """
        # http://pytorch.org/docs/master/nn.html#embedding
        # Keep in mind that only a limited number of optimizers
        # support sparse gradients: currently it’s optim.SGD
        # (cuda and cpu), optim.SparseAdam (cuda and cpu) and
        # optim.Adagrad (cpu).
        optimizer = optim.SparseAdam(self.parameters(), lr=lr)
        # Sigmoid + BCELoss = multiclass, multilabel
        loss = nn.CrossEntropyLoss(yhat, y)
        for _ in epoches:
            start = 0
            n, _ = x.shape
            while start < n:
                self.zero_grad()
                xbatch = x[start:start+batch_size]
                ybatch = y[start:start+batch_size]
                xgrad = autograd.Variable(xbatch)
                ygrad = autograd.Variable(ybatch)
                yhat = self(xgrad)
                loss.backward()
                self.optimizer.step()
                start += batch_size
