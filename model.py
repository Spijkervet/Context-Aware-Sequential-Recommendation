import torch

class DummyModel(torch.nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, dropout=0, device='cuda:0'):

        super(DummyModel, self).__init__()
        # Initialization here...

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.device = device

        self.lstm = torch.nn.LSTM(
            input_size=vocabulary_size,
            hidden_size=lstm_num_hidden,
            num_layers=lstm_num_layers,
            dropout=dropout
        )

        self.linear = torch.nn.Linear(lstm_num_hidden, vocabulary_size)
        self.softmax = torch.nn.Softmax(dim=0)

        self.h = torch.zeros((lstm_num_hidden, 1)).to(device)
        self.c = torch.zeros((lstm_num_hidden, 1)).to(device)

    def one_hot(self, x):
        t = torch.zeros((x.shape[0], x.shape[1], self.vocabulary_size)).to(self.device)
        # flatten it to get a row vector containing the ids
        ids = x.unsqueeze(-1).to(self.device)
        one_hot = t.scatter_(2, ids, 1)
        return one_hot
        # # old code for one batch:
        # t = torch.zeros((batch.shape[0], depth))
        # one_hot = t.scatter_(1, batch, 1)
        # # print(one_hot.argmax(dim=1))
        # return one_hot


    def forward(self, x):
        # Implementation here...

        h = self.h
        c = self.c

        # self.lstm.flatten_parameters()
        one_hot = self.one_hot(x)
        out, (h, c) = self.lstm(one_hot)
        out = self.linear(out)
        return out