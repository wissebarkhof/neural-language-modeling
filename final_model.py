from torch import nn
from .lstm import LSTMCustom

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        # quick fix to match up dimensions of internal LSTM layers
        # setting all but last LSTM layer to transform [ninp -> nhid], and last one [nhid -> nhid]
        LSTM_dimensions = [(ninp, nhid) if i == 0 else (nhid, nhid) for i in range(nlayers)]
        rnns = [LSTMCustom(*LSTM_dimensions[i]).to(device) for i in range(nlayers)]
        self.rnns = ListModule(*rnns)

#         if rnn_type in ['LSTM', 'GRU']:
#             self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
#         else:
#             try:
#                 nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
#             except KeyError:
#                 raise ValueError( """An invalid option for `--model` was supplied,
#                                  options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
#             self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, hidden):
        # Send to device explicitly, because the loop over LSTM layers
        # in the Net breaks the normal .to(decvice) functionality
        x = x.to(device)
        hidden = (hidden[0].to(device), hidden[1].to(device))
        x = self.drop(self.encoder(x))
        for i, rnn in enumerate(self.rnns):
          x, hidden = rnn(x, hidden)
        x = self.drop(x)
        decoded = self.decoder(x.view(x.size(0)*x.size(1), x.size(2)))
        ext_output = decoded.view(x.size(0), x.size(1), decoded.size(1))
        return ext_output, hidden

    def init_hidden(self, bsz):
        init_range = 1/np.sqrt(self.nhid)
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(1, bsz, self.nhid).uniform_(-init_range, init_range),
                    weight.new_zeros(1, bsz, self.nhid).uniform_(-init_range, init_range))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid).uniform_(-init_range, init_range)
