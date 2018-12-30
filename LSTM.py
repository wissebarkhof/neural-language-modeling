
from torch import nn
import torch.nn.functional as F



class LSTMCustom(nn.Module):
    def __init__(self, word_dim, hidden_dim, nlayers = 1, activation = 'sigmoid'):
        super(LSTMCustom, self).__init__()

        self.word_dim = word_dim
        self.hidden_dim = hidden_dim

        # weights for x
        self.weights_xi = self.init_weights((hidden_dim, word_dim))
        self.weights_xo = self.init_weights((hidden_dim, word_dim))
        self.weights_xf = self.init_weights((hidden_dim, word_dim))
        self.weights_xc = self.init_weights((hidden_dim, word_dim))

        # weights for hidden
        self.weights_hi = self.init_weights((hidden_dim, hidden_dim))
        self.weights_ho = self.init_weights((hidden_dim, hidden_dim))
        self.weights_hf = self.init_weights((hidden_dim, hidden_dim))
        self.weights_hc = self.init_weights((hidden_dim, hidden_dim))
        self.weights_h_out = self.init_weights((hidden_dim, hidden_dim))

    def init_weights(self, dim):
        return nn.Parameter(torch.FloatTensor(dim[0], dim[1]).uniform_(-np.sqrt(1./dim[0]), np.sqrt(1./dim[1])), requires_grad=True)

    def init_gates(self):
        return torch.zeros(self.hidden_dim,  requires_grad = True)

    def init_hidden(self, batch_size):
        layer = torch.zeros((1, batch_size, self.hidden_dim),  requires_grad = True)
        return (layer.clone(), layer.clone())

    def step(self, lr):
        for p in self.parameters():
            p.data.add_(-lr, p.grad.data)

    def forward_step(self, xt, hidden_t_1):

        ht_1, ct_1 = hidden_t_1
        gate_i = torch.sigmoid(
            F.linear(xt, self.weights_xi) +
            F.linear(ht_1, self.weights_hi))

        gate_f = torch.sigmoid(
            F.linear(xt, self.weights_xf) +
            F.linear(ht_1, self.weights_hf))

        gate_o = torch.sigmoid(
            F.linear(xt, self.weights_xo) +
            F.linear(ht_1, self.weights_ho))

        new_c = torch.tanh(
            F.linear(xt, self.weights_xc) +
            F.linear(ht_1, self.weights_hc))

        ct = gate_f * ct_1 + gate_i * new_c

        ht = gate_o * torch.tanh(ct)

        output = F.linear(ht, self.weights_h_out)

        return output, (ht, ct)

    def forward_propagation(self, x, hidden_t_1):
        # Get sequence length (bptt), batch_size from the input
        bptt, batch_size, _ = x.size()
        output = torch.zeros((bptt, batch_size, self.hidden_dim)).to(device)

        # loop over sequence
        for t in torch.arange(bptt):
            xt = x[t,:,:]
            output[t], hidden_t_1 = self.forward_step(xt, hidden_t_1)

        return [output, hidden_t_1]

    def __call__(self, x, hidden_t_1):
        return self.forward_propagation(x, hidden_t_1)
