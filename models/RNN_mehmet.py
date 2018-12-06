
from torch import nn
import torch.nn.functional as F



class RNN_mehmet(nn.Module):
    def __init__(self, word_dim, hidden_dim = 100, activation = 'sigmoid'):
        super(RNN_mehmet, self).__init__()

        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.weights_hh = self.init_weights((hidden_dim, hidden_dim))
        self.weights_xh = self.init_weights((hidden_dim, word_dim))
        self.weights_o = self.init_weights((word_dim, hidden_dim))
        self.activation = getattr(torch, activation)



    def init_weights(self, dim):
        return nn.Parameter(torch.FloatTensor(dim[0], dim[1]).uniform_(-np.sqrt(1./dim[0]), np.sqrt(1./dim[1])), requires_grad=True)


    def init_hidden(self, batch_size, dim):

        layer = torch.zeros((1, batch_size, self.hidden_dim),  requires_grad = True)
        if dim > 1:
           layer = (layer.clone(), layer.clone())
        return layer

    def step(self, lr):
        for p in self.parameters():
            p.data.add_(-lr, p.grad.data)

    def forward_step(self, xt, hidden_t_1):

        # calculate left and right terms
        left_term = F.linear(xt, self.weights_xh)
        right_term = F.linear(hidden_t_1, self.weights_hh)

        # sum terms
        sum_ = left_term + right_term

        # activation for hidden state
        hidden_t = self.activation(sum_)

        # calculate output
        output = F.linear(hidden_t, self.weights_o)
        return output, hidden_t

    def forward_propagation(self, x, hidden_t_1):
        # Get sequence length (bptt), batch_size from the input
        bptt, batch_size, _ = x.size()
        output = torch.zeros((bptt, batch_size, self.word_dim)).to(device)

        # loop over sequence
        for t in torch.arange(bptt):
            xt = x[t,:,:]
            output[t], hidden_t_1 = self.forward_step(xt, hidden_t_1)
        return [output, hidden_t_1]

    def __call__(self, x, hidden_t_1):
        return self.forward_propagation(x, hidden_t_1)
