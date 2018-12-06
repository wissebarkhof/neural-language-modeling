
# coding: utf-8

# In[1]:

import itertools
import operator
from datetime import datetime
import sys
from torch import FloatTensor
from torch.autograd import Variable
import torch

vocabulary_size = 8000


class RNN_mehmet():
    def __init__(self, word_dim, hidden_dim = 100, T=5):
        self.T = T
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        
        self.U = Variable(torch.FloatTensor(hidden_dim, word_dim).uniform_(-np.sqrt(1./word_dim), np.sqrt(1./word_dim)), requires_grad=True)
        self.V = Variable(torch.FloatTensor(word_dim, hidden_dim).uniform_(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim)), requires_grad=True)
        self.W = Variable(torch.FloatTensor(hidden_dim, hidden_dim).uniform_(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim)), requires_grad=True)


    def softmax(self, x):
        x = x.clone()
        xt = torch.exp(x.clone() - torch.max(x))
        return xt / xt.sum()
      

    def forward_propagation(self, x):
        T = self.T
        s = torch.zeros((T+1, self.hidden_dim))
        s[-1] = torch.zeros(self.hidden_dim) 
        o = torch.zeros((T, self.word_dim))
        for t in torch.arange(T):           
            U = self.U.clone()
            W = self.W.clone()
            V = self.V.clone()
            left_term = U[:, int(x[t])]
            right_term = torch.mv(W, s[t-1] )
            sum_ = left_term + right_term
            s[t] = nn.functional.tanh(sum_)
            o[t] = self.softmax(torch.mv(V, s[t]))
        return [o, s]

fake_net = RNN_mehmet(vocabulary_size, hidden_dim = 130, T=10)

#forward_prop
some_hot_encoded_input = torch.zeros(vocabulary_size)
some_hot_encoded_input[2] = 0.4
result = fake_net.forward_propagation(some_hot_encoded_input)




#example backward and one step
loss = Variable(result[0][0].mean(), requires_grad = True)
loss.backward()
print(fake_net.U, loss.grad)
fake_net.U = fake_net.U - 0.001 * loss.grad
print(fake_net.U, loss.grad)



