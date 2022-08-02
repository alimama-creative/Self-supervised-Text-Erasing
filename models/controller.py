

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class Controller(nn.Module):
    def __init__(self, n_subpolicies=1, embedding_dim=32, hidden_dim=100, input_dim=1000, layers=1, gen_space="random1", is_cuda=False):
        super(Controller, self).__init__() 
        self.gen_space = gen_space       
        if gen_space == "random1":
            self.num_size = [2,12,3,2,2,3,2,2,8,4,2]
            self.info = [[0,1,11],[3,4,11],[4,5,7],[7,8,11]]
        elif gen_space == "random2":
            self.num_size = [2,12,3,3,2,2,3,3,2,8,4,3]
            self.info = [[0,1,12],[4,5,12],[5,6,8],[8,9,13]]
        elif gen_space == "random3":
            self.num_size = [2,12,3,2,2,2]
            self.info = [[0,1,6],[3,4,6]]
        elif gen_space == "random4":
            self.num_size = [2,12,3,5,2,2,5,3,2,8,6,3]
            self.info = [[0,1,12],[4,5,12],[5,6,8],[8,9,13]]
        elif gen_space == "random5":
            self.num_size = [2,12,3,3,2,3,2,2,2,2,2,3,3,2,8,4,3,4]
            self.info = [[0,1,18],[9,10,18],[10,11,13],[13,14,18]]
        elif gen_space == "random7":
            self.num_size = [2,12,3,3,2,2,2,3,3,2,8,4,3,4,3]
            self.info = [[0,1,15],[5,6,15],[6,7,9],[9,10,15]]
        elif gen_space == "random8":
            self.num_size = [2,12,3,3,4,2,2,2,2,2,4,3,2,8,4,3,5,5]
            self.info = [[0,1,18],[8,9,18],[9,11,12],[12,13,18]]
        elif gen_space == "random9" or gen_space == "random6":
            self.num_size = [2,12,3,3, 2,4,2,2,2,2,2,2,4,3,2,8,4,3,5,5]
            self.info = [[0,1,20],[10,11,20],[11,12,14],[14,15,20]]
        elif gen_space=="random13":
            self.num_size = [2,12,3,3, 4,3,2,3,2,2, 2,2,4,3,2,8,4,3,5,5]
            self.info = [[0,1,20],[10,11,20],[11,12,14],[14,15,20]]
        elif gen_space=="random14":
            self.num_size = [2,12,3,5, 4,4,2,5,2,2, 2,2,5,3,2,8,6,3,5,5]
            self.info = [[0,1,20],[10,11,20],[11,12,14],[14,15,20]]

        elif gen_space == "random11":
            self.num_size = [2,12,3, 2,2,2,2,2, 2,2,2]
            self.info = [[0,1,11],[8,9,11]]
        elif gen_space == "random15":
            self.num_size = [4,3,2,2,2,2,3,3,5,2,2,4,3,2,3,8,12,4,5]
            self.info = []
        elif gen_space == "random16":
            self.num_size = [5,3,3,3,8,5,2,2,2,4,2,3,12,2,4,2,4,2,3]
            self.info = []
        elif gen_space == "random17":
            self.num_size = [2,12,3,3,4,3,2,2,2,2,2,4,3,2,8,4,3,5,5]
            self.info = []
        # self.info = []        
        self.Q = n_subpolicies
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.project = nn.Linear(input_dim, embedding_dim)
        self.tanh = nn.Tanh()
        if input_dim == 1000:
            self.pool = nn.AvgPool1d(39,31)
        self.embedding = nn.Embedding(sum(self.num_size),embedding_dim) # (# of operation) + (# of magnitude) 
        # self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers)
        self.out = nn.ModuleList([nn.Linear(hidden_dim, n) for n in self.num_size])
        
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.cnt = 1
        # self.is_cuda = True
        self.is_cuda = is_cuda
        
    def get_variable(self, inputs, cuda=False, **kwargs):
        if type(inputs) in [list, np.ndarray]:
            inputs = torch.Tensor(inputs)
        if cuda:
            out = Variable(inputs.cuda(), **kwargs)
        else:
            out = Variable(inputs, **kwargs)
        return out
    
    def create_static(self, x, batch_size):
        if batch_size != -1:
            inp = self.get_variable(torch.zeros(batch_size, self.embedding_dim), cuda = self.is_cuda, requires_grad = False)
        else:
            # inp = self.pool(x.unsqueeze(0)).squeeze(0)
            inp = self.tanh(self.project(x))
            # inp = self.project(x)
            batch_size = x.shape[0]
            # inp = self.get_variable(torch.zeros(batch_size, self.embedding_dim), cuda = self.is_cuda, requires_grad = False)

        hx = self.get_variable(torch.zeros(batch_size, self.hidden_dim), cuda = self.is_cuda, requires_grad = False)
        cx = self.get_variable(torch.zeros(batch_size, self.hidden_dim), cuda = self.is_cuda, requires_grad = False)
        # hx = self.get_variable(torch.zeros(1, batch_size, self.hidden_dim), cuda = self.is_cuda, requires_grad = False)
        # cx = self.get_variable(torch.zeros(1, batch_size, self.hidden_dim), cuda = self.is_cuda, requires_grad = False)
        return inp,hx,cx
    
    def calculate(self,logits):
        # self.temp = min(1, 10/(1+np.log(self.cnt)))
        self.temp = 1 
        probs = F.softmax(logits/self.temp, dim=-1)
        log_prob = F.log_softmax(logits/self.temp, dim=-1)
        entropy = -(log_prob * probs).sum(1, keepdim=False)
        # print(probs.shape)
        action = probs.multinomial(num_samples=1).data
        selected_log_prob = log_prob.gather(1, self.get_variable(action,requires_grad = False))
        
        return entropy, selected_log_prob[:, 0], action[:,0]
    
    def forward(self, x=None, batch_size=-1, verbose=False):
        return self.sample(x, batch_size, verbose)
    
    def sample(self, x, batch_size, verbose):
        policies = []
        entropies = []
        log_probs = []
           
        inp,hx,cx = self.create_static(x, batch_size)
        batch_size = inp.shape[0]
        # if verbose:
        #     print(x[0:3, :10])
        #     print("-----------samplinnnnnnnnng------------")
        for j in range(len(self.num_size)):
            if j > 0:
                inp = self.get_variable(sum(self.num_size[:j-1])+action, requires_grad = False)
                inp = self.embedding(inp) # B,embedding_dim
                ou, (hx, cx) = self.lstm(inp.unsqueeze(0), (hx, cx))
            else:
                ou, (hx, cx) = self.lstm(inp.unsqueeze(0))
            
            op = self.out[j](ou.squeeze(0))
            # hx, cx = self.lstm(inp, (hx, cx))
            # op = self.out[j](hx) # B,NUM_OPS
            entropy, log_prob, action = self.calculate(op)
            # if verbose:
            #     print("output of layer ",j, inp[0:3, :10], inp.shape)
            #     # print(hx[ 0:3, :10],cx[0:3, :10])
            #     print(hx[0, 0:3, :10],cx[0, 0:3, :10])
            #     print(ou[0, 0:3, :10], ou.shape)
            #     print(op[0:3])
            #     print(log_prob.shape, log_prob, action.shape, action)
            entropies.append(entropy)
            log_probs.append(log_prob)
            policies.append(action)
        entropies = torch.stack(entropies, dim = -1) ## B,Q*4
        # print(entropies.shape)
        log_probs = torch.stack(log_probs, dim = -1) ## B,Q*4
        policies = torch.stack(policies, dim = -1)
        if verbose:
            print(policies)
        mask = self.get_variable(torch.ones(batch_size, len(self.num_size)), cuda = self.is_cuda, requires_grad = False)
        for i in range(batch_size):
            for j in range(len(self.info)):
                if policies[i][self.info[j][0]] == 0:
                    mask[i][self.info[j][1]:self.info[j][2]] = 0
        entropies = mask*entropies
        log_probs = mask*log_probs  
        return policies, torch.sum(log_probs, dim = -1), torch.sum(entropies, dim = -1) # (B,Q*4) (B,) (B,)


    def print_grad(self):
        for name, param in self.named_parameters():
            print(name, param.grad)