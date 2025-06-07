import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attention(nn.Module):
    def __init__(self, filter_size,num_heads):
        super().__init__()  
        
        
        self.filter_size = filter_size
        self.num_heads = num_heads
        self.q_k_dim = filter_size//self.num_heads
        
        self.query = nn.Linear(filter_size, self.q_k_dim*self.num_heads)
        self.key = nn.Linear(filter_size, self.q_k_dim*self.num_heads)
        self.value = nn.Linear(filter_size, self.q_k_dim*self.num_heads)

        self.out_linear = nn.Linear(self.q_k_dim*self.num_heads, filter_size)
        self.lnorm = nn.LayerNorm(self.filter_size)
        
        self.dropout = nn.Dropout(.25)
  
    def forward(self,x):
        
        batch_size = x.data.size()[0]
        time_steps = x.data.size()[1]
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        shape = q.shape
        q = q.view(shape[0], shape[1], self.q_k_dim, self.num_heads)
        q = q.permute(0,3,1,2)
        q = q.reshape(shape[0]*self.num_heads, shape[1],self.q_k_dim)
        shape = k.shape
        k = k.view(shape[0], shape[1], self.q_k_dim, self.num_heads)
        k = k.permute(0,3,1,2)
        k = k.reshape(shape[0]*self.num_heads, shape[1],self.q_k_dim)
        shape = v.shape
        v = v.view(shape[0], shape[1], self.q_k_dim, self.num_heads)
        v = v.permute(0,3,1,2)
        v = v.reshape(shape[0]*self.num_heads, shape[1],self.q_k_dim)
        attention_weights = torch.bmm(q,k.permute(0,2,1))
        attention_weights = attention_weights/ np.sqrt(self.q_k_dim)
        attention_weights = F.softmax(attention_weights,dim = -1)
        attention_weights = self.dropout(attention_weights)
        out = torch.bmm(attention_weights,v)
        out = out.reshape(batch_size , self.num_heads,time_steps, self.q_k_dim)
        out = out.permute(0,2,3,1)
        out = out.reshape(batch_size,time_steps, self.q_k_dim*self.num_heads)
        out = self.out_linear(out)
        out = out + x
        out =self.lnorm(out)
        return out
    
class LAFS(nn.Module):
    def __init__(self, n_channels,sequence_length,k = 20, lafs_heads = 4,attention_heads = 1):
        super().__init__()  

        self.n_channels = n_channels
        self.k = k
        self.heads =lafs_heads


        self.ff_after_time_att = nn.Sequential(
            nn.Linear(self.k,self.k),
            nn.ReLU()
        )
        self.ff_after_channel_att = nn.Sequential(
            nn.Linear(n_channels,n_channels//2),
            nn.ReLU(),
            nn.Linear(n_channels//2,n_channels)
        )
        
        self.get_weights = nn.Sequential(
                                     nn.Linear(self.k,self.heads)
                                    )
        # self.weights[0].weight.data = torch.zeros((self.weights[0].weight.data.shape))
        # self.weights[0].bias.data = torch.ones((n_channels*self.heads))/n_channels
        self.attention_across_time = Attention(n_channels,attention_heads)

        self.attention_across_channels = Attention(sequence_length,attention_heads)

        
    def gather_channels(self,x, weights):
        x = x.unsqueeze(2)
        x = x.repeat(1,1,self.heads,1)
        weights = weights
        x = x*weights
        x = torch.sum(x,dim = -1)
        weights = torch.mean(weights,dim = 2)
        return x,weights

    def forward(self,x,training):
        ## in shape = (B, T, C)
        weights  = self.attention_across_time(x)  
        # weights=  self.fc(weights)
        weights = weights.permute(0,2,1)

        weights_sum = torch.mean(weights*weights,dim= 1,keepdims = True)
        dis,ind = torch.topk(weights_sum,dim = -1,k=self.k)
        ind = ind.repeat(1,weights.shape[1],1)
        weights = torch.gather(weights,dim = -1,index = ind)

        weights = self.ff_after_time_att(weights)        

        weights  = self.attention_across_channels(weights)   
        
        weights = weights.permute(0,2,1)
        weights = self.ff_after_channel_att(weights)
        weights = weights.permute(0,2,1)
        
        weights = self.get_weights(weights)
        weights = weights.reshape(weights.shape[0],weights.shape[1],self.heads,1)
        weights = torch.relu(weights)
        weights = weights.permute(0,3,2,1)

        x,weights = self.gather_channels(x, weights)
        return x, weights