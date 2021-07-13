import torch.nn as nn
import torch
from . import utils

@utils.register_model(name='fcn')
class FCN(nn.Module):
    def __init__(self, config): 
        super(FCN, self).__init__()
        state_size = config.model.state_size
        hidden_layers = config.model.hidden_layers
        hidden_nodes = config.model.hidden_nodes
        dropout = config.model.dropout

        input_size = state_size + 1 #+1 because of the time dimension.
        output_size = state_size

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(input_size, hidden_nodes))
        self.mlp.append(nn.Dropout(dropout)) #addition
        self.mlp.append(nn.ReLU())

        for _ in range(hidden_layers):
            self.mlp.append(nn.Linear(hidden_nodes, hidden_nodes))
            self.mlp.append(nn.Dropout(dropout)) #addition
            self.mlp.append(nn.ReLU())
        
        self.mlp.append(nn.Linear(hidden_nodes, output_size))
        self.mlp = nn.Sequential(*self.mlp)
             
    def forward(self, x, t):
        x_shape = x.shape
        t_shape = t.shape

        if len(x_shape)==2 and len(t_shape)==0:
            #x_shape = (batchsize, state_size) --> sampling process - reverse SDE
            t = t.repeat((x_shape[0],1))
            x = torch.cat([x, t], dim=1)
            x = self.mlp(x)
            return x

        elif len(x_shape)==3 and len(t_shape)==1:
            #x_shape = (time_samples, batch_size, state_size) -->training process
            aug_t = []
            for i in range(t_shape[0]):
                aug_t.append(t[i].repeat((x_shape[1], 1)))
            aug_t = torch.stack(aug_t)

            out=[]
            for i in range(x_shape[0]):
                y = torch.cat([x[i], aug_t[i]], dim=1) #(batch_size, state_size+1)
                out.append(y)

            out = torch.cat(out, dim=0) #(time_samples*batch_size, state_size+1)
            out = self.mlp(out) #(time_samples*batch_size, output_size)
            out = torch.chunk(out, x_shape[0])
            out = torch.stack(out)
            return out
        else:
            raise NotImplementedError




        