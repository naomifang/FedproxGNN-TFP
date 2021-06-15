import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        
        self.out = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x):
        '''
            x: [src_len, batch_size, embedding]
        '''
        _, hid = self.gru(x)
        
        output = self.out(hid)
        
        return output