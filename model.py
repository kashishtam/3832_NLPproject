import torch
import torch.nn as nn
import torch.nn.functional as F

class SarcasmDetector(nn.Module):
    # code inspired from https://github.com/AbdelkaderMH/iSarcasmEval/blob/main/modeling2.py
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SarcasmDetector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2) #dropout layers to pervent overfitting
        self.dropout2 = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out
    