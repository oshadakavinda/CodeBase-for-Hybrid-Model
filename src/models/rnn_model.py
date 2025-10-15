# rnn_model.py
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, num_classes=454, hidden_size=256):
        super(RNNModel, self).__init__()
        # Treat image rows as sequence
        self.rnn = nn.LSTM(
            input_size=80,      # Process each row (80 pixels)
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch_size, 1, 80, 80)
        batch_size = x.size(0)
        x = x.squeeze(1)  # (batch_size, 80, 80) - treat as 80 rows
        x, (h, c) = self.rnn(x)  # Process row by row
        x = h[-1]  # Take last hidden state: (batch_size, hidden_size)
        x = self.fc(x)  # (batch_size, num_classes)
        return x