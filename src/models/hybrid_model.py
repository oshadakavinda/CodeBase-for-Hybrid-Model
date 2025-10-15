import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, num_classes, input_channels=1, hidden_size=128, dropout_rate=0.5):
        super(HybridModel, self).__init__()
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 40x40
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 20x20
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 10x10
            nn.Dropout2d(0.25),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 5x5
        )
        
        # LSTM layer for temporal processing
        # Input: CNN features flattened (256*5*5 = 6400)
        # Treat as sequence of length 1 for classification
        self.lstm = nn.LSTM(
            input_size=256 * 5 * 5,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate if hidden_size > 0 else 0,
            bidirectional=True
        )
        
        # Classification head
        lstm_output_size = hidden_size * 2  # bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # CNN feature extraction
        # x: (batch_size, 1, 80, 80)
        x = self.cnn(x)  # (batch_size, 256, 5, 5)
        
        # Flatten for LSTM
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1)  # (batch_size, 1, 256*5*5)
        
        # LSTM processing
        x, (h_n, c_n) = self.lstm(x)  # (batch_size, 1, lstm_hidden_size*2)
        
        # Take last hidden state
        x = x[:, -1, :]  # (batch_size, lstm_hidden_size*2)
        
        # Classification
        x = self.classifier(x)  # (batch_size, num_classes)
        
        return x