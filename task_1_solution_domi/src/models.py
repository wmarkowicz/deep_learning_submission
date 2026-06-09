import torch
from torch import nn


class OneHeadDeepSTARR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=7, padding=3), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 60, kernel_size=3, padding=1), nn.BatchNorm1d(60), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(60, 60, kernel_size=5, padding=2), nn.BatchNorm1d(60), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(60, 120, kernel_size=3, padding=1), nn.BatchNorm1d(120), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.flatten = nn.Flatten()

        self.dense = nn.Sequential(
            nn.Linear(120 * 12, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.4)
        )

        self.regression_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.regression_head(x)
    
class OneHeadDeepSTARRWithAdditionalDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=7, padding=3), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.Conv1d(256, 60, kernel_size=3, padding=1), nn.BatchNorm1d(60), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.Conv1d(60, 60, kernel_size=5, padding=2), nn.BatchNorm1d(60), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.Conv1d(60, 120, kernel_size=3, padding=1), nn.BatchNorm1d(120), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.1),
        )
        self.flatten = nn.Flatten()

        self.dense = nn.Sequential(
            nn.Linear(120 * 12, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.4)
        )

        self.regression_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.regression_head(x)
    
class OneHeadDeepSTARRWithLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=7, padding=3), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 60, kernel_size=3, padding=1), nn.BatchNorm1d(60), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(60, 60, kernel_size=5, padding=2), nn.BatchNorm1d(60), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(60, 120, kernel_size=3, padding=1), nn.BatchNorm1d(120), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=120, hidden_size=128, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(128 * 12, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.4)
        )
        self.regression_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.dense(x)
        return self.regression_head(x)
    
class OneHeadDeepSTARRWithLSTMAndAdditionalDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=7, padding=3), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.Conv1d(256, 60, kernel_size=3, padding=1), nn.BatchNorm1d(60), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.Conv1d(60, 60, kernel_size=5, padding=2), nn.BatchNorm1d(60), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.Conv1d(60, 120, kernel_size=3, padding=1), nn.BatchNorm1d(120), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.1),
        )
        self.lstm = nn.LSTM(input_size=120, hidden_size=128, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(128 * 12, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.4)
        )
        self.regression_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.dense(x)
        return self.regression_head(x)
    
class OneHeadDeepSTARRWithLSTMAndAdditionalDropoutGELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=7, padding=3), nn.BatchNorm1d(256), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.Conv1d(256, 60, kernel_size=3, padding=1), nn.BatchNorm1d(60), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.Conv1d(60, 60, kernel_size=5, padding=2), nn.BatchNorm1d(60), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.Conv1d(60, 120, kernel_size=3, padding=1), nn.BatchNorm1d(120), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.1),
        )
        self.lstm = nn.LSTM(input_size=120, hidden_size=128, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(128 * 12, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.4)
        )
        self.regression_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.dense(x)
        return self.regression_head(x)

class OneHeadDeepSTARRWithLSTMAndAdditionalBiggerDropoutGELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=7, padding=3), nn.BatchNorm1d(256), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(256, 60, kernel_size=3, padding=1), nn.BatchNorm1d(60), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(60, 60, kernel_size=5, padding=2), nn.BatchNorm1d(60), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(60, 120, kernel_size=3, padding=1), nn.BatchNorm1d(120), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.2),
        )
        self.lstm = nn.LSTM(input_size=120, hidden_size=128, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(128 * 12, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.4)
        )
        self.regression_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.dense(x)
        return self.regression_head(x)
    
class OneHeadDeepSTARRWithLSTMAndAdditionalSmallerDropoutGELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=7, padding=3), nn.BatchNorm1d(256), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.Conv1d(256, 60, kernel_size=3, padding=1), nn.BatchNorm1d(60), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.Conv1d(60, 60, kernel_size=5, padding=2), nn.BatchNorm1d(60), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.Conv1d(60, 120, kernel_size=3, padding=1), nn.BatchNorm1d(120), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.1),
        )
        self.lstm = nn.LSTM(input_size=120, hidden_size=128, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(128 * 12, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.3)
        )
        self.regression_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.dense(x)
        return self.regression_head(x)

class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))

class DeepSTARRwithResidualDilatedBlockAndLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.GELU(),
            ResidualDilatedBlock(256, kernel_size=7, dilation=1, dropout=0.1),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(256, 60, kernel_size=3, padding=1),
            nn.BatchNorm1d(60),
            nn.GELU(),
            ResidualDilatedBlock(60, kernel_size=3, dilation=1, dropout=0.1),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(60, 60, kernel_size=5, padding=2),
            nn.BatchNorm1d(60),
            nn.GELU(),
            ResidualDilatedBlock(60, kernel_size=5, dilation=2, dropout=0.1),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(60, 120, kernel_size=3, padding=1),
            nn.BatchNorm1d(120),
            nn.GELU(),
            ResidualDilatedBlock(120, kernel_size=3, dilation=2, dropout=0.1),
            ResidualDilatedBlock(120, kernel_size=3, dilation=4, dropout=0.1),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
        )

        self.lstm = nn.LSTM(
            input_size=120,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.dense = nn.Sequential(
            nn.Linear(128 * 12, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.4)
        )

        self.regression_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.dense(x)
        return self.regression_head(x)

class SEB(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.pool(x).squeeze(-1)
        scale = self.fc(scale).unsqueeze(-1)
        return x * scale
    
class ResidualDilatedBlockSEB(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        assert kernel_size % 2 == 1
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.se = SEB(channels)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.se(self.block(x)))

class DeepSTARRRDSEB(nn.Module):
    def __init__(self, dropout_prob=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.GELU(),
            ResidualDilatedBlockSEB(256, kernel_size=7, dilation=1, dropout=0.1),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(256, 60, kernel_size=3, padding=1),
            nn.BatchNorm1d(60),
            nn.GELU(),
            ResidualDilatedBlockSEB(60, kernel_size=3, dilation=1, dropout=0.1),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(60, 60, kernel_size=5, padding=2),
            nn.BatchNorm1d(60),
            nn.GELU(),
            ResidualDilatedBlockSEB(60, kernel_size=5, dilation=2, dropout=0.1),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(60, 120, kernel_size=3, padding=1),
            nn.BatchNorm1d(120),
            nn.GELU(),
            ResidualDilatedBlockSEB(120, kernel_size=3, dilation=2, dropout=0.1),
            ResidualDilatedBlockSEB(120, kernel_size=3, dilation=4, dropout=0.1),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
        )

        self.lstm = nn.LSTM(
            input_size=120,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )

        self.dense = nn.Sequential(
            nn.Linear(128 * 12, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.4)
        )

        self.regression_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.dense(x)
        return self.regression_head(x)
