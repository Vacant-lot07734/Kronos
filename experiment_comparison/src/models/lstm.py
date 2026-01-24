import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    """
    输入:  [B, 20, 6]
    输出:  [B, 7,  6]
    """
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, dropout=0.1, pred_len=7, output_dim=6):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * output_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)        # [B,20,H]
        last = out[:, -1, :]         # [B,H]
        y = self.head(last)          # [B,7*6]
        return y.view(x.size(0), self.pred_len, self.output_dim)
