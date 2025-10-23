# models/seq_model.py
import torch.nn as nn

class SeqModel(nn.Module):
    def __init__(self, inp_dim: int, hidden_dim: int = 64, n_layers: int = 1, rnn_type: str = "lstm", dropout: float = 0.2):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(inp_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers>1 else 0.0)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(inp_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers>1 else 0.0)
        else:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, max(16, hidden_dim//2)),
            nn.ReLU(),
            nn.Linear(max(16, hidden_dim//2), 1)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last)
