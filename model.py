import torch
import torch.nn as nn

class PredictiveKeyboard(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256):
        super(PredictiveKeyboard, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out