# ==================================================================================================================
# Base LSTM model (bidirectional, pad-aware)
# ==================================================================================================================

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ChordLSTM(nn.Module):
    """
    LSTM-based model for chord prediction.
    
    Args:
        vocab_size (int): Size of the chord vocabulary.
        embedding_dim (int): Dimension of the chord embeddings.
        hidden_size (int): Number of features in the hidden state of the LSTM.
        num_layers (int): Number of recurrent layers.
        dropout (float): Dropout probability.
        padding_idx (int): Index of the padding token (default: 0).
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.5, padding_idx=0):
        super(ChordLSTM, self).__init__()

        self.pad_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx, scale_grad_by_freq=True, max_norm=1.0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        mask = x.ne(self.pad_idx)
        lengths = mask.sum(dim=1).clamp(min=1).cpu()

        embeds = self.embedding(x)  # (B, T, E)
        packed = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))

        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)  # (B, T, V)
        return out