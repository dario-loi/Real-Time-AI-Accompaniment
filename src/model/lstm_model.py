# ==================================================================================================================
# Base LSTM model
# ==================================================================================================================

import torch.nn as nn

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
        
        # padding_idx=0 means the vector at index 0 will be all zeros and won't be updated
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embeds = self.embedding(x)              # shape (batch_size, seq_len, embedding_dim)
        
        # LSTM returns: output, (h_n, c_n)
        lstm_out, _ = self.lstm(embeds)         # shape (batch_size, seq_len, hidden_size)
        
        # last output for prediction
        last_out = lstm_out[:, -1, :]           # shape (batch_size, hidden_size)
        
        last_out = self.dropout(last_out)
        out = self.fc(last_out)                 # shape (batch_size, vocab_size)
        return out