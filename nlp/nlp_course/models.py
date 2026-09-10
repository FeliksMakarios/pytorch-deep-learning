"""Model klasifikasi teks dengan antarmuka forward(ids, lengths)."""
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class MeanClassifier(nn.Module):
    def __init__(self, vocab_size, dim=32, classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.head = nn.Linear(dim, classes)
    def forward(self, ids, lengths):
        vectors = self.embedding(ids)
        mask = ids.ne(0).unsqueeze(-1)
        pooled = (vectors * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(pooled)

class RecurrentClassifier(nn.Module):
    def __init__(self, vocab_size, dim=32, hidden=32, kind='lstm', classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        recurrent = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[kind]
        self.rnn = recurrent(dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, classes)
    def forward(self, ids, lengths):
        packed = pack_padded_sequence(self.embedding(ids), lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        if isinstance(hidden, tuple):
            hidden = hidden[0]  # LSTM mengembalikan (h_n, c_n)
        return self.head(hidden[-1])

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, dim=32, heads=4, classes=2, max_length=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.position = nn.Embedding(max_length, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=dim*2,
                                           dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Linear(dim, classes)
        self.scale = math.sqrt(dim)
    def forward(self, ids, lengths):
        if ids.shape[1] > self.position.num_embeddings:
            raise ValueError('Urutan melampaui kapasitas posisi model')
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.embedding(ids) * self.scale + self.position(pos)
        x = self.encoder(x, src_key_padding_mask=ids.eq(0))
        mask = ids.ne(0).unsqueeze(-1)
        return self.head((x * mask).sum(1) / mask.sum(1).clamp_min(1))
