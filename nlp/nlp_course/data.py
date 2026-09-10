"""Data teks: kosakata hanya dipelajari dari bagian latih."""
import csv
import re
from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD, UNK = 0, 1
ROOT = Path(__file__).resolve().parents[1]

def tokenize(text):
    # Negasi dan kata pendek dipertahankan. Tokenisasi sederhana untuk pembelajaran.
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)

def build_vocab(texts, min_freq=1):
    counts = Counter(token for text in texts for token in tokenize(text))
    vocab = {'<pad>': PAD, '<unk>': UNK}
    for token, count in sorted(counts.items()):
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab

def encode(text, vocab, max_length=64):
    if max_length < 1:
        raise ValueError('max_length harus positif')
    return [vocab.get(t, UNK) for t in tokenize(text)[:max_length]] or [UNK]

def read_rows(path=None, domain=None):
    with open(path or ROOT / 'data/reviews.csv', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    required = {'text', 'label', 'split'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('CSV harus mempunyai kolom text,label,split dan setidaknya satu baris')
    seen = set()
    for row in rows:
        row['label'] = int(row['label'])
        if row['label'] not in (0, 1) or row['split'] not in ('train', 'val', 'test'):
            raise ValueError('label harus 0/1 dan split harus train/val/test')
        key = ' '.join(tokenize(row['text']))
        if not key or key in seen:
            raise ValueError('Teks kosong atau duplikat ditemukan')
        seen.add(key)
    return [r for r in rows if domain is None or r.get('domain') == domain]

class TextDataset(Dataset):
    def __init__(self, rows, vocab, max_length=64):
        self.rows, self.vocab, self.max_length = rows, vocab, max_length
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        return torch.tensor(encode(row['text'], self.vocab, self.max_length)), row['label']

def collate_batch(batch):
    tokens, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in tokens], dtype=torch.long)
    return pad_sequence(tokens, batch_first=True, padding_value=PAD), lengths, torch.tensor(labels)

def loaders(rows=None, vocab=None, batch_size=16, seed=42):
    rows = read_rows() if rows is None else rows
    parts = {s: [r for r in rows if r['split'] == s] for s in ('train','val','test')}
    if any(not x for x in parts.values()):
        raise ValueError('Setiap split harus berisi data')
    vocab = build_vocab(r['text'] for r in parts['train']) if vocab is None else vocab
    result = [DataLoader(TextDataset(parts[s], vocab), batch_size=batch_size,
                         shuffle=s == 'train', collate_fn=collate_batch,
                         generator=torch.Generator().manual_seed(seed)) for s in parts]
    return vocab, *result
