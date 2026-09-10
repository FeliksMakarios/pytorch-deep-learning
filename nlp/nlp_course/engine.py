"""Pelatihan, evaluasi, dan checkpoint yang menyimpan prapemrosesan."""
import copy
import random
from pathlib import Path
import torch
from torch import nn
from .data import encode
from .models import MeanClassifier

def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def metrics(y_true, y_pred, classes=2):
    confusion = torch.zeros(classes, classes, dtype=torch.long)
    for truth, pred in zip(y_true, y_pred):
        confusion[truth, pred] += 1
    tp = confusion.diag().float()
    precision = tp / confusion.sum(0).clamp_min(1)
    recall = tp / confusion.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {'accuracy': float(tp.sum() / confusion.sum().clamp_min(1)),
            'macro_f1': float(f1.mean()), 'confusion': confusion.tolist()}

def run_epoch(model, loader, optimizer=None, device='cpu'):
    training = optimizer is not None
    model.train(training)
    total_loss, n, truth, preds = 0., 0, [], []
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(training):
        for ids, lengths, labels in loader:
            ids, labels = ids.to(device), labels.to(device)
            logits = model(ids, lengths)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            n += len(labels)
            truth.extend(labels.cpu().tolist())
            preds.extend(logits.argmax(1).detach().cpu().tolist())
    if n == 0:
        raise ValueError('DataLoader kosong')
    return {'loss': total_loss / n, **metrics(truth, preds)}

def fit(model, train_loader, val_loader, epochs=12, lr=0.01, device='cpu'):
    if epochs < 1:
        raise ValueError('epochs harus positif')
    model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    history, best_loss, best_state = [], float('inf'), None
    for epoch in range(epochs):
        train = run_epoch(model, train_loader, optimizer, device)
        val = run_epoch(model, val_loader, device=device)
        history.append({'epoch': epoch+1, 'train_loss': train['loss'],
                        'val_loss': val['loss'], 'val_macro_f1': val['macro_f1']})
        if val['loss'] < best_loss:
            best_loss = val['loss']
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return history

def save_mean(model, vocab, path, max_length=64):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'vocab': vocab,
                'dim': model.embedding.embedding_dim, 'classes': model.head.out_features,
                'max_length': max_length, 'tokenizer': 'regex_lower_v1',
                'labels': ['negatif', 'positif']}, path)

def load_mean(path):
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if obj['tokenizer'] != 'regex_lower_v1':
        raise ValueError('Versi tokenizer tidak didukung')
    model = MeanClassifier(len(obj['vocab']), obj['dim'], obj['classes'])
    model.load_state_dict(obj['state_dict'])
    model.eval()
    return model, obj

def predict(text, model, metadata):
    if not text.strip():
        raise ValueError('Masukkan teks yang tidak kosong')
    device = next(model.parameters()).device
    ids = torch.tensor([encode(text, metadata['vocab'], metadata['max_length'])], device=device)
    lengths = torch.tensor([ids.shape[1]])
    model.eval()
    with torch.inference_mode():
        probs = model(ids, lengths).softmax(-1)[0].cpu().tolist()
    return dict(zip(metadata['labels'], probs))
