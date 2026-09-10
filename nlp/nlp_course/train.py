"""Jalankan dari direktori nlp: python -m nlp_course.train --epochs 12."""
import argparse
import json
import torch
from .data import read_rows, loaders
from .models import MeanClassifier
from .engine import seed_all, fit, run_epoch, save_mean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=None)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='artifacts/sentiment.pt')
    args = parser.parse_args()
    torch.set_num_threads(1)
    seed_all(args.seed)
    vocab, train, val, test = loaders(read_rows(args.csv), seed=args.seed)
    model = MeanClassifier(len(vocab))
    history = fit(model, train, val, epochs=args.epochs)
    save_mean(model, vocab, args.output)
    print(json.dumps({'history': history, 'test': run_epoch(model, test)}, indent=2))

if __name__ == '__main__':
    main()
