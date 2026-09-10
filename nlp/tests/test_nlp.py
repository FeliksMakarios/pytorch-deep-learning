"""Pengujian risiko mekanisme NLP, bukan ambang akurasi sintetis."""
import tempfile
import unittest
from pathlib import Path
import torch
from nlp_course.data import build_vocab, encode, loaders, read_rows
from nlp_course.models import MeanClassifier, RecurrentClassifier, TinyTransformer
from nlp_course.engine import fit, metrics, save_mean, load_mean, predict, seed_all

class NLPTests(unittest.TestCase):
    def setUp(self):
        seed_all(42)
        torch.set_num_threads(1)
        self.vocab,self.train,self.val,self.test=loaders()
    def test_unseen_and_empty(self):
        vocab=build_vocab(['buku baik'])
        self.assertEqual(encode('tidakterlihat',vocab),[1])
        self.assertEqual(encode('',vocab),[1])
        self.assertEqual(len(encode('buku '*100,vocab,max_length=5)),5)
    def test_no_split_overlap(self):
        rows=read_rows()
        groups={s:{r['template_id'] for r in rows if r['split']==s} for s in ['train','val','test']}
        self.assertTrue(groups['train'].isdisjoint(groups['val']))
        self.assertTrue(groups['train'].isdisjoint(groups['test']))
        self.assertTrue(groups['val'].isdisjoint(groups['test']))
        expected=build_vocab(r['text'] for r in rows if r['split']=='train')
        self.assertEqual(self.vocab,expected)
    def test_padding_invariance_and_gradients(self):
        ids,lengths,labels=next(iter(self.val))
        for cls in [MeanClassifier,RecurrentClassifier,TinyTransformer]:
            with self.subTest(model=cls.__name__):
                model=cls(len(self.vocab)).eval()
                with torch.no_grad():
                    a=model(ids,lengths)
                    b=model(torch.nn.functional.pad(ids,(0,4)),lengths)
                self.assertTrue(torch.allclose(a,b,atol=1e-5))
                model.train()
                loss=torch.nn.functional.cross_entropy(model(ids,lengths),labels)
                loss.backward()
                self.assertTrue(torch.isfinite(model.embedding.weight.grad).all())
                self.assertEqual(model.embedding.weight.grad[0].abs().sum().item(),0)
    def test_metrics_known_example(self):
        result=metrics([0,0,1,1],[0,1,1,1])
        self.assertEqual(result['confusion'],[[1,1],[0,2]])
        self.assertAlmostEqual(result['accuracy'],0.75)
        self.assertAlmostEqual(result['macro_f1'],(2/3+0.8)/2,places=6)
    def test_checkpoint_and_input_contract(self):
        model=MeanClassifier(len(self.vocab))
        fit(model,self.train,self.val,epochs=2)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt'
            save_mean(model,self.vocab,path)
            restored,meta=load_mean(path)
            model.eval()
            ids,lengths,_=next(iter(self.val))
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(ids,lengths),restored(ids,lengths)))
            result=predict('buku baik',restored,meta)
            self.assertAlmostEqual(sum(result.values()),1,places=6)
            with self.assertRaises(ValueError): predict(' ',restored,meta)
    def test_freeze_preserves_embeddings(self):
        model=MeanClassifier(len(self.vocab))
        before=model.embedding.weight.detach().clone()
        model.embedding.weight.requires_grad_(False)
        fit(model,self.train,self.val,epochs=2)
        self.assertTrue(torch.equal(before,model.embedding.weight))

if __name__=='__main__': unittest.main()
