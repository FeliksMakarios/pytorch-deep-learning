"""Antarmuka lokal. Instal requirements-app.txt, latih model, lalu python app.py."""
from pathlib import Path
import gradio as gr
from nlp_course.engine import load_mean, predict

checkpoint = Path(__file__).parent / 'artifacts/sentiment.pt'
if not checkpoint.exists():
    raise SystemExit('Latih model dahulu: python -m nlp_course.train')
model, metadata = load_mean(checkpoint)

def classify(text):
    if not text.strip():
        raise gr.Error('Masukkan teks terlebih dahulu.')
    return predict(text, model, metadata)

if __name__ == '__main__':
    gr.Interface(fn=classify, inputs=gr.Textbox(label='Ulasan'),
                 outputs=gr.Label(label='Skor sentimen'),
                 title='Latihan klasifikasi sentimen',
                 description='Model pembelajaran dengan data sintetis. Skor belum dikalibrasi.',
                 examples=['layanan ini sangat baik', 'produk ini mengecewakan']).launch(share=False)
