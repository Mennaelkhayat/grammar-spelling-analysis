"""Gradio interface combining grammar classification and spelling correction.

Run:
    python app.py

Set GRAMMAR_MODEL_DIR to a fine-tuned checkpoint (see src/grammar_classifier.py).
Without it, the app falls back to a pretrained RoBERTa CoLA checkpoint from the
Hub so the demo works out of the box.
"""

import os

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.spelling_corrector import SpellingCorrector

MODEL_DIR = os.environ.get("GRAMMAR_MODEL_DIR", "textattack/roberta-base-CoLA")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
corrector = SpellingCorrector()

LABELS = ["unacceptable", "acceptable"]


def analyze(sentence):
    corrected = corrector.correct_text(sentence)
    inputs = tokenizer(corrected, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    verdict = LABELS[probs.argmax().item()]
    confidence = probs.max().item()
    return corrected, f"{verdict} (confidence: {confidence:.1%})"


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(label="Sentence", placeholder="Type a sentence with mistakes..."),
    outputs=[
        gr.Textbox(label="Spelling-corrected sentence"),
        gr.Textbox(label="Grammar verdict (RoBERTa/CoLA)"),
    ],
    title="Grammar and Spelling Analysis System",
    description=(
        "Rule-based spelling correction plus a RoBERTa grammar acceptability "
        "classifier fine-tuned on CoLA."
    ),
)

if __name__ == "__main__":
    demo.launch()
