# Grammar and Spelling Analysis System

An NLP pipeline that analyzes English sentences for correctness in two stages:

1. **Rule-based spelling correction** - a noisy-channel corrector (Norvig-style):
   candidate words are generated with edit-distance-1/2 rules (deletes, transposes,
   replaces, inserts) and ranked by word frequency. No ML - pure rules plus a
   frequency table.
2. **Transformer grammar classification** - `roberta-base` fine-tuned on the CoLA
   corpus (Corpus of Linguistic Acceptability) to predict whether a sentence is
   grammatically acceptable, with a confidence score.

Both stages are exposed through a Gradio web interface: type a sentence (typos and
all), get back the corrected sentence and a grammar verdict.

## Project structure

```
├── app.py                      # Gradio app combining both stages
├── data/
│   └── wordlist.txt            # 20k most common English words (frequency-ranked)
├── src/
│   ├── grammar_classifier.py   # RoBERTa fine-tuning on CoLA (train + eval)
│   └── spelling_corrector.py   # rule-based spelling corrector
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt

# Launch the interface (uses a pretrained RoBERTa CoLA checkpoint by default)
python app.py
```

### Fine-tune your own grammar classifier

```bash
python -m src.grammar_classifier --epochs 2 --output-dir checkpoints/grammar
GRAMMAR_MODEL_DIR=checkpoints/grammar python app.py
```

Training evaluates on the CoLA validation split and reports accuracy and the
Matthews correlation coefficient (CoLA's official metric).

### Spelling corrector standalone

```bash
python -m src.spelling_corrector
# -> "This sentence has some spelling errors."
```

The default vocabulary is `data/wordlist.txt` - the 20,000 most common English
words (derived from Google's Trillion Word Corpus, via the
`first20hours/google-10000-english` list), ranked by frequency. For full
vocabulary coverage, pass a word-frequency file (one `word count` pair per line,
such as Norvig's `big.txt`):

```python
from src.spelling_corrector import SpellingCorrector
corrector = SpellingCorrector(frequency_file="big.txt")
```

## Notes

- The default demo checkpoint is `textattack/roberta-base-CoLA` from the Hugging
  Face Hub; `src/grammar_classifier.py` reproduces an equivalent fine-tune.
- Spelling correction is intentionally rule-based: it is fast, transparent, and
  its mistakes are explainable, which made it a good contrast to the learned
  grammar classifier in this project.
