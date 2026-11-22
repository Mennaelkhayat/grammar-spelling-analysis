"""Transformer-based grammar acceptability classifier (RoBERTa).

Fine-tunes `roberta-base` on CoLA (Corpus of Linguistic Acceptability) to
predict whether an English sentence is grammatically acceptable.

Usage:
    python -m src.grammar_classifier --epochs 2 --output-dir checkpoints/grammar
"""

import argparse

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "roberta-base"


def tokenize_dataset(tokenizer):
    """Load CoLA and tokenize it."""
    dataset = load_dataset("glue", "cola")

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=128)

    return dataset.map(tokenize, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    # Matthews correlation is CoLA's official metric
    tp = ((predictions == 1) & (labels == 1)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {"accuracy": accuracy, "matthews_correlation": mcc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", default="checkpoints/grammar")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = tokenize_dataset(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved grammar classifier to {args.output_dir}/")


if __name__ == "__main__":
    main()
