# -*- coding: utf-8 -*-
"""BART-Large-CNN summarization on XSum with before/after fine-tuning evaluation.

This mirrors the existing `run_bart.py` workflow:
- evaluate the `facebook/bart-large-cnn` checkpoint first on the full XSum test split
- fine-tune that checkpoint on the XSum train split with the same BART hyperparameters
- evaluate again on the XSum test split after fine-tuning

Unlike `run_bart.py`, this script keeps the XSum experiment isolated so the
existing CNN/DailyMail checkpoints remain unchanged.
"""

import importlib.util
import json
from pathlib import Path

import nltk
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm.auto import tqdm

# XSum is text-only, but some local Python environments fail while importing
# `transformers` or `datasets` because of an optional `soundfile` package issue.
_original_find_spec = importlib.util.find_spec


def _find_spec_without_soundfile(name, package=None):
    if name == "soundfile":
        return None
    return _original_find_spec(name, package)


importlib.util.find_spec = _find_spec_without_soundfile
try:
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    from datasets import load_dataset
finally:
    importlib.util.find_spec = _original_find_spec


ROOT = Path(__file__).resolve().parent
DATASET_NAME = "EdinburghNLP/xsum"
MODEL_NAME = "facebook/bart-large-cnn"
SOURCE_FIELD = "document"
TARGET_FIELD = "summary"
INPUT_MAX_LENGTH = 1024
SUMMARY_MAX_LENGTH = 128
NUM_EPOCHS = 2
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
LEARNING_RATE = 3e-5
WARMUP_STEPS = 2000
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100
SAVE_STEPS = 1000
EVAL_STEPS = 1000
NUM_BEAMS = 4
LENGTH_PENALTY = 2.0

CHECKPOINT_DIR = ROOT / "checkpoints_bart_large_cnn_xsum"
LOG_DIR = ROOT / "logs_bart_large_cnn_xsum"
METRICS_DIR = ROOT / "xsum_bart_large_cnn_results"
PRE_PREDICTIONS_PATH = ROOT / "bart_large_cnn_xsum_pre_predictions.txt"
PRE_REFERENCES_PATH = ROOT / "bart_large_cnn_xsum_pre_references.txt"
POST_PREDICTIONS_PATH = ROOT / "bart_large_cnn_xsum_post_predictions.txt"
POST_REFERENCES_PATH = ROOT / "bart_large_cnn_xsum_post_references.txt"
METRICS_PATH = METRICS_DIR / "bart_large_cnn_xsum_metrics.json"


def ensure_meteor_resources():
    for resource_path, resource_name in (
        ("tokenizers/punkt", "punkt"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ):
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(str(line).strip() + "\n")


def safe_meteor(reference, prediction):
    reference_tokens = str(reference).split()
    prediction_tokens = str(prediction).split()
    if not reference_tokens or not prediction_tokens:
        return 0.0
    return meteor_score([reference_tokens], prediction_tokens)


def evaluate_model(
    model,
    tokenizer,
    dataset,
    device,
    predictions_path,
    references_path,
    model_label="Model",
):
    scorer_obj = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    smoother = SmoothingFunction()

    rouge_scores = []
    bleu_scores = []
    meteor_scores = []
    predictions = []
    references = []

    model.eval()
    with torch.no_grad():
        for document, reference in tqdm(
            zip(dataset[SOURCE_FIELD], dataset[TARGET_FIELD]),
            total=len(dataset),
            desc=f"Evaluating {model_label}",
        ):
            inputs = tokenizer(
                document,
                return_tensors="pt",
                max_length=INPUT_MAX_LENGTH,
                truncation=True,
            ).to(device)

            summary_ids = model.generate(
                **inputs,
                max_length=SUMMARY_MAX_LENGTH,
                num_beams=NUM_BEAMS,
                length_penalty=LENGTH_PENALTY,
                early_stopping=True,
            )
            prediction = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            predictions.append(prediction)
            references.append(reference)
            rouge_scores.append(scorer_obj.score(reference, prediction))
            bleu_scores.append(
                sentence_bleu(
                    [reference.split()],
                    prediction.split(),
                    smoothing_function=smoother.method1,
                )
            )
            meteor_scores.append(safe_meteor(reference, prediction))

    write_lines(predictions_path, predictions)
    write_lines(references_path, references)

    metrics = {
        "rouge1": sum(s["rouge1"].fmeasure for s in rouge_scores) / len(rouge_scores),
        "rouge2": sum(s["rouge2"].fmeasure for s in rouge_scores) / len(rouge_scores),
        "rougeL": sum(s["rougeL"].fmeasure for s in rouge_scores) / len(rouge_scores),
        "bleu": sum(bleu_scores) / len(bleu_scores),
        "meteor": sum(meteor_scores) / len(meteor_scores),
    }

    print(f"\n{'=' * 50}")
    print(f"Results for {model_label}")
    print(f"{'=' * 50}")
    print(f"Average ROUGE-1 : {metrics['rouge1']:.4f}")
    print(f"Average ROUGE-2 : {metrics['rouge2']:.4f}")
    print(f"Average ROUGE-L : {metrics['rougeL']:.4f}")
    print(f"Average BLEU    : {metrics['bleu']:.4f}")
    print(f"Average METEOR  : {metrics['meteor']:.4f}")

    return metrics


def make_preprocess_fn(tokenizer):
    def preprocess(example):
        documents = example[SOURCE_FIELD]
        summaries = example[TARGET_FIELD]

        model_inputs = tokenizer(
            documents,
            padding="max_length",
            truncation=True,
            max_length=INPUT_MAX_LENGTH,
        )
        labels = tokenizer(
            summaries,
            padding="max_length",
            truncation=True,
            max_length=SUMMARY_MAX_LENGTH,
        )

        label_ids = [
            [
                token if token != tokenizer.pad_token_id else -100
                for token in ids
            ]
            for ids in labels["input_ids"]
        ]

        model_inputs["labels"] = label_ids
        return model_inputs

    return preprocess


def main():
    ensure_meteor_resources()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading XSum test split for baseline evaluation ...")
    test_dataset = load_dataset(DATASET_NAME, split="test")
    # Uncomment the next line to run a smaller debug sample.
    # test_dataset = test_dataset.select(range(500))

    print("\n[1/2] BART-large-CNN - evaluation on XSum before XSum fine-tuning")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    pre_metrics = evaluate_model(
        model,
        tokenizer,
        test_dataset,
        device,
        predictions_path=PRE_PREDICTIONS_PATH,
        references_path=PRE_REFERENCES_PATH,
        model_label="BART-large-CNN (before XSum fine-tuning)",
    )

    print("\n" + "=" * 60)
    print("TRAINING - BART-large-CNN on XSum")
    print("=" * 60)

    preprocess_fn = make_preprocess_fn(tokenizer)

    train_dataset = load_dataset(DATASET_NAME, split="train")
    val_dataset = load_dataset(DATASET_NAME, split="validation")

    train_dataset = train_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=str(LOG_DIR),
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model(str(CHECKPOINT_DIR / "final"))
    tokenizer.save_pretrained(str(CHECKPOINT_DIR / "final"))

    post_metrics = evaluate_model(
        model,
        tokenizer,
        test_dataset,
        device,
        predictions_path=POST_PREDICTIONS_PATH,
        references_path=POST_REFERENCES_PATH,
        model_label="BART-large-CNN (after fine-tuning on XSum)",
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": DATASET_NAME,
                "model_name": MODEL_NAME,
                "source_field": SOURCE_FIELD,
                "target_field": TARGET_FIELD,
                "hyperparameters": {
                    "input_max_length": INPUT_MAX_LENGTH,
                    "summary_max_length": SUMMARY_MAX_LENGTH,
                    "num_train_epochs": NUM_EPOCHS,
                    "per_device_train_batch_size": TRAIN_BATCH_SIZE,
                    "per_device_eval_batch_size": EVAL_BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "warmup_steps": WARMUP_STEPS,
                    "weight_decay": WEIGHT_DECAY,
                    "logging_steps": LOGGING_STEPS,
                    "save_steps": SAVE_STEPS,
                    "eval_steps": EVAL_STEPS,
                    "num_beams": NUM_BEAMS,
                    "length_penalty": LENGTH_PENALTY,
                },
                "pre_training": pre_metrics,
                "post_training": post_metrics,
            },
            handle,
            indent=2,
        )

    print("\nAll experiments complete.")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
