# -*- coding: utf-8 -*-
"""TF-IDF extractive summarization baseline on CNN/DailyMail.

This script provides a classical non-neural baseline to compare against the
existing abstractive models in `checkingfiles`. It:

- loads the CNN/DailyMail test split
- ranks article sentences with TF-IDF
- applies a light redundancy filter
- builds an extractive summary under a length budget
- reports ROUGE, BLEU, and METEOR

The method is extractive, so it should be treated as a baseline rather than a
drop-in replacement for BART/T5/PEGASUS.
"""

import importlib.util
import json
import re
from pathlib import Path

import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

# Some local environments fail while importing `datasets` because of an
# optional soundfile/libsoundfile issue. This task is text-only.
_original_find_spec = importlib.util.find_spec


def _find_spec_without_soundfile(name, package=None):
    if name == "soundfile":
        return None
    return _original_find_spec(name, package)


importlib.util.find_spec = _find_spec_without_soundfile
try:
    from datasets import load_dataset
finally:
    importlib.util.find_spec = _original_find_spec


ROOT = Path(__file__).resolve().parent
DATASET_NAME = "cnn_dailymail"
DATASET_CONFIG = "3.0.0"
SOURCE_FIELD = "article"
TARGET_FIELD = "highlights"
MAX_SUMMARY_SENTENCES = 3
MAX_SUMMARY_TOKENS = 128
MIN_SENTENCE_TOKENS = 4
REDUNDANCY_THRESHOLD = 0.7
NGRAM_RANGE = (1, 2)
MAX_FEATURES = 5000

RESULTS_DIR = ROOT / "tfidf_cnn_dailymail_results"
PREDICTIONS_PATH = ROOT / "tfidf_cnn_dailymail_predictions.txt"
REFERENCES_PATH = ROOT / "tfidf_cnn_dailymail_references.txt"
METRICS_PATH = RESULTS_DIR / "tfidf_cnn_dailymail_metrics.json"


def ensure_nltk_resources():
    resources = (
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    )
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except (LookupError, OSError):
            try:
                nltk.download(resource_name, quiet=True)
            except Exception:
                pass


def normalize_text(text):
    return " ".join(str(text).split())


def split_sentences(text):
    normalized = normalize_text(text)
    if not normalized:
        return []

    try:
        sentences = sent_tokenize(normalized)
    except (LookupError, OSError):
        sentences = re.split(r"(?<=[.!?])\s+", normalized)

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(normalize_text(line) + "\n")


def truncate_to_budget(text, max_tokens):
    tokens = normalize_text(text).split()
    return " ".join(tokens[:max_tokens])


def safe_meteor(reference, prediction):
    reference_tokens = str(reference).split()
    prediction_tokens = str(prediction).split()
    if not reference_tokens or not prediction_tokens:
        return 0.0
    return meteor_score([reference_tokens], prediction_tokens)


def rank_sentences_with_tfidf(article):
    sentences = split_sentences(article)

    if not sentences:
        return [], None, None
    if len(sentences) == 1:
        return [0], None, sentences

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
        lowercase=True,
    )

    try:
        sentence_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return [0], None, sentences

    centroid = np.asarray(sentence_matrix.mean(axis=0)).reshape(1, -1)
    scores = cosine_similarity(sentence_matrix, centroid).ravel()
    ranked_indices = sorted(
        range(len(sentences)),
        key=lambda idx: (scores[idx], -idx),
        reverse=True,
    )
    return ranked_indices, sentence_matrix, sentences


def build_summary(article):
    ranked_indices, sentence_matrix, sentences = rank_sentences_with_tfidf(article)

    if not sentences:
        return ""
    if sentence_matrix is None:
        return truncate_to_budget(sentences[0], MAX_SUMMARY_TOKENS)

    selected = []
    token_budget_used = 0

    for idx in ranked_indices:
        sentence = sentences[idx]
        token_count = len(sentence.split())

        if token_count < MIN_SENTENCE_TOKENS and len(sentences) > 1:
            continue

        if selected and token_budget_used + token_count > MAX_SUMMARY_TOKENS:
            continue

        is_redundant = False
        for chosen_idx in selected:
            similarity = cosine_similarity(
                sentence_matrix[idx],
                sentence_matrix[chosen_idx],
            )[0, 0]
            if similarity >= REDUNDANCY_THRESHOLD:
                is_redundant = True
                break
        if is_redundant:
            continue

        selected.append(idx)
        token_budget_used += token_count

        if len(selected) >= MAX_SUMMARY_SENTENCES:
            break

    if not selected:
        selected = [ranked_indices[0]]

    selected.sort()
    summary = " ".join(sentences[idx] for idx in selected)
    return truncate_to_budget(summary, MAX_SUMMARY_TOKENS)


def evaluate_dataset(dataset):
    scorer_obj = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    smoother = SmoothingFunction()

    rouge_scores = []
    bleu_scores = []
    meteor_scores = []
    predictions = []
    references = []

    for article, reference in tqdm(
        zip(dataset[SOURCE_FIELD], dataset[TARGET_FIELD]),
        total=len(dataset),
        desc="Evaluating TF-IDF baseline",
    ):
        prediction = build_summary(article)

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

    write_lines(PREDICTIONS_PATH, predictions)
    write_lines(REFERENCES_PATH, references)

    metrics = {
        "rouge1": sum(s["rouge1"].fmeasure for s in rouge_scores) / len(rouge_scores),
        "rouge2": sum(s["rouge2"].fmeasure for s in rouge_scores) / len(rouge_scores),
        "rougeL": sum(s["rougeL"].fmeasure for s in rouge_scores) / len(rouge_scores),
        "bleu": sum(bleu_scores) / len(bleu_scores),
        "meteor": sum(meteor_scores) / len(meteor_scores),
    }

    print(f"\n{'=' * 50}")
    print("Results for TF-IDF CNN/DailyMail baseline")
    print(f"{'=' * 50}")
    print(f"Average ROUGE-1 : {metrics['rouge1']:.4f}")
    print(f"Average ROUGE-2 : {metrics['rouge2']:.4f}")
    print(f"Average ROUGE-L : {metrics['rougeL']:.4f}")
    print(f"Average BLEU    : {metrics['bleu']:.4f}")
    print(f"Average METEOR  : {metrics['meteor']:.4f}")

    return metrics


def main():
    ensure_nltk_resources()

    print("Loading CNN/DailyMail test split for TF-IDF baseline evaluation ...")
    test_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
    # Uncomment to run a smaller debug subset first.
    # test_dataset = test_dataset.select(range(500))

    metrics = evaluate_dataset(test_dataset)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": f"{DATASET_NAME}/{DATASET_CONFIG}",
                "method": "tfidf_extractive_baseline",
                "source_field": SOURCE_FIELD,
                "target_field": TARGET_FIELD,
                "parameters": {
                    "max_summary_sentences": MAX_SUMMARY_SENTENCES,
                    "max_summary_tokens": MAX_SUMMARY_TOKENS,
                    "min_sentence_tokens": MIN_SENTENCE_TOKENS,
                    "redundancy_threshold": REDUNDANCY_THRESHOLD,
                    "ngram_range": list(NGRAM_RANGE),
                    "max_features": MAX_FEATURES,
                },
                "test_metrics": metrics,
            },
            handle,
            indent=2,
        )

    print("\nEvaluation complete.")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
