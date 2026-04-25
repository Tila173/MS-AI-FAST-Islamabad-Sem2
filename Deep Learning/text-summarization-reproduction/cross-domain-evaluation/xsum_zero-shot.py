import argparse
import csv
import importlib.util
import json
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import nltk
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# XSum is text-only. Some environments have a broken optional soundfile install,
# which can make `datasets` fail during import while checking audio support.
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
DEFAULT_OUTPUT_DIR = ROOT / "xsum_zero_shot_results"
XSUM_SPLITS = ("train", "validation", "test")
CNN_DAILYMAIL_SPLIT_SIZES = {
    "train": 287113,
    "validation": 13368,
    "test": 11490,
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    checkpoint_root: str
    base_model: str
    input_prefix: str
    input_max_length: int
    prefer_final: bool = False


MODEL_SPECS = {
    "bart": ModelSpec(
        key="bart",
        label="BART-large-CNN fine-tuned",
        checkpoint_root="checkpoints_bart",
        base_model="facebook/bart-large-cnn",
        input_prefix="",
        input_max_length=1024,
    ),
    "pegasus": ModelSpec(
        key="pegasus",
        label="PEGASUS-CNN/DailyMail fine-tuned",
        checkpoint_root="checkpoints_pegasus",
        base_model="google/pegasus-cnn_dailymail",
        input_prefix="",
        input_max_length=1024,
    ),
    "t5base": ModelSpec(
        key="t5base",
        label="T5-base fine-tuned",
        checkpoint_root="checkpoints_t5base",
        base_model="t5-base",
        input_prefix="summarize: ",
        input_max_length=512,
    ),
    "t5large": ModelSpec(
        key="t5large",
        label="T5-large fine-tuned",
        checkpoint_root="checkpoints_t5large",
        base_model="t5-large",
        input_prefix="summarize: ",
        input_max_length=512,
    ),
}


TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "spiece.model",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate existing fine-tuned summarization checkpoints on the "
            "XSum test split for zero-shot transfer."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", *MODEL_SPECS.keys()],
        help="Model keys to evaluate. Use all for every local checkpoint.",
    )
    parser.add_argument(
        "--dataset-name",
        default="EdinburghNLP/xsum",
        help="Hugging Face dataset name for XSum.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=XSUM_SPLITS,
        help="XSum split to evaluate. Uses the same train/validation/test names as CNN/DailyMail.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional number of XSum samples to evaluate. Omit for full test split.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle before applying --max-samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle is enabled.",
    )
    parser.add_argument(
        "--summary-max-length",
        type=int,
        default=64,
        help="Maximum generated summary tokens for XSum.",
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=5,
        help="Minimum generated summary tokens.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Beam size for generation.",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=1.0,
        help="Length penalty for generation.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=3,
        help="No-repeat ngram size for generation. Use 0 to disable.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Load models in float16 on CUDA to reduce GPU memory use.",
    )
    parser.add_argument(
        "--skip-meteor",
        action="store_true",
        help="Skip METEOR computation.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use local Hugging Face cache only.",
    )
    parser.add_argument(
        "--no-graphs",
        action="store_true",
        help="Do not save metric bar graph images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metrics, predictions, references, and logs.",
    )
    return parser.parse_args()


def selected_specs(model_keys):
    if "all" in model_keys:
        return list(MODEL_SPECS.values())
    return [MODEL_SPECS[key] for key in model_keys]


def checkpoint_step(path):
    try:
        return int(path.name.split("-")[-1])
    except ValueError:
        return -1


def resolve_checkpoint(spec):
    root = ROOT / spec.checkpoint_root
    if not root.exists():
        raise FileNotFoundError(f"Missing checkpoint root: {root}")

    final_dir = root / "final"
    if spec.prefer_final and final_dir.exists():
        return final_dir

    checkpoints = [
        item for item in root.iterdir()
        if item.is_dir() and item.name.startswith("checkpoint-")
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* folders found in: {root}")

    return max(checkpoints, key=checkpoint_step)


def has_tokenizer_files(path):
    return any((path / file_name).exists() for file_name in TOKENIZER_FILES)


def load_tokenizer(spec, checkpoint_path, offline):
    local_files_only = bool(offline)
    if has_tokenizer_files(checkpoint_path):
        tokenizer_source = str(checkpoint_path)
    else:
        tokenizer_source = spec.base_model

    return AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
        local_files_only=local_files_only,
    )


def load_xsum(args):
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    datasets_by_split = {
        split_name: load_dataset(args.dataset_name, split=split_name)
        for split_name in XSUM_SPLITS
    }
    split_info = {
        "xsum": {
            split_name: len(dataset)
            for split_name, dataset in datasets_by_split.items()
        },
        "cnn_dailymail_3_0_0": CNN_DAILYMAIL_SPLIT_SIZES,
        "evaluated_split": args.split,
    }

    print("\nSplit sizes:")
    print("CNN/DailyMail 3.0.0:")
    for split_name in XSUM_SPLITS:
        print(f"  {split_name:10s}: {CNN_DAILYMAIL_SPLIT_SIZES[split_name]}")
    print("XSum:")
    for split_name in XSUM_SPLITS:
        print(f"  {split_name:10s}: {split_info['xsum'][split_name]}")

    dataset = datasets_by_split[args.split]
    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    if args.max_samples is not None:
        n_samples = min(args.max_samples, len(dataset))
        dataset = dataset.select(range(n_samples))

    required = {"document", "summary"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset is missing XSum columns {sorted(missing)}. "
            f"Available columns: {dataset.column_names}"
        )

    return dataset, split_info


def normalize_line(text):
    return " ".join(str(text).split())


def safe_meteor(reference_tokens, prediction_tokens):
    if not reference_tokens or not prediction_tokens:
        return 0.0
    try:
        return meteor_score([reference_tokens], prediction_tokens)
    except LookupError:
        return None


def ensure_meteor_resources(offline):
    if offline:
        return
    for resource in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            ok = nltk.download(resource, quiet=True)
            if not ok:
                print(f"Warning: could not download NLTK resource: {resource}")


def average(values):
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def evaluate_model(spec, checkpoint_path, dataset, args, device):
    output_dir = args.output_dir
    predictions_path = output_dir / f"{spec.key}_xsum_predictions.txt"
    references_path = output_dir / f"{spec.key}_xsum_references.txt"
    generations_path = output_dir / f"{spec.key}_xsum_generations.jsonl"

    print(f"\nLoading {spec.label}")
    print(f"Checkpoint: {checkpoint_path}")

    tokenizer = load_tokenizer(spec, checkpoint_path, args.offline)
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else None
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint_path,
        torch_dtype=dtype,
        local_files_only=args.offline,
    ).to(device)
    model.eval()

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    smoother = SmoothingFunction()

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_scores = []
    meteor_scores = []

    predictions = []
    references = []

    gen_kwargs = {
        "max_new_tokens": args.summary_max_length,
        "num_beams": args.num_beams,
        "length_penalty": args.length_penalty,
        "early_stopping": True,
    }
    if args.min_new_tokens > 0:
        gen_kwargs["min_new_tokens"] = args.min_new_tokens
    if args.no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    with torch.inference_mode():
        for idx, item in enumerate(tqdm(dataset, desc=f"Evaluating {spec.key}")):
            document = normalize_line(item["document"])
            reference = normalize_line(item["summary"])
            source_text = spec.input_prefix + document

            inputs = tokenizer(
                source_text,
                return_tensors="pt",
                max_length=spec.input_max_length,
                truncation=True,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            generated_ids = model.generate(**inputs, **gen_kwargs)
            prediction = normalize_line(
                tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            )

            scores = scorer.score(reference, prediction)
            reference_tokens = reference.split()
            prediction_tokens = prediction.split()

            rouge1_scores.append(scores["rouge1"].fmeasure)
            rouge2_scores.append(scores["rouge2"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)
            if prediction_tokens:
                bleu_scores.append(
                    sentence_bleu(
                        [reference_tokens],
                        prediction_tokens,
                        smoothing_function=smoother.method1,
                    )
                )
            else:
                bleu_scores.append(0.0)

            if args.skip_meteor:
                meteor_scores.append(None)
            else:
                meteor_scores.append(safe_meteor(reference_tokens, prediction_tokens))

            predictions.append(prediction)
            references.append(reference)

            with generations_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "index": idx,
                    "id": item.get("id", idx),
                    "reference": reference,
                    "prediction": prediction,
                }, ensure_ascii=False) + "\n")

    predictions_path.write_text("\n".join(predictions) + "\n", encoding="utf-8")
    references_path.write_text("\n".join(references) + "\n", encoding="utf-8")

    result = {
        "model_key": spec.key,
        "model_label": spec.label,
        "checkpoint": str(checkpoint_path),
        "base_model": spec.base_model,
        "num_samples": len(dataset),
        "rouge1": average(rouge1_scores),
        "rouge2": average(rouge2_scores),
        "rougeL": average(rougeL_scores),
        "bleu": average(bleu_scores),
        "meteor": average(meteor_scores),
        "predictions_file": str(predictions_path),
        "references_file": str(references_path),
        "generations_file": str(generations_path),
    }

    print_result(result)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def print_result(result):
    print("\n" + "=" * 70)
    print(f"XSum zero-shot transfer results: {result['model_label']}")
    print("=" * 70)
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"Samples   : {result['num_samples']}")
    print(f"ROUGE-1   : {result['rouge1']:.4f}")
    print(f"ROUGE-2   : {result['rouge2']:.4f}")
    print(f"ROUGE-L   : {result['rougeL']:.4f}")
    print(f"BLEU      : {result['bleu']:.4f}")
    if result["meteor"] is None:
        print("METEOR    : unavailable, NLTK wordnet data is missing")
    else:
        print(f"METEOR    : {result['meteor']:.4f}")


def save_summary(results, args, split_info, graph_files):
    csv_path = args.output_dir / "xsum_zero_shot_metrics.csv"
    json_path = args.output_dir / "xsum_zero_shot_metrics.json"

    fieldnames = [
        "model_key",
        "model_label",
        "checkpoint",
        "base_model",
        "num_samples",
        "rouge1",
        "rouge2",
        "rougeL",
        "bleu",
        "meteor",
        "predictions_file",
        "references_file",
        "generations_file",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    payload = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "split_sizes": split_info,
        "max_samples": args.max_samples,
        "generation": {
            "summary_max_length": args.summary_max_length,
            "min_new_tokens": args.min_new_tokens,
            "num_beams": args.num_beams,
            "length_penalty": args.length_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
        },
        "skip_meteor": args.skip_meteor,
        "graph_files": [str(path) for path in graph_files],
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nSaved summary files:")
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")


def save_bar_graphs(results, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is not installed, so graph images were not saved.")
        return []

    metrics = [
        ("rouge1", "ROUGE-1"),
        ("rouge2", "ROUGE-2"),
        ("rougeL", "ROUGE-L"),
        ("bleu", "BLEU"),
        ("meteor", "METEOR"),
    ]
    model_names = [result["model_key"] for result in results]
    graph_files = []

    available_metrics = [
        (metric_key, metric_label)
        for metric_key, metric_label in metrics
        if any(result[metric_key] is not None for result in results)
    ]

    if not available_metrics:
        print("No metric values available for graph generation.")
        return []

    x_positions = list(range(len(results)))
    width = min(0.16, 0.8 / max(len(available_metrics), 1))
    offset = (len(available_metrics) - 1) / 2

    fig, ax = plt.subplots(figsize=(12, 6))
    for metric_idx, (metric_key, metric_label) in enumerate(available_metrics):
        values = [
            0.0 if result[metric_key] is None else result[metric_key]
            for result in results
        ]
        positions = [
            x + (metric_idx - offset) * width
            for x in x_positions
        ]
        bars = ax.bar(positions, values, width=width, label=metric_label)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("XSum Zero-Shot Transfer Metrics")
    ax.set_ylabel("Score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()

    combined_path = output_dir / "xsum_zero_shot_metric_bars.png"
    fig.savefig(combined_path, dpi=300)
    plt.close(fig)
    graph_files.append(combined_path)

    for metric_key, metric_label in available_metrics:
        values = [
            0.0 if result[metric_key] is None else result[metric_key]
            for result in results
        ]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(model_names, values)
        ax.set_title(f"XSum Zero-Shot {metric_label}")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", labelrotation=15)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        fig.tight_layout()
        metric_path = output_dir / f"xsum_zero_shot_{metric_key}_bar.png"
        fig.savefig(metric_path, dpi=300)
        plt.close(fig)
        graph_files.append(metric_path)

    print("\nSaved graph images:")
    for path in graph_files:
        print(path)

    return graph_files


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.fp16 and device.type != "cuda":
        print("--fp16 was requested, but CUDA is not available. Using float32.")

    print(f"Loading XSum split: {args.dataset_name}/{args.split}")
    dataset, split_info = load_xsum(args)
    print(f"Evaluation samples: {len(dataset)}")
    if not args.skip_meteor:
        ensure_meteor_resources(args.offline)

    results = []
    for spec in selected_specs(args.models):
        checkpoint_path = resolve_checkpoint(spec)
        generations_path = args.output_dir / f"{spec.key}_xsum_generations.jsonl"
        if generations_path.exists():
            generations_path.unlink()
        results.append(evaluate_model(spec, checkpoint_path, dataset, args, device))

    graph_files = []
    if not args.no_graphs:
        graph_files = save_bar_graphs(results, args.output_dir)

    save_summary(results, args, split_info, graph_files)

    print("\nFinal summary:")
    for result in results:
        meteor_text = "NA" if result["meteor"] is None else f"{result['meteor']:.4f}"
        print(
            f"{result['model_key']}: "
            f"R1={result['rouge1']:.4f} "
            f"R2={result['rouge2']:.4f} "
            f"RL={result['rougeL']:.4f} "
            f"BLEU={result['bleu']:.4f} "
            f"METEOR={meteor_text}"
        )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"xsum_zero_shot_{timestamp}.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            print(f"Saving log to: {log_path}")
            run(args)
        except Exception:
            traceback.print_exc()
            raise
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
