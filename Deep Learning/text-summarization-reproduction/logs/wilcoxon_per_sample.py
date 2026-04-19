import argparse
import csv
from pathlib import Path

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from scipy.stats import wilcoxon


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two summarization systems using per-sample metrics and the Wilcoxon signed-rank test."
    )
    parser.add_argument("--pred-a", required=True)
    parser.add_argument("--pred-b", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--label-a", default="system_a")
    parser.add_argument("--label-b", default="system_b")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["rouge1", "rouge2", "rougel", "bleu", "meteor"],
        choices=["rouge1", "rouge2", "rougel", "bleu", "meteor"],
    )
    parser.add_argument("--output-csv", default="wilcoxon_results.csv")
    parser.add_argument("--save-per-sample", default="")
    parser.add_argument(
        "--alternative",
        default="two-sided",
        choices=["two-sided", "greater", "less"],
        help="Alternative for scipy.stats.wilcoxon. 'greater' means system A > system B."
    )
    return parser.parse_args()


def load_lines(path_str):
    return Path(path_str).read_text(encoding="utf-8").splitlines()


def mean(values):
    return sum(values) / len(values) if values else 0.0


def compute_scores(preds, refs, metrics):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method1

    rows = []
    for pred, ref in zip(preds, refs):
        row = {}

        if "rouge1" in metrics or "rouge2" in metrics or "rougel" in metrics:
            scores = scorer.score(ref, pred)
            if "rouge1" in metrics:
                row["rouge1"] = scores["rouge1"].fmeasure
            if "rouge2" in metrics:
                row["rouge2"] = scores["rouge2"].fmeasure
            if "rougel" in metrics:
                row["rougel"] = scores["rougeL"].fmeasure

        if "bleu" in metrics:
            row["bleu"] = sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)

        if "meteor" in metrics:
            row["meteor"] = meteor_score([ref.split()], pred.split())

        rows.append(row)

    return rows


def write_per_sample_csv(path_str, scores_a, scores_b, label_a, label_b, metrics):
    with open(path_str, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["sample_id"]
        for metric in metrics:
            header.append(f"{label_a}_{metric}")
            header.append(f"{label_b}_{metric}")
        writer.writerow(header)

        for idx, (row_a, row_b) in enumerate(zip(scores_a, scores_b)):
            row = [idx]
            for metric in metrics:
                row.append(row_a[metric])
                row.append(row_b[metric])
            writer.writerow(row)


def safe_wilcoxon(a, b, alternative):
    try:
        stat, p = wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    except ValueError:
        # usually happens when all paired differences are zero
        stat, p = 0.0, 1.0
    return stat, p


def main():
    args = parse_args()

    preds_a = load_lines(args.pred_a)
    preds_b = load_lines(args.pred_b)
    refs = load_lines(args.references)

    if not (len(preds_a) == len(preds_b) == len(refs)):
        raise ValueError(
            f"Length mismatch: predictions A={len(preds_a)}, predictions B={len(preds_b)}, references={len(refs)}"
        )

    scores_a = compute_scores(preds_a, refs, args.metrics)
    scores_b = compute_scores(preds_b, refs, args.metrics)

    if args.save_per_sample:
        write_per_sample_csv(args.save_per_sample, scores_a, scores_b, args.label_a, args.label_b, args.metrics)

    rows = []
    for metric in args.metrics:
        a_vals = [r[metric] for r in scores_a]
        b_vals = [r[metric] for r in scores_b]
        stat, p = safe_wilcoxon(a_vals, b_vals, args.alternative)
        rows.append({
            "metric": metric,
            "mean_a": mean(a_vals),
            "mean_b": mean(b_vals),
            "delta_a_minus_b": mean(a_vals) - mean(b_vals),
            "wilcoxon_statistic": stat,
            "p_value": p,
            "significant_at_0_05": "yes" if p < 0.05 else "no",
        })

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "mean_a",
                "mean_b",
                "delta_a_minus_b",
                "wilcoxon_statistic",
                "p_value",
                "significant_at_0_05",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Compared {args.label_a} vs {args.label_b}")
    print(f"Samples: {len(refs)}")
    print(f"Alternative hypothesis: {args.alternative}")
    print()

    for row in rows:
        print(
            f"{row['metric']:7s} | "
            f"{args.label_a}={row['mean_a']:.4f} | "
            f"{args.label_b}={row['mean_b']:.4f} | "
            f"delta={row['delta_a_minus_b']:.4f} | "
            f"W={row['wilcoxon_statistic']:.1f} | "
            f"p={row['p_value']:.6g} | "
            f"significant={row['significant_at_0_05']}"
        )

    print()
    print(f"Saved summary to {args.output_csv}")
    if args.save_per_sample:
        print(f"Saved per-sample scores to {args.save_per_sample}")


if __name__ == "__main__":
    main()
