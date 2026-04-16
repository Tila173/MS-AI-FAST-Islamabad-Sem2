# -*- coding: utf-8 -*-
"""TEXT SUMMARIZATION - Corrected version aligned with:
Rani Krishna et al., "Deep learning for text summarization using NLP
for automated news digest", Scientific Reports (2025) 15:36343.

Corrections summary:
- Fixed dataset split error for T5-base pre-training (was DatasetDict, no split specified)
- Added required "summarize: " task prefix for all T5 models
- Fixed BART model: facebook/bart-large -> facebook/bart-large-cnn
- Fixed PEGASUS model: google/pegasus-large -> google/pegasus-cnn_dailymail
- Fixed max summary generation length: 512 -> 128 tokens (per Table 2)
- Fixed BART/PEGASUS preprocessing max_input_length: 512 -> 1024 (per Table 2)
- Fixed max_target_length: 150 -> 128 (per Table 2)
- Fixed learning rates per model (per Table 2)
- Fixed batch sizes per model (per Table 2)
- Fixed warmup_steps to 2000 for all models (per Table 2)
- Added predict_with_generate=True to Seq2SeqTrainingArguments
- Applied -100 label masking to ignore padding in loss computation
- Used full test split for all post-training evaluations
- Pre-training baselines now evaluated on full test split for consistency
"""

# ── Installation ─────────────────────────────────────────────────────────────
# !pip install accelerate -U
# !pip install transformers datasets rouge_score nltk tqdm

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm.auto import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: evaluate a model on the CNN/DailyMail test split
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, tokenizer, dataset, device,
                   input_prefix="", input_max_length=1024,
                   summary_max_length=128, model_label="Model"):
    """
    Generate summaries for every article in *dataset* and compute
    ROUGE-1/2/L and BLEU scores.

    Parameters
    ----------
    input_prefix      : str   - task prefix prepended to each article
                                (required for T5 models, e.g. "summarize: ")
    input_max_length  : int   - tokeniser truncation length for the article
    summary_max_length: int   - maximum tokens for the generated summary
                                (paper Table 2: 128 for all models)
    """
    scorer_obj = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )
    smoother = SmoothingFunction()

    rouge_scores = []
    bleu_scores  = []

    model.eval()
    with torch.no_grad():
        for article, reference in tqdm(
            zip(dataset['article'], dataset['highlights']),
            total=len(dataset),
            desc=f"Evaluating {model_label}"
        ):
            # Prepend task prefix where required (T5 models)
            text = input_prefix + article

            inputs = tokenizer(
                text,
                return_tensors='pt',
                max_length=input_max_length,
                truncation=True
            ).to(device)

            summary_ids = model.generate(
                **inputs,
                max_length=summary_max_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            summary_text = tokenizer.decode(
                summary_ids[0], skip_special_tokens=True
            )

            rouge_scores.append(scorer_obj.score(reference, summary_text))
            bleu_scores.append(
                sentence_bleu(
                    [reference.split()],
                    summary_text.split(),
                    smoothing_function=smoother.method1
                )
            )

    avg_rouge1 = sum(s['rouge1'].fmeasure for s in rouge_scores) / len(rouge_scores)
    avg_rouge2 = sum(s['rouge2'].fmeasure for s in rouge_scores) / len(rouge_scores)
    avg_rougeL = sum(s['rougeL'].fmeasure for s in rouge_scores) / len(rouge_scores)
    avg_bleu   = sum(bleu_scores) / len(bleu_scores)

    print(f"\n{'='*50}")
    print(f"Results for {model_label}")
    print(f"{'='*50}")
    print(f"Average ROUGE-1 : {avg_rouge1:.4f}")
    print(f"Average ROUGE-2 : {avg_rouge2:.4f}")
    print(f"Average ROUGE-L : {avg_rougeL:.4f}")
    print(f"Average BLEU    : {avg_bleu:.4f}")

    return avg_rouge1, avg_rouge2, avg_rougeL, avg_bleu


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: preprocess dataset for Seq2Seq training
# ─────────────────────────────────────────────────────────────────────────────

def make_preprocess_fn(tokenizer, input_prefix="",
                       input_max_length=512, target_max_length=128):
    """
    Returns a HuggingFace-compatible map function.

    Corrections vs original:
    - target_max_length changed 150 → 128 (paper Table 2)
    - input_max_length for BART/PEGASUS set to 1024 (paper Table 2)
    - -100 masking applied to label padding tokens so they are
      excluded from the cross-entropy loss computation
    - T5 models receive the required "summarize: " prefix
    """
    def preprocess(example):
        articles   = [input_prefix + a for a in example['article']]
        highlights = example['highlights']

        model_inputs = tokenizer(
            articles,
            padding='max_length',
            truncation=True,
            max_length=input_max_length
        )
        labels = tokenizer(
            highlights,
            padding='max_length',
            truncation=True,
            max_length=target_max_length
        )

        # Replace padding token id in labels with -100 so loss ignores them
        label_ids = [
            [(token if token != tokenizer.pad_token_id else -100)
             for token in ids]
            for ids in labels['input_ids']
        ]

        model_inputs['labels'] = label_ids
        return model_inputs

    return preprocess


# ─────────────────────────────────────────────────────────────────────────────
# Load the CNN / Daily Mail test split once (shared across pre-training evals)
# ─────────────────────────────────────────────────────────────────────────────

print("Loading CNN/DailyMail test split for baseline evaluation …")
test_dataset = load_dataset('cnn_dailymail', '3.0.0', split="test")
# Paper uses the full test split (~11,490 samples) for evaluation.
# Uncomment the line below to use a smaller subset during development:
# test_dataset = test_dataset.select(range(500))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================================
# SECTION 1 – PRE-TRAINING (BASELINE) EVALUATION
# Models are loaded from their pre-trained checkpoints and evaluated WITHOUT
# any fine-tuning on CNN/DailyMail to reproduce Table 7 of the paper.
# =============================================================================

# ─── 1A. T5-base (pre-training baseline) ─────────────────────────────────────
# FIX 1: original code did dataset.select() on a DatasetDict (no split given).
# FIX 2: T5 requires a task prefix; without it the model ignores the task type.
# FIX 3: summary max_length corrected 512 → 128 (paper Table 2).
print("\n[1/4] T5-base – baseline evaluation")
model_name = 't5-base'
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

evaluate_model(
    model, tokenizer, test_dataset, device,
    input_prefix="summarize: ",   # required for T5
    input_max_length=512,          # Table 2: T5-base max input = 512
    summary_max_length=128,        # Table 2: max target = 128
    model_label="T5-base (pre-training)"
)

# ─── 1B. T5-large (pre-training baseline) ────────────────────────────────────
# FIX: task prefix added; summary max_length corrected 512 → 128.
print("\n[2/4] T5-large – baseline evaluation")
model_name = 't5-large'
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

evaluate_model(
    model, tokenizer, test_dataset, device,
    input_prefix="summarize: ",
    input_max_length=512,          # Table 2: T5-large max input = 512
    summary_max_length=128,
    model_label="T5-large (pre-training)"
)

# ─── 1C. BART-large-CNN (pre-training baseline) ───────────────────────────────
# FIX: model changed facebook/bart-large → facebook/bart-large-cnn.
#      The paper explicitly names "BART CNN" and its CNN/DailyMail fine-tuned
#      checkpoint is what produces the scores in Table 5/7.
#      No task prefix needed for BART.
print("\n[3/4] BART-large-CNN – baseline evaluation")
model_name = 'facebook/bart-large-cnn'
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

evaluate_model(
    model, tokenizer, test_dataset, device,
    input_prefix="",               # BART does not use a task prefix
    input_max_length=1024,         # Table 2: BART max input = 1024
    summary_max_length=128,
    model_label="BART-large-CNN (pre-training)"
)

# ─── 1D. PEGASUS-large (pre-training baseline) ───────────────────────────────
# FIX: model changed google/pegasus-large → google/pegasus-cnn_dailymail.
#      The paper uses the CNN/DailyMail-specific PEGASUS checkpoint.
print("\n[4/4] PEGASUS-large – baseline evaluation")
model_name = 'google/pegasus-cnn_dailymail'
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

evaluate_model(
    model, tokenizer, test_dataset, device,
    input_prefix="",               # PEGASUS does not use a task prefix
    input_max_length=1024,         # Table 2: PEGASUS max input = 1024
    summary_max_length=128,
    model_label="PEGASUS-large (pre-training)"
)


# =============================================================================
# SECTION 2 – FINE-TUNING + POST-TRAINING EVALUATION
# Each model is fine-tuned on CNN/DailyMail train split and re-evaluated on
# the test split to reproduce Table 8 of the paper.
#
# Hyperparameters are set per Table 2:
#   T5-base  : lr=3e-4, batch=64, warmup=2000, optimizer=AdamW
#   T5-large : lr=3e-4, batch=32, warmup=2000, optimizer=AdamW
#   BART     : lr=3e-5, batch=64, warmup=2000, optimizer=Adam
#   PEGASUS  : lr=1e-4, batch=64, warmup=2000, optimizer=Adam
# =============================================================================

# ─── 2A. T5-base fine-tuning ─────────────────────────────────────────────────
print("\n" + "="*60)
print("TRAINING – T5-base")
print("="*60)

model_name = 't5-base'
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# FIX: added "summarize: " prefix; target max_length 150 → 128 (Table 2)
preprocess_fn = make_preprocess_fn(
    tokenizer,
    input_prefix="summarize: ",
    input_max_length=512,   # Table 2
    target_max_length=128   # Table 2
)

train_dataset_t5b = load_dataset('cnn_dailymail', '3.0.0', split="train")
val_dataset_t5b   = load_dataset('cnn_dailymail', '3.0.0', split="validation")

train_dataset_t5b = train_dataset_t5b.map(preprocess_fn, batched=True,
                                           remove_columns=train_dataset_t5b.column_names)
val_dataset_t5b   = val_dataset_t5b.map(preprocess_fn, batched=True,
                                          remove_columns=val_dataset_t5b.column_names)

# FIX: batch_size=64, lr=3e-4, warmup=2000, predict_with_generate=True (Table 2)
training_args_t5b = Seq2SeqTrainingArguments(
    output_dir="./checkpoints_t5base",
    num_train_epochs=3,                    # paper: 100-200K steps ≈ 3 epochs on full data
    per_device_train_batch_size=64,        # Table 2: batch size 64
    per_device_eval_batch_size=64,
    learning_rate=3e-4,                    # Table 2: lr 3e-4
    warmup_steps=2000,                     # Table 2: warm-up 2000
    weight_decay=0.01,                     # Table 2: weight decay 0.01
    logging_dir="./logs",
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_total_limit=2,
    predict_with_generate=True,            # FIX: required for Seq2Seq evaluation
    fp16=torch.cuda.is_available(),        # use mixed precision if GPU available
)

data_collator_t5b = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer_t5b = Seq2SeqTrainer(
    model=model,
    args=training_args_t5b,
    data_collator=data_collator_t5b,
    train_dataset=train_dataset_t5b,
    eval_dataset=val_dataset_t5b,
)

trainer_t5b.train()

evaluate_model(
    model, tokenizer, test_dataset, device,
    input_prefix="summarize: ",
    input_max_length=512,
    summary_max_length=128,
    model_label="T5-base (after fine-tuning)"
)


# ─── 2B. T5-large fine-tuning ────────────────────────────────────────────────
print("\n" + "="*60)
print("TRAINING – T5-large")
print("="*60)

model_name = 't5-large'
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# FIX: added "summarize: " prefix; target max_length 150 → 128 (Table 2)
preprocess_fn = make_preprocess_fn(
    tokenizer,
    input_prefix="summarize: ",
    input_max_length=512,   # Table 2
    target_max_length=128   # Table 2
)

train_dataset_t5l = load_dataset('cnn_dailymail', '3.0.0', split="train")
val_dataset_t5l   = load_dataset('cnn_dailymail', '3.0.0', split="validation")

train_dataset_t5l = train_dataset_t5l.map(preprocess_fn, batched=True,
                                            remove_columns=train_dataset_t5l.column_names)
val_dataset_t5l   = val_dataset_t5l.map(preprocess_fn, batched=True,
                                          remove_columns=val_dataset_t5l.column_names)

# FIX: batch_size=32, lr=3e-4, warmup=2000 (Table 2)
training_args_t5l = Seq2SeqTrainingArguments(
    output_dir="./checkpoints_t5large",
    num_train_epochs=2,                    # paper: 50-100K steps ≈ 2 epochs
    per_device_train_batch_size=32,        # Table 2: batch size 32
    per_device_eval_batch_size=32,
    learning_rate=3e-4,                    # Table 2: lr 3e-4
    warmup_steps=2000,                     # Table 2: warm-up 2000
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
)

data_collator_t5l = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer_t5l = Seq2SeqTrainer(
    model=model,
    args=training_args_t5l,
    data_collator=data_collator_t5l,
    train_dataset=train_dataset_t5l,
    eval_dataset=val_dataset_t5l,
)

trainer_t5l.train()

evaluate_model(
    model, tokenizer, test_dataset, device,
    input_prefix="summarize: ",
    input_max_length=512,
    summary_max_length=128,
    model_label="T5-large (after fine-tuning)"
)


# ─── 2C. BART-large-CNN fine-tuning ──────────────────────────────────────────
# FIX: model changed facebook/bart-large → facebook/bart-large-cnn
# FIX: input_max_length 512 → 1024 (Table 2); target 150 → 128 (Table 2)
# FIX: lr=3e-5, batch=64, warmup=2000 (Table 2)
print("\n" + "="*60)
print("TRAINING – BART-large-CNN")
print("="*60)

model_name = 'facebook/bart-large-cnn'
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

preprocess_fn = make_preprocess_fn(
    tokenizer,
    input_prefix="",        # BART needs no task prefix
    input_max_length=1024,  # Table 2: BART max input = 1024
    target_max_length=128   # Table 2
)

train_dataset_bart = load_dataset('cnn_dailymail', '3.0.0', split="train")
val_dataset_bart   = load_dataset('cnn_dailymail', '3.0.0', split="validation")

train_dataset_bart = train_dataset_bart.map(preprocess_fn, batched=True,
                                              remove_columns=train_dataset_bart.column_names)
val_dataset_bart   = val_dataset_bart.map(preprocess_fn, batched=True,
                                            remove_columns=val_dataset_bart.column_names)

training_args_bart = Seq2SeqTrainingArguments(
    output_dir="./checkpoints_bart",
    num_train_epochs=2,                    # paper: 30-60K steps ≈ 2 epochs
    per_device_train_batch_size=64,        # Table 2: batch size 64
    per_device_eval_batch_size=64,
    learning_rate=3e-5,                    # Table 2: lr 3e-5
    warmup_steps=2000,                     # Table 2: warm-up 2000
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
)

data_collator_bart = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer_bart = Seq2SeqTrainer(
    model=model,
    args=training_args_bart,
    data_collator=data_collator_bart,
    train_dataset=train_dataset_bart,
    eval_dataset=val_dataset_bart,
)

trainer_bart.train()

evaluate_model(
    model, tokenizer, test_dataset, device,
    input_prefix="",
    input_max_length=1024,
    summary_max_length=128,
    model_label="BART-large-CNN (after fine-tuning)"
)


# ─── 2D. PEGASUS-large fine-tuning ───────────────────────────────────────────
# FIX: model changed google/pegasus-large → google/pegasus-cnn_dailymail
# FIX: input_max_length 512 → 1024 (Table 2); target 150 → 128 (Table 2)
# FIX: lr=1e-4, batch=64, warmup=2000 (Table 2)
print("\n" + "="*60)
print("TRAINING – PEGASUS-large")
print("="*60)

model_name = 'google/pegasus-cnn_dailymail'
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

preprocess_fn = make_preprocess_fn(
    tokenizer,
    input_prefix="",        # PEGASUS needs no task prefix
    input_max_length=1024,  # Table 2: PEGASUS max input = 1024
    target_max_length=128   # Table 2
)

train_dataset_peg = load_dataset('cnn_dailymail', '3.0.0', split="train")
val_dataset_peg   = load_dataset('cnn_dailymail', '3.0.0', split="validation")

train_dataset_peg = train_dataset_peg.map(preprocess_fn, batched=True,
                                            remove_columns=train_dataset_peg.column_names)
val_dataset_peg   = val_dataset_peg.map(preprocess_fn, batched=True,
                                          remove_columns=val_dataset_peg.column_names)

training_args_peg = Seq2SeqTrainingArguments(
    output_dir="./checkpoints_pegasus",
    num_train_epochs=2,                    # paper: 30-60K steps ≈ 2 epochs
    per_device_train_batch_size=64,        # Table 2: batch size 64
    per_device_eval_batch_size=64,
    learning_rate=1e-4,                    # Table 2: lr 1e-4
    warmup_steps=2000,                     # Table 2: warm-up 2000
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
)

data_collator_peg = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer_peg = Seq2SeqTrainer(
    model=model,
    args=training_args_peg,
    data_collator=data_collator_peg,
    train_dataset=train_dataset_peg,
    eval_dataset=val_dataset_peg,
)

trainer_peg.train()

evaluate_model(
    model, tokenizer, test_dataset, device,
    input_prefix="",
    input_max_length=1024,
    summary_max_length=128,
    model_label="PEGASUS-large (after fine-tuning)"
)

print("\nAll experiments complete.")
