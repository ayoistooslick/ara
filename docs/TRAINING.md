# Training and evaluation

Ara distinguishes three things that are often confused:

1. **Using an existing pretrained model.** This is what Ara's conversations do, through the model adapter. No Ara training is involved.
2. **Fine-tuning an open-weight model.** A reproducible data bundle and plan are provided. No fine-tune has been run.
3. **Training a model from initialization.** A small counted n-gram baseline is trained from scratch on CPU to make the pipeline real and reproducible. It is not a chat model.

Ara has **not** trained a production language model from scratch. Do not describe the n-gram baseline as Ara's brain.

## Dataset

`ara/training/corpus.py` derives documents and instruction pairs from the curated corpus in `ara/data/nigeria/`:

- 111 source documents after content deduplication;
- 100 training documents and 11 validation documents (deterministic 90/10 split, seed 7);
- 111 chat-format instruction pairs, one per corpus entry, each with a source path and licence;
- `manifest.json` with SHA-256 hashes, counts, seed, privacy statement, and limitations.

No private messages, private accounts, restricted groups, or scraped personal data are used. Instruction answers are derived from documented entries, not invented. The corpus's coverage limits are the dataset's limits.

```sh
python - <<'PY'
from ara.training.corpus import prepare
print(prepare("runs/baseline/data"))
PY
```

Outputs: `corpus.txt`, `train.txt`, `val.txt`, `chat.jsonl`, `manifest.json`.

## Tokenizer

`ara/training/tokenizer.py` trains a byte-level BPE tokenizer. A byte-level base means arbitrary Nigerian text and emoji round-trip without unknown-token loss. The tokenizer and its training log are saved as `tokenizer.json` and `tokenizer_training.json`.

You can reuse a tokenizer by placing a compatible `tokenizer.json` in the output directory before training. The baseline's model card records the tokenizer hash.

## Baseline training

```sh
python -m ara.training.train --out runs/baseline
```

What happens:

1. Prepares the dataset if absent.
2. Trains or loads the tokenizer.
3. Initializes a counted n-gram model (default order 3).
4. Observes each training document, writing `checkpoint.json` every 20 documents.
5. Evaluates interpolation-smoothed perplexity on the validation split.
6. Writes `model.json`, `metrics.json`, `config.json`, and `MODEL_CARD.md`.

The config explicitly records:

```json
{
  "type": "counted-ngram-baseline",
  "training": "from initialization (counted parameters)",
  "fine_tuned": false,
  "chat_model": false,
  "hardware": "CPU"
}
```

Options: `--vocab-size` (default 320), `--order` (1–5), `--data`, `--out`.

### Resume and recovery

```sh
python -m ara.training.train --out runs/baseline --resume
python -m ara.training.train --out runs/baseline --eval-only
```

`--resume` loads `checkpoint.json` and skips documents already counted, so a completed run does not duplicate training. The checkpoint contains the counted model; it is not a transformer optimizer state. If the checkpoint is missing or unreadable, start a new run rather than editing it by hand.

`--eval-only` loads `model.json` and `tokenizer.json` and reports validation perplexity without retraining.

### What the metrics mean

- **Validation perplexity:** exponentiated average negative log-likelihood on held-out corpus text. Lower is better.
- **Training tokens/documents:** how much data was counted.
- **Sample:** a short generated fragment to show the model is producing tokens. It is not a quality demonstration.

The validation split is small and stylistically uniform. Perplexity here is a reproducibility check, not a comparison with other models or evidence of general language ability. Do not publish it as a benchmark.

## Fine-tuning an existing model

```sh
python -m ara.training.train --out runs/baseline --export-finetune \
  --base-model Qwen/Qwen2.5-1.5B-Instruct
```

This writes `runs/baseline/finetune/`:

- `train_chat.jsonl`: chat-format instruction data;
- `finetune_plan.json`: base model, dataset licence notes, detected Python packages, detected binaries, CUDA/device/VRAM, and `"status": "NOT RUN"`;
- `README.md`: warnings and a model-card checklist.

The detector checks for `torch`, `transformers`, `datasets`, `peft`, `trl`, `bitsandbytes`, and llama.cpp tools. Availability is detected, not assumed. This environment may have none of them.

The command does **not** launch a fine-tune. A real run needs:

- a compatible base model and its licence;
- a GPU or a deliberately slow CPU setup;
- installed training libraries or llama.cpp finetune tools;
- hyperparameters recorded in an experiment file;
- evaluation on held-out Nigerian-context cases, not only training loss;
- a model card with data version, base model, licence, hardware, and measured limits.

Do not distribute fine-tuned weights without the base model's licence and a model card.

## Quantization and export

`ara/training/finetune.py` exposes guidance, not an executed quantization. A typical llama.cpp path is:

```sh
llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
```

Flags differ between llama.cpp releases. Check `llama-quantize --help`. Quantization is not run by Ara and is not tested in this environment. Validate output with the actual inference server before claiming a quantized model works.

## Hardware boundary

The baseline runs on a small CPU and is covered by tests. A 1–8B transformer fine-tune does not fit a phone or a free CPU-only environment. Training hardware, storage, and time must be measured on the target machine.

Ara remains useful without any training: it uses a pretrained model through the adapter. Do not block the product on model-development work, and do not represent unfinished training as a completed achievement.

## Evaluation

- Baseline: `--eval-only` and the test suite (`tests/test_training.py`) verify deterministic preparation, tokenizer round-trip, checkpoint resume, metrics, model card, and export honesty.
- Nigerian behaviour: `ara nigeria eval` measures retrieval and interpretation. `--model` adds a keyword rubric over real model answers; it is not a human evaluation.
- Conversation quality: not automatically measured. A real model must be run manually against representative prompts. Record the model, quantization, hardware, and failures.

## Reproduction checklist

1. Commit the corpus and any imported entry provenance.
2. Run `ara nigeria validate` and `ara nigeria eval`.
3. Prepare the dataset and record the manifest hash.
4. Train on a named machine; keep `config.json`, `metrics.json`, and `MODEL_CARD.md`.
5. Evaluate on held-out cases before making claims.
6. Label the result correctly: pretrained, fine-tuned, or trained from initialization.
