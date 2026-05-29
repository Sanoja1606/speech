# Speech — DSACIS + ATR Novelty Extension

> **Extension of [DualSpeechLM](../DualSpeechLM/)** — adds two novel NLP modules (DSACIS and ATR) that sit between the speech tokenizer and the frozen/LoRA-tuned language model, enabling dialogue-aware adaptive token compression for speech-language understanding.

---

## Overview

This folder (`speech/`) contains our **novelty contribution** on top of the base DualSpeechLM framework. The core idea is:

1. **DSACIS** (Dialogue-State Aware Conversational Importance Scoring) — a GRU-based module that tracks multi-turn pragmatic context (semantic embeddings, dialogue acts, emotion trajectory, acoustic-semantic congruence) and outputs a per-turn **importance score**.
2. **ATR** (Adaptive Token Router) — uses the DSACIS importance score alongside prosody features to dynamically **route, compress, or drop** audio tokens before they reach the LLM, achieving real sequence-length reduction.
3. **OptimizedSpeechLMPipeline** — wraps the base LLM (Phi-3.5-mini-instruct, 4-bit NF4 quantized + LoRA) with DSACIS and ATR, training end-to-end with auxiliary losses for dialogue-act classification and contrastive congruence.

### Architecture Diagram

```
Audio Waveform ──► USTokenizer ──► Audio Token IDs
                                        │
                                        ▼
Text Transcript ──► DSACIS ──► importance_score ──► ATR ──► Routed Embeddings
                     │              │                              │
                     │         soft_prefix                         │
                     │              │                              ▼
                     │              └──────────────────────► Phi-3.5-mini (LoRA)
                     │                                             │
                     ▼                                             ▼
              Auxiliary Losses                              Generation Loss
         (L_dialogue_act + L_congruence)                    (L_generation)
                     │                                             │
                     └──────────── Total Loss ◄────────────────────┘
```

---

## Folder Structure

```
speech/
├── README.md                          # ← You are here
├── train_on_colab.py                  # Google Colab training script (T4 GPU)
├── eval_on_colab.py                   # Google Colab evaluation script (LibriSpeech)
├── .gitignore
└── src/
    ├── data/                          # Dataset loaders
    │   ├── __init__.py
    │   ├── iemocap_dataset.py         # IEMOCAP CSV → SER emotion labels
    │   ├── iemocap_full_dataset.csv   # Pre-processed IEMOCAP data
    │   ├── meld_dataset.py            # MELD (HuggingFace) → emotion + dialogue acts
    │   ├── daily_dialog_dataset.py    # DailyDialog (HuggingFace) → dialogue act labels
    │   ├── librispeech_eval.py        # LibriSpeech test-clean WER evaluator
    │   └── DailyDialog/              # Cached DailyDialog data
    └── novelty/                       # ★ Our novel modules
        ├── __init__.py
        ├── pipeline.py                # OptimizedSpeechLMPipeline (forward + inference)
        ├── train_novelty.py           # Full training script (local GPU / multi-GPU)
        ├── dsacis/
        │   ├── __init__.py
        │   └── importance_scorer.py   # DSACISModule, PragmaticStateGRU, SER head
        └── atr/
            ├── __init__.py
            └── token_router.py        # AdaptiveTokenRouter, ProsodyFeatureExtractor
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Tested on 3.10 (Colab) |
| PyTorch | 2.4+ | CUDA 11.8 or 12.x |
| Transformers | ≥ 4.45.0 | For Phi-3.5-mini support |
| PEFT | ≥ 0.4.0 | LoRA adapter training |
| BitsAndBytes | latest | 4-bit NF4 quantization |
| sentence-transformers | latest | Real NLP encoder (`all-MiniLM-L6-v2`) |
| datasets | ≥ 3.0.0 | HuggingFace dataset loading |
| GPU | NVIDIA T4 (16 GB) minimum | Colab free tier works |

---

## Setup

### Option A — Google Colab (Recommended)

This is the tested and recommended path. Runs on a free T4 GPU.

```bash
# 1. Clone the repository into Colab
!git clone https://github.com/<your-repo>/nlp_project_colabedit.git /content/nlp_project_colabedit

# 2. Install dependencies
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q transformers>=4.45.0 peft bitsandbytes accelerate
!pip install -q sentence-transformers datasets

# 3. Mount Google Drive (for checkpoint persistence)
from google.colab import drive
drive.mount('/content/drive')
```

### Option B — Local / Multi-GPU (Full Training)

Requires the base DualSpeechLM dependencies plus our additions:

```bash
# 1. Install base DualSpeechLM requirements
pip install -r ../DualSpeechLM/requirements.txt

# 2. Install novelty-specific packages
pip install sentence-transformers bitsandbytes peft accelerate

# 3. Ensure the project root is on PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/nlp_project_colabedit/speech"
```

---

## Training

### Colab Training (Quick Start — `train_on_colab.py`)

This is the self-contained Colab script. It loads Phi-3.5-mini with 4-bit quantization, builds the DSACIS+ATR pipeline, and trains using multi-task sampling across all auxiliary datasets.

```bash
# From the speech/ directory (or Colab cell):
!cd /content/nlp_project_colabedit/speech && python train_on_colab.py
```

**What it does:**

1. Loads `microsoft/Phi-3.5-mini-instruct` with NF4 4-bit quantization
2. Adds 1024 simulated audio tokens + `<audio>`/`</audio>` markers
3. Applies LoRA (r=8, alpha=16) to Q/K/V/O projections
4. Builds DSACIS (hidden=256, GRU layers=2) + ATR (thresholds: high=0.65, low=0.35)
5. Loads auxiliary datasets: **MELD**, **DailyDialog**, **IEMOCAP**
6. Multi-task training for 100 steps with random dataset sampling:
   - 40% — dummy dialogue samples
   - 20% — MELD
   - 20% — DailyDialog
   - 20% — IEMOCAP
7. Saves checkpoint to `/content/drive/MyDrive/novelty_outputs/checkpoint-step-100`

**Key hyperparameters** (edit in `train_on_colab.py`):

| Parameter | Default | Description |
|---|---|---|
| `num_steps` | 100 | Total training steps |
| `lr` | 2e-5 | AdamW learning rate |
| `batch_size` | 8 | Samples per step |
| `seq_len` | 256 | Max sequence length |
| `lambda_da` | 0.10 | Dialogue act loss weight |
| `lambda_cong` | 0.05 | Congruence loss weight |

**Expected training log output:**

```
Step    0 | loss=12.3456 | L_gen=12.1234 | L_da=0.6932 | Compression=0.72 | GPU=5.23GB
Step   10 | loss=11.8901 | L_gen=11.6789 | L_da=0.6543 | Compression=0.68 | GPU=5.31GB
...
Step  100 | loss=8.2345  | L_gen=8.0123  | L_da=0.4567 | Compression=0.55 | GPU=5.28GB
```

---

### Full Training (Local / Multi-GPU — `train_novelty.py`)

For full-scale training using the base DualSpeechLM data pipeline with Hydra configs:

```bash
cd /path/to/nlp_project_colabedit/speech

python src/novelty/train_novelty.py \
    --model       configs/model/Phi-3.5-mini-instruct_lora.yaml \
    --tokenizer   configs/tokenizer/speech_llama_tokenizer.yaml \
    --train_data  configs/data/train_sft_multi_task.yaml \
    --output_dir  outputs/novelty_run \
    --iemocap_csv src/data/iemocap_full_dataset.csv \
    --use_meld    True \
    --use_daily_dialog True \
    --run_librispeech_eval True \
    --bf16 True \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --save_steps 1000 \
    --dsacis_hidden_dim 256 \
    --atr_high_threshold 0.65 \
    --atr_low_threshold 0.35 \
    --lambda_da 0.10 \
    --lambda_cong 0.05
```

**Full training pipeline stages:**

1. **Config loading** — Hydra/OmegaConf YAML configs from `../DualSpeechLM/configs/`
2. **Tokenizer + Model init** — Base model loaded via DualSpeechLM infra
3. **DSACIS + ATR construction** — Our novelty modules injected
4. **Auxiliary pre-training** — One epoch of DSACIS head warm-up on:
   - IEMOCAP (SER emotion classification)
   - MELD (emotion + dialogue acts)
   - DailyDialog (dialogue act labels)
5. **Main speech LM training** — NoveltyTrainer (extends DualSpeechLM's CustomTrainer)
6. **LibriSpeech WER evaluation** — Post-training validation on test-clean

---

## Evaluation

After training completes and a checkpoint is saved:

```bash
# Colab:
!cd /content/nlp_project_colabedit/speech && python eval_on_colab.py
```

**What evaluation measures:**

| Metric | Description |
|---|---|
| **Compression Ratio** | Fraction of audio tokens retained after ATR routing (lower = more compression) |
| **Tokens Saved** | Absolute count of tokens dropped by ATR |
| **Generated Text** | Autoregressive output from Phi-3.5-mini on LibriSpeech transcripts |

**Expected output:**

```
================================================================================
SAMPLE 0
================================================================================
TRANSCRIPT:
HE HOPED THERE WOULD BE STEW FOR DINNER ...

GENERATED:
The narrator reflects on simple daily hopes and routines ...

Compression Ratio: 0.6234
Tokens Retained: 28/45
================================================================================

FINAL EVALUATION RESULTS
================================================================================
Mean Compression Ratio: 0.5847
Total Tokens Saved: 87
Original Tokens: 203
Retained Tokens: 116
================================================================================
```

---

## Novel Components — Technical Summary

### DSACIS (`src/novelty/dsacis/importance_scorer.py`)

| Component | Description |
|---|---|
| `encode_text()` | Sentence embedding via `all-MiniLM-L6-v2` (384-dim), with hash fallback |
| `SpeechEmotionRecognizer` | Whisper-feature → 7-class emotion classifier |
| `PragmaticStateGRU` | Multi-turn GRU tracking utterance meaning, dialogue acts, emotion, congruence |
| `DSACISModule` | Top-level wrapper: `process_turn()` → importance score + soft prefix |

**Auxiliary losses:**
- `dialogue_act_loss` — Multi-label BCE on 6 dialogue act categories (question, uncertainty, negation, affirmation, topic_shift, emotion_word)
- `congruence_contrastive_loss` — InfoNCE pushing incongruent text-emotion pairs apart and pulling congruent pairs together

### ATR (`src/novelty/atr/token_router.py`)

| Component | Description |
|---|---|
| `ProsodyFeatureExtractor` | STFT-based: energy, spectral centroid, spectral flux, ZCR |
| `TokenImportanceScorer` | Fuses prosody features with NLP importance score |
| `AdaptiveTokenRouter` | Three-tier routing: HIGH → keep acoustic, MED → keep semantic, LOW → drop |

**Routing logic:**
- Thresholds dynamically adjusted by DSACIS importance score
- Minimum keep ratio of 20% enforced
- Soft masking (×0.25) on removed token embeddings in pipeline

### Pipeline (`src/novelty/pipeline.py`)

`OptimizedSpeechLMPipeline(nn.Module)` wraps everything:
- **Forward pass**: DSACIS → ATR routing → LLM forward → combined loss
- **Inference**: `inference_step()` for eval with `@torch.no_grad()`
- **Total loss**: `L_total = L_generation + λ_da · L_dialogue_act + λ_cong · L_congruence`

---

## Auxiliary Datasets

| Dataset | Source | Role | Loaded By |
|---|---|---|---|
| **IEMOCAP** | Local CSV (`src/data/iemocap_full_dataset.csv`) | SER emotion labels (7-class) | `load_iemocap_splits()` |
| **MELD** | HuggingFace (`declaration-ai/meld`) | Emotion + dialogue act labels | `load_meld_splits()` |
| **DailyDialog** | HuggingFace (`daily_dialog`) | Dialogue act classification | `load_daily_dialog()` |
| **LibriSpeech** | HuggingFace (`openslr/librispeech_asr`, test-clean) | Post-training WER eval only | Streaming in `eval_on_colab.py` |

---

## Checkpoints

Training saves two artifacts:

| File | Contents |
|---|---|
| `checkpoint-step-N/` | LoRA adapter weights (HuggingFace-compatible) |
| `checkpoint-step-N/custom_novelty_modules.pt` | DSACIS + ATR state dicts |

**Colab default save path:** `/content/drive/MyDrive/novelty_outputs/checkpoint-step-100`

To resume or load for evaluation:
```python
# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, checkpoint_dir)

# Load DSACIS + ATR
ckpt = torch.load(os.path.join(checkpoint_dir, "custom_novelty_modules.pt"))
pipeline.dsacis.load_state_dict(ckpt["dsacis"])
pipeline.atr.load_state_dict(ckpt["atr"])
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `sentence-transformers not found` | `pip install sentence-transformers` — falls back to hash encoding otherwise |
| Colab RAM crash | Reduce `batch_size` to 4, or `seq_len` to 128 |
| `RuntimeError: inference mode tensors` | Already fixed — `encode_text()` converts through numpy to strip inference mode |
| MELD download timeout | HuggingFace rate limits — retry or set `HF_DATASETS_CACHE` |
| Checkpoint not found during eval | Ensure Google Drive is mounted and training completed successfully |

---

## Relationship to Base DualSpeechLM

```
nlp_project_colabedit/
├── DualSpeechLM/        # Base framework (configs, data scripts, training infra)
│   ├── configs/         # Hydra YAML configs (model, tokenizer, data)
│   ├── src/             # Base model, trainer, data pipeline
│   └── requirements.txt
│
├── USTokenizer/         # Speech tokenizer (audio → token IDs)
│
└── speech/              # ★ Our extension (this folder)
    ├── src/novelty/     # DSACIS + ATR + Pipeline (novel contribution)
    └── src/data/        # Dataset loaders for auxiliary training
```

Our extension **does not modify** any base DualSpeechLM code. It imports from `DualSpeechLM/src/` when running the full training pipeline (`train_novelty.py`) and is fully self-contained for Colab training (`train_on_colab.py`).
