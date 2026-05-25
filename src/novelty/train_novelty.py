"""
=============================================================================
NOVELTY TRAINER — with full dataset integration
=============================================================================
Datasets integrated:
  • IEMOCAP   (src/data/iemocap_full_dataset.csv) — SER emotion labels
  • MELD       (HuggingFace: declaration-ai/meld)  — SER + dialogue acts
  • DailyDialog(HuggingFace: daily_dialog)          — dialogue_act_labels
  • LibriSpeech(HuggingFace: test-clean only)       — post-training WER eval

How to use:
    python src/novelty/train_novelty.py \
        --model  configs/model/Phi-3.5-mini-instruct_lora.yaml \
        --tokenizer configs/tokenizer/speech_llama_tokenizer.yaml \
        --train_data configs/data/train_sft_multi_task.yaml \
        --output_dir outputs/novelty_run \
        --iemocap_csv src/data/iemocap_full_dataset.csv \
        --use_meld True \
        --use_daily_dialog True \
        --run_librispeech_eval True \
        --bf16 True \
        --per_device_train_batch_size 4
"""

import os, sys, logging, random
import numpy as np
import torch
import transformers
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import OmegaConf
import hydra
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.train.trainer import CustomTrainer, compute_metrics
from src.novelty.dsacis.importance_scorer import DSACISModule
from src.novelty.atr.token_router import AdaptiveTokenRouter
from src.novelty.pipeline import OptimizedSpeechLMPipeline
from src.data.iemocap_dataset import load_iemocap_splits
from src.data.meld_dataset import load_meld_splits
from src.data.daily_dialog_dataset import load_daily_dialog
from src.data.librispeech_eval import run_librispeech_wer_eval

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)


@dataclass
class ConfigPathArguments:
    model:       Optional[str] = field(default=None)
    tokenizer:   Optional[str] = field(default=None)
    train_data:  Optional[str] = field(default=None)
    eval_data:   Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir:                  str   = field(default="outputs/novelty")
    overwrite_output_dir:        bool  = field(default=False)
    optim:                       str   = field(default="adamw_hf")
    gradient_accumulation_steps: int   = field(default=1)
    learning_rate:               float = field(default=5e-5)
    min_lr_ratio:                float = field(default=0.1)
    weight_decay:                float = field(default=0.0)
    num_train_epochs:            float = field(default=3.0)
    max_steps:                   int   = field(default=-1)
    lr_scheduler_type:           str   = field(default="cosine")
    save_steps:                  int   = field(default=1000)
    bf16:                        bool  = field(default=False)
    fp16:                        bool  = field(default=False)
    dataloader_num_workers:      int   = field(default=4)
    per_device_train_batch_size: int   = field(default=4)
    per_device_eval_batch_size:  int   = field(default=4)
    run_name:                    str   = field(default="novelty_run")
    dsacis_hidden_dim:           int   = field(default=256)
    atr_high_threshold:          float = field(default=0.65)
    atr_low_threshold:           float = field(default=0.35)
    lambda_da:                   float = field(default=0.10)
    lambda_cong:                 float = field(default=0.05)
    # Dataset flags
    iemocap_csv:                 Optional[str] = field(default="src/data/iemocap_full_dataset.csv")
    iemocap_min_agreement:       int   = field(default=2)
    use_meld:                    bool  = field(default=True)
    use_daily_dialog:            bool  = field(default=True)
    run_librispeech_eval:        bool  = field(default=True)
    librispeech_max_samples:     int   = field(default=500)


id2task = {1:"asr", 2:"tts", 3:"vc", 4:"t2st", 5:"sc", 6:"sqa", 7:"s2tt", 8:"ser"}


# ─── Auxiliary dataset warm-up ────────────────────────────────────────────────

def _aux_warmup(pipeline, dataset, batch_size, device, optimizer, source,
                has_dialogue_acts=False):
    """One pass through a text+emotion dataset to warm up DSACIS heads."""
    from src.novelty.dsacis.importance_scorer import (
        encode_text, NUM_EMOTIONS, NUM_DIALOGUE_ACTS, detect_dialogue_acts,
    )
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    total_loss, steps = 0.0, 0
    for start in range(0, len(indices), batch_size):
        batch = [dataset[i] for i in indices[start:start + batch_size]]
        B = len(batch)
        emotion_ids = torch.tensor([item["emotion_id"] for item in batch],
                                   dtype=torch.long, device=device)
        texts = [item.get("utterance", "") for item in batch]
        optimizer.zero_grad()
        text_embs = torch.cat([encode_text(t, device) for t in texts], dim=0)
        emo_ohs = torch.zeros(B, NUM_EMOTIONS, device=device)
        for b, eid in enumerate(emotion_ids.tolist()):
            if 0 <= eid < NUM_EMOTIONS:
                emo_ohs[b, eid] = 1.0
        cong_labels = torch.ones(B, device=device)
        act_vecs = (
            torch.stack([detect_dialogue_acts(t).to(device) for t in texts], dim=0)
            if has_dialogue_acts
            else torch.zeros(B, NUM_DIALOGUE_ACTS, device=device)
        )
        _, _, d_t = pipeline.dsacis.process_turn(
            text=texts[0] if texts[0] else "hello",
            emotion_id=emotion_ids[0].item(),
        )
        loss_da, loss_cong = pipeline.dsacis.compute_aux_losses(
            d_t=d_t.expand(B, -1),
            text_embeddings=text_embs,
            emotion_one_hots=emo_ohs,
            congruence_labels=cong_labels,
            dialogue_act_labels=act_vecs,
        )
        loss = loss_da + loss_cong
        if loss.requires_grad:
            loss.backward(); optimizer.step()
            total_loss += loss.item(); steps += 1
    if steps:
        logger.info(f"  [{source}] avg loss {total_loss/steps:.4f} over {steps} steps")


def _da_warmup(pipeline, dataset, batch_size, device, optimizer):
    """Dialogue-act warm-up using DailyDialog labels."""
    from src.novelty.dsacis.importance_scorer import encode_text, NUM_EMOTIONS
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    total_loss, steps = 0.0, 0
    for start in range(0, len(indices), batch_size):
        batch = [dataset[i] for i in indices[start:start + batch_size]]
        B = len(batch)
        texts = [item["utterance"] for item in batch]
        act_vecs = torch.stack([item["dialogue_act_vec"].to(device) for item in batch], dim=0)
        emotion_ids = [item["emotion_id"] for item in batch]
        optimizer.zero_grad()
        text_embs = torch.cat([encode_text(t, device) for t in texts], dim=0)
        emo_ohs = torch.zeros(B, NUM_EMOTIONS, device=device)
        for b, eid in enumerate(emotion_ids):
            if 0 <= eid < NUM_EMOTIONS:
                emo_ohs[b, eid] = 1.0
        _, _, d_t = pipeline.dsacis.process_turn(
            text=texts[0] if texts[0] else "hello",
            emotion_id=emotion_ids[0],
        )
        loss_da, loss_cong = pipeline.dsacis.compute_aux_losses(
            d_t=d_t.expand(B, -1),
            text_embeddings=text_embs,
            emotion_one_hots=emo_ohs,
            congruence_labels=torch.ones(B, device=device),
            dialogue_act_labels=act_vecs,
        )
        loss = loss_da + 0.1 * loss_cong
        if loss.requires_grad:
            loss.backward(); optimizer.step()
            total_loss += loss.item(); steps += 1
    if steps:
        logger.info(f"  [DailyDialog] avg loss {total_loss/steps:.4f} over {steps} steps")


def run_auxiliary_dataset_epoch(pipeline, iemocap_train=None, meld_train=None,
                                 daily_dialog_train=None, batch_size=32, device=None):
    if device is None:
        device = next(pipeline.parameters()).device
    optimizer = torch.optim.Adam(list(pipeline.dsacis.parameters()), lr=1e-4)
    pipeline.dsacis.train()
    logger.info("=== Auxiliary dataset pre-training (DSACIS heads) ===")
    if iemocap_train:
        logger.info(f"  IEMOCAP: {len(iemocap_train)} samples")
        _aux_warmup(pipeline, iemocap_train, batch_size, device, optimizer, "IEMOCAP")
    if meld_train:
        logger.info(f"  MELD: {len(meld_train)} samples")
        _aux_warmup(pipeline, meld_train, batch_size, device, optimizer, "MELD",
                    has_dialogue_acts=True)
    if daily_dialog_train:
        logger.info(f"  DailyDialog: {len(daily_dialog_train)} samples")
        _da_warmup(pipeline, daily_dialog_train, batch_size, device, optimizer)
    pipeline.dsacis.eval()
    logger.info("=== Auxiliary pre-training complete ===")


# ─── Novelty Trainer (unchanged logic) ────────────────────────────────────────

class NoveltyTrainer(CustomTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids      = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels         = inputs.get("labels", None)
        target_audio_ids = inputs.get("target_audio_ids", None)
        spk_emb        = inputs.get("spk_emb", None)
        attention_mask_question = inputs.get("attention_mask_question", None)
        attention_mask_answer   = inputs.get("attention_mask_answer",   None)
        task_id        = inputs.get("task_id", None)
        try:
            texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        except Exception:
            texts = [""] * input_ids.shape[0]
        result = model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels if labels is not None else torch.full_like(input_ids, -100),
            target_audio_ids=target_audio_ids, spk_emb=spk_emb, texts=texts,
            whisper_features=None, emotion_ids=None, dialogue_act_labels=None,
            waveforms=None, attention_mask_question=attention_mask_question,
            attention_mask_answer=attention_mask_answer,
        )
        loss = result["loss"]
        if self.state.is_world_process_zero:
            if self.state.global_step % self.args.gradient_accumulation_steps == 0:
                breakdown = result.get("loss_breakdown", {})
                routing   = result.get("routing_stats", [{}])
                log_dict  = dict(breakdown)
                if routing and isinstance(routing[0], dict):
                    log_dict["token_compression_ratio"] = routing[0].get("compression_ratio", 0.0)
                if task_id is not None:
                    try:
                        log_dict["task"] = id2task.get(task_id[0][0].item(), "unknown")
                    except Exception:
                        pass
                if log_dict:
                    self.log(log_dict)
        return (loss, result) if return_outputs else loss


# ─── Entry point ──────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def train():
    parser = transformers.HfArgumentParser((ConfigPathArguments, TrainingArguments))
    cfg_path, training_args = parser.parse_args_into_dataclasses()
    set_seed(42)

    train_data_cfg = OmegaConf.load(cfg_path.train_data)
    model_cfg      = OmegaConf.load(cfg_path.model)
    tokenizer_cfg  = OmegaConf.load(cfg_path.tokenizer)

    logger.info("Loading tokenizer …")
    tokenizer = hydra.utils.instantiate(tokenizer_cfg)
    tokenizer.pad_token = tokenizer.unk_token

    logger.info("Loading training data …")
    train_data = hydra.utils.instantiate(train_data_cfg, tokenizer=tokenizer)

    logger.info("Loading base model …")
    use_peft = "peft" in model_cfg._target_ or "lora" in model_cfg._target_
    base_model = (hydra.utils.instantiate(model_cfg, tokenizer=tokenizer) if use_peft
                  else hydra.utils.instantiate(model_cfg))
    if not use_peft:
        base_model.resize_token_embeddings(len(tokenizer))

    eval_data = None
    if cfg_path.eval_data:
        eval_data = hydra.utils.instantiate(OmegaConf.load(cfg_path.eval_data), tokenizer=tokenizer)

    logger.info("Building DSACIS …")
    dsacis = DSACISModule(hidden_dim=training_args.dsacis_hidden_dim,
                          output_dim=base_model.config.hidden_size,
                          num_gru_layers=2, whisper_dim=1280)
    logger.info("Building ATR …")
    atr = AdaptiveTokenRouter(high_threshold=training_args.atr_high_threshold,
                               low_threshold=training_args.atr_low_threshold,
                               min_keep_ratio=0.20)
    logger.info("Wrapping in OptimizedSpeechLMPipeline …")
    pipeline = OptimizedSpeechLMPipeline(base_model=base_model, dsacis=dsacis, atr=atr)
    pipeline.multi_task_loss.lambda_da   = training_args.lambda_da
    pipeline.multi_task_loss.lambda_cong = training_args.lambda_cong
    pipeline.config = base_model.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load datasets ──────────────────────────────────────────────────────
    iemocap_train = meld_train = dd_train = None

    if training_args.iemocap_csv and os.path.exists(training_args.iemocap_csv):
        try:
            iemocap_train, _ = load_iemocap_splits(
                training_args.iemocap_csv,
                min_agreement=training_args.iemocap_min_agreement,
            )
            logger.info(f"IEMOCAP loaded: {len(iemocap_train)} train samples | "
                        f"{iemocap_train.emotion_distribution()}")
        except Exception as e:
            logger.warning(f"IEMOCAP skipped: {e}")
    else:
        logger.warning(f"IEMOCAP CSV not found at '{training_args.iemocap_csv}' — skipping.")

    if training_args.use_meld:
        try:
            meld_train, _, _ = load_meld_splits()
            logger.info(f"MELD loaded: {len(meld_train)} train samples")
        except Exception as e:
            logger.warning(f"MELD skipped: {e}")

    if training_args.use_daily_dialog:
        try:
            dd_train, _, _ = load_daily_dialog()
            logger.info(f"DailyDialog loaded: {len(dd_train)} train samples")
        except Exception as e:
            logger.warning(f"DailyDialog skipped: {e}")

    # ── Auxiliary pre-training ─────────────────────────────────────────────
    if any(x is not None for x in [iemocap_train, meld_train, dd_train]):
        pipeline = pipeline.to(device)
        run_auxiliary_dataset_epoch(
            pipeline=pipeline,
            iemocap_train=iemocap_train,
            meld_train=meld_train,
            daily_dialog_train=dd_train,
            batch_size=training_args.per_device_train_batch_size,
            device=device,
        )

    # ── Main speech LM training ────────────────────────────────────────────
    logger.info("Starting main novelty training …")
    pipeline.base_model.config.use_cache = False
    trainer = NoveltyTrainer(
        model=pipeline, args=training_args,
        train_dataset=train_data, eval_dataset=eval_data,
        tokenizer=tokenizer, compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    logger.info("Training complete.")

    # ── LibriSpeech WER eval ───────────────────────────────────────────────
    if training_args.run_librispeech_eval:
        try:
            results = run_librispeech_wer_eval(
                pipeline=pipeline, tokenizer=tokenizer, device=device,
                max_samples=training_args.librispeech_max_samples,
            )
            logger.info(f"LibriSpeech WER: {results['wer']:.4f}  "
                        f"compression: {results['mean_compression']:.4f}  "
                        f"tokens saved: {results['total_tokens_saved']}")
        except Exception as e:
            logger.warning(f"LibriSpeech WER eval failed: {e}")


if __name__ == "__main__":
    train()
