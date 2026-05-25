"""
=============================================================================
DailyDialog Dataset Loader
=============================================================================
Loads DailyDialog from HuggingFace and maps its dialogue act labels to
the DSACIS scheme used in detect_dialogue_acts() / compute_aux_losses().

DailyDialog act integer labels:
    0=__dummy__  (skip)
    1=inform
    2=question
    3=directive
    4=commissive

Our DSACIS NUM_DIALOGUE_ACTS = 6 (multi-hot):
    ["question","uncertainty","negation","affirmation","topic_shift","emotion_word"]

Mapping strategy:
    DailyDialog act 2 (question)    → DSACIS index 0 (question)
    DailyDialog act 1 (inform)      → DSACIS index 3 (affirmation) — declarative
    DailyDialog act 3 (directive)   → DSACIS index 0 (question) + weight 0.5
    DailyDialog act 4 (commissive)  → DSACIS index 3 (affirmation)
    act 0 → skip

This gives real multi-hot training signal for the dialogue_act_head inside
DSACISModule.compute_aux_losses(), supplementing the keyword-based
detect_dialogue_acts() fallback already in importance_scorer.py.

Usage:
    from src.data.daily_dialog_dataset import load_daily_dialog
    train_ds, val_ds, test_ds = load_daily_dialog()
=============================================================================
"""

import torch
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset

# ── DSACIS dialogue act indices (from importance_scorer.py) ──────────────────
# DIALOGUE_ACTS = ["question","uncertainty","negation","affirmation","topic_shift","emotion_word"]
NUM_DIALOGUE_ACTS = 6

DA_QUESTION    = 0
DA_UNCERTAINTY = 1
DA_NEGATION    = 2
DA_AFFIRMATION = 3
DA_TOPIC_SHIFT = 4
DA_EMOTION     = 5

# DailyDialog integer → DSACIS multi-hot vector
# Returns a float tensor of shape [NUM_DIALOGUE_ACTS]
def _dd_act_to_multihot(dd_act: int) -> torch.Tensor:
    vec = torch.zeros(NUM_DIALOGUE_ACTS, dtype=torch.float32)
    if dd_act == 1:   # inform
        vec[DA_AFFIRMATION] = 1.0
    elif dd_act == 2: # question
        vec[DA_QUESTION] = 1.0
    elif dd_act == 3: # directive (imperative / request)
        vec[DA_QUESTION] = 0.5          # partial overlap
        vec[DA_AFFIRMATION] = 0.5
    elif dd_act == 4: # commissive (promise / offer)
        vec[DA_AFFIRMATION] = 1.0
    # act 0 → zero vector (will be skipped or treated as unknown)
    return vec


# ── Dataset ───────────────────────────────────────────────────────────────────

class DailyDialogDataset(Dataset):
    """
    Wraps a HuggingFace DailyDialog split.

    Each item:
        {
            "utterance"         : str,
            "dialogue_act_id"   : int,           # raw DailyDialog act 0-4
            "dialogue_act_vec"  : torch.Tensor,  # [NUM_DIALOGUE_ACTS] multi-hot
            "emotion_id"        : int,           # DailyDialog emotion 0-6
        }

    DailyDialog emotion integers:
        0=no_emotion, 1=anger, 2=disgust, 3=fear,
        4=happiness, 5=sadness, 6=surprise

    Mapped to our 7-class scheme:
        0=neutral(no_emotion), 1=happy(happiness), 2=sad(sadness),
        3=angry(anger), 4=fearful(fear), 5=surprised(surprise), 6=disgusted(disgust)
    """

    # DailyDialog emotion int → our emotion_id
    _EMO_MAP = {0: 0, 1: 3, 2: 6, 3: 4, 4: 1, 5: 2, 6: 5}

    def __init__(self, hf_split):
        self.records: List[Dict] = []

        for dialogue in hf_split:
            utterances = dialogue["dialog"]
            acts       = dialogue["act"]
            emotions   = dialogue["emotion"]

            for utt, act, emo in zip(utterances, acts, emotions):
                if act == 0:
                    # Skip dummy / unknown acts
                    continue
                self.records.append({
                    "utterance"       : str(utt),
                    "dialogue_act_id" : int(act),
                    "dialogue_act_vec": _dd_act_to_multihot(int(act)),
                    "emotion_id"      : self._EMO_MAP.get(int(emo), 0),
                })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        return self.records[idx]


# ── Loader ────────────────────────────────────────────────────────────────────

def load_daily_dialog() -> Tuple[DailyDialogDataset, DailyDialogDataset, DailyDialogDataset]:
    """
    Downloads DailyDialog from HuggingFace (no login needed) and
    returns (train_ds, val_ds, test_ds).

    One line download:  datasets.load_dataset("daily_dialog")
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    print("[DailyDialog] Downloading from HuggingFace …")
    raw = load_dataset("daily_dialog", trust_remote_code=True)

    train_ds = DailyDialogDataset(raw["train"])
    val_ds   = DailyDialogDataset(raw["validation"])
    test_ds  = DailyDialogDataset(raw["test"])

    print(f"[DailyDialog] Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


# ── Collator for dialogue_act auxiliary training ──────────────────────────────

def daily_dialog_collate(batch: List[Dict]) -> Dict:
    """
    Returns tensors suitable for DSACISModule.compute_aux_losses():
        texts              : List[str]
        dialogue_act_labels: torch.Tensor [B, NUM_DIALOGUE_ACTS]
        emotion_ids        : List[int]
    """
    return {
        "texts"              : [item["utterance"] for item in batch],
        "dialogue_act_labels": torch.stack([item["dialogue_act_vec"] for item in batch]),
        "emotion_ids"        : [item["emotion_id"] for item in batch],
    }


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_daily_dialog()
    print("Sample:", train_ds[0])
    print("Act vec shape:", train_ds[0]["dialogue_act_vec"].shape)
