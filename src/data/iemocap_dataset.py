"""
=============================================================================
IEMOCAP Dataset Loader
=============================================================================
Loads the IEMOCAP CSV (iemocap_full_dataset.csv) and maps its emotion labels
to the 7-class scheme used by DSACISModule / SpeechEmotionRecognizer:

    0=neutral, 1=happy, 2=sad, 3=angry,
    4=fearful, 5=surprised, 6=disgusted

IEMOCAP raw labels and their mappings:
    neu → 0 (neutral)
    hap → 1 (happy)
    exc → 1 (excited → happy; closest match)
    sad → 2 (sad)
    ang → 3 (angry)
    fru → 3 (frustrated → angry; closest match)
    fea → 4 (fearful)
    sur → 5 (surprised)
    dis → 6 (disgusted)
    xxx → skipped (unclear/no consensus)
    oth → skipped

Usage:
    from src.data.iemocap_dataset import IEMOCAPDataset, load_iemocap_splits
    train_ds, val_ds = load_iemocap_splits("src/data/iemocap_full_dataset.csv")
=============================================================================
"""

import csv
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset

# ── Emotion label mapping ────────────────────────────────────────────────────

# Maps IEMOCAP raw labels → our 7-class ID
IEMOCAP_LABEL_MAP: Dict[str, int] = {
    "neu": 0,
    "hap": 1,
    "exc": 1,   # excited ≈ happy
    "sad": 2,
    "ang": 3,
    "fru": 3,   # frustrated ≈ angry
    "fea": 4,
    "sur": 5,
    "dis": 6,
}

# Labels to skip (no consensus / other)
_SKIP_LABELS = {"xxx", "oth"}

# Canonical emotion names used by DSACISModule
EMOTION_LABELS = {
    0: "neutral", 1: "happy", 2: "sad",    3: "angry",
    4: "fearful", 5: "surprised", 6: "disgusted",
}


# ── Dataset class ─────────────────────────────────────────────────────────────

class IEMOCAPDataset(Dataset):
    """
    PyTorch Dataset wrapping the IEMOCAP CSV.

    Each item returns:
        {
            "path"       : str,       # relative wav path from CSV
            "emotion_id" : int,       # mapped 0-6
            "session"    : int,       # IEMOCAP session 1-5
            "gender"     : str,       # "M" or "F"
            "agreement"  : int,       # annotator agreement count
        }

    Audio is NOT loaded here — the path is returned so the caller
    (training collator) can load waveforms as needed (or skip for
    text-only runs using dialogue_act supervision from DailyDialog).

    Args:
        csv_path    : path to iemocap_full_dataset.csv
        sessions    : list of session IDs to include (1-5).
                      Default None = all sessions.
        min_agreement: minimum annotator agreement to include a row.
                      Set to 2 to keep only majority-agreed labels.
    """

    def __init__(
        self,
        csv_path: str,
        sessions: Optional[List[int]] = None,
        min_agreement: int = 1,
    ):
        self.csv_path = csv_path
        self.records: List[Dict] = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                emo_raw = row["emotion"].strip().lower()
                if emo_raw in _SKIP_LABELS:
                    continue
                if emo_raw not in IEMOCAP_LABEL_MAP:
                    continue

                session_id = int(row["session"])
                if sessions is not None and session_id not in sessions:
                    continue

                agreement = int(row["agreement"])
                if agreement < min_agreement:
                    continue

                self.records.append({
                    "path"       : row["path"].strip(),
                    "emotion_id" : IEMOCAP_LABEL_MAP[emo_raw],
                    "session"    : session_id,
                    "gender"     : row["gender"].strip(),
                    "agreement"  : agreement,
                    "raw_emotion": emo_raw,
                })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        return self.records[idx]

    def emotion_distribution(self) -> Dict[str, int]:
        """Utility: count per emotion class."""
        from collections import Counter
        return dict(Counter(
            EMOTION_LABELS[r["emotion_id"]] for r in self.records
        ))


# ── Convenience split loader ─────────────────────────────────────────────────

def load_iemocap_splits(
    csv_path: str,
    val_sessions: Optional[List[int]] = None,
    min_agreement: int = 2,
    seed: int = 42,
) -> Tuple["IEMOCAPDataset", "IEMOCAPDataset"]:
    """
    Returns (train_dataset, val_dataset).

    Standard IEMOCAP leave-one-session-out: session 5 → val by default.
    Set min_agreement=2 to keep only rows where ≥2 annotators agreed.

    Args:
        csv_path      : path to iemocap_full_dataset.csv
        val_sessions  : sessions to hold out for val (default [5])
        min_agreement : minimum annotator agreement (default 2)
        seed          : random seed (unused here, kept for API consistency)
    """
    if val_sessions is None:
        val_sessions = [5]

    all_sessions = [1, 2, 3, 4, 5]
    train_sessions = [s for s in all_sessions if s not in val_sessions]

    train_ds = IEMOCAPDataset(csv_path, sessions=train_sessions, min_agreement=min_agreement)
    val_ds   = IEMOCAPDataset(csv_path, sessions=val_sessions,   min_agreement=min_agreement)

    return train_ds, val_ds


# ── Collator for SER-only fine-tuning ────────────────────────────────────────

def iemocap_ser_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Minimal collator for SER-only training (no audio loading).
    Returns just the emotion_id tensor — audio must be loaded separately
    by a waveform-aware collator if whisper_features are needed.

    For full pipeline training (with whisper_features), replace this
    collator with one that loads wav files from `record["path"]`.
    """
    emotion_ids = torch.tensor(
        [item["emotion_id"] for item in batch], dtype=torch.long
    )
    return {
        "emotion_ids": emotion_ids,
        "paths"      : [item["path"] for item in batch],
    }


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "src/data/iemocap_full_dataset.csv"

    train_ds, val_ds = load_iemocap_splits(csv_path)
    print(f"Train: {len(train_ds)} samples   Val: {len(val_ds)} samples")
    print("Train emotion distribution:", train_ds.emotion_distribution())
    print("Val   emotion distribution:", val_ds.emotion_distribution())
    print("Sample record:", train_ds[0])
