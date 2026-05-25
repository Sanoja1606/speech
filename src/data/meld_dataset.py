"""
=============================================================================
MELD Dataset Loader
=============================================================================
Loads the MELD dataset from HuggingFace (emotion/sentiment labels on
Friends TV dialogue). Used to:
  • Supplement IEMOCAP SER training with additional emotion labels
  • Provide dialogue_act_labels via sentiment → act heuristic when
    DailyDialog labels are not available

MELD emotion labels → our 7-class mapping:
    neutral    → 0
    joy        → 1 (happy)
    sadness    → 2
    anger      → 3
    fear       → 4
    surprise   → 5
    disgust    → 6

MELD sentiment labels (positive/neutral/negative) are used as a coarse
congruence signal when acoustic emotion is not available.

Usage:
    from src.data.meld_dataset import load_meld_splits
    train_ds, val_ds, test_ds = load_meld_splits()
=============================================================================
"""

from typing import Optional, Tuple, Dict, List
import torch
from torch.utils.data import Dataset

# ── Label mappings ────────────────────────────────────────────────────────────

MELD_EMOTION_MAP: Dict[str, int] = {
    "neutral" : 0,
    "joy"     : 1,
    "sadness" : 2,
    "anger"   : 3,
    "fear"    : 4,
    "surprise": 5,
    "disgust" : 6,
}

# MELD dialogue_act heuristics based on sentiment + emotion
# (used when DailyDialog is unavailable)
# Returns one of: question, inform, directive, commissive
MELD_SENTIMENT_TO_ACT: Dict[str, str] = {
    "positive": "commissive",
    "neutral" : "inform",
    "negative": "directive",
}


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class MELDDataset(Dataset):
    """
    Wraps a HuggingFace MELD split (already loaded).

    Each item returns:
        {
            "utterance"   : str,
            "speaker"     : str,
            "emotion_id"  : int,   # 0-6 mapped from MELD emotion string
            "sentiment"   : str,   # "positive" / "neutral" / "negative"
            "dialogue_act": str,   # heuristic act label
            "season"      : int,
            "episode"     : int,
            "dialogue_id" : int,
        }
    """

    def __init__(self, hf_split):
        """
        Args:
            hf_split : a HuggingFace dataset split object
                       (from datasets.load_dataset("meld_ted", split="train"))
        """
        self.records: List[Dict] = []
        for row in hf_split:
            emo_str = str(row.get("emotion", "neutral")).lower().strip()
            emotion_id = MELD_EMOTION_MAP.get(emo_str, 0)
            sentiment  = str(row.get("sentiment", "neutral")).lower().strip()
            act_label  = MELD_SENTIMENT_TO_ACT.get(sentiment, "inform")

            self.records.append({
                "utterance"   : str(row.get("utterance", "")),
                "speaker"     : str(row.get("speaker", "")),
                "emotion_id"  : emotion_id,
                "sentiment"   : sentiment,
                "dialogue_act": act_label,
                "season"      : int(row.get("season", 0)),
                "episode"     : int(row.get("episode", 0)),
                "dialogue_id" : int(row.get("dialogue_id", 0)),
            })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        return self.records[idx]


# ── Loader ────────────────────────────────────────────────────────────────────

def load_meld_splits(
    hf_dataset_name: str = "declaration-ai/meld",
    trust_remote_code: bool = True,
) -> Tuple[MELDDataset, MELDDataset, MELDDataset]:
    """
    Downloads MELD from HuggingFace and returns (train, val, test) splits.

    The meld_ted (or declaration-ai/meld) variant is used since it has
    clean column names. Falls back gracefully if offline.

    Returns:
        (train_ds, val_ds, test_ds) as MELDDataset instances.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package not found. Install with:\n"
            "    pip install datasets"
        )

    print(f"[MELD] Loading '{hf_dataset_name}' from HuggingFace …")
    raw = load_dataset(hf_dataset_name, trust_remote_code=trust_remote_code)

    train_ds = MELDDataset(raw["train"])
    val_ds   = MELDDataset(raw["validation"])
    test_ds  = MELDDataset(raw["test"])

    print(f"[MELD] Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


# ── Collator ──────────────────────────────────────────────────────────────────

def meld_collate(batch: List[Dict]) -> Dict:
    """
    Minimal collator — returns utterances + emotion_ids for SER training.
    No audio in MELD, so whisper_features are None.
    """
    return {
        "texts"      : [item["utterance"] for item in batch],
        "emotion_ids": [item["emotion_id"] for item in batch],
        "dialogue_acts": [item["dialogue_act"] for item in batch],
    }


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_meld_splits()
    print("Sample:", train_ds[0])
