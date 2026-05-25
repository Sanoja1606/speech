"""
src/data — Dataset loaders for novelty training and evaluation.

Modules:
    iemocap_dataset    : IEMOCAP CSV loader → SER emotion_ids
    meld_dataset       : MELD HuggingFace loader → emotion_ids + dialogue acts
    daily_dialog_dataset: DailyDialog HuggingFace → dialogue_act_labels
    librispeech_eval   : LibriSpeech test-clean WER evaluator (ATR quality check)
"""

from .iemocap_dataset import (
    IEMOCAPDataset,
    load_iemocap_splits,
    iemocap_ser_collate,
    IEMOCAP_LABEL_MAP,
)
from .meld_dataset import (
    MELDDataset,
    load_meld_splits,
    meld_collate,
)
from .daily_dialog_dataset import (
    DailyDialogDataset,
    load_daily_dialog,
    daily_dialog_collate,
)
from .librispeech_eval import (
    run_librispeech_wer_eval,
    load_librispeech_test_clean,
)

__all__ = [
    "IEMOCAPDataset", "load_iemocap_splits", "iemocap_ser_collate", "IEMOCAP_LABEL_MAP",
    "MELDDataset", "load_meld_splits", "meld_collate",
    "DailyDialogDataset", "load_daily_dialog", "daily_dialog_collate",
    "run_librispeech_wer_eval", "load_librispeech_test_clean",
]
