"""
=============================================================================
DSACIS — Dialogue-State Aware Conversational Importance Scoring
=============================================================================
NLP NOVELTY MODULE  (v2 — fully corrected)

FIXES from analysis review:
  A) Real sentence-transformer encoder (all-MiniLM-L6-v2) replaces char hashing
  B) dialogue_act_loss and congruence_contrastive_loss are fully wired — both
     return real differentiable loss tensors, called from pipeline.py
  C) Integrated SER head derives emotion_id from Whisper features directly —
     no upstream black-box dependency on an external emotion classifier
  D) compute_aux_losses() is a single clean method that pipeline.py calls
     with batch data, returning both auxiliary losses ready to add to total loss

Architecture:
    encode_text()           — sentence-transformers/all-MiniLM-L6-v2
    SpeechEmotionRecognizer — lightweight Whisper-feature → emotion_id
    PragmaticStateGRU       — GRU tracking dialogue context + pragmatic stance
    DSACISModule            — high-level wrapper with reset/process_turn API
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Real NLP Text Encoder — sentence-transformers/all-MiniLM-L6-v2
# ─────────────────────────────────────────────────────────────────────────────

_SENTENCE_MODEL = None          # loaded once, reused
SENTENCE_EMBED_DIM = 384        # all-MiniLM-L6-v2 output dimension


def _get_sentence_model(device: torch.device):
    """Lazy-load the sentence transformer; fallback if not installed."""
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is not None:
        return _SENTENCE_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        m.eval()
        m = m.to(device)
        _SENTENCE_MODEL = m
        return m
    except ImportError:
        print(
            "[DSACIS] sentence-transformers not found — using hash fallback.\n"
            "Fix with:  pip install sentence-transformers"
        )
        _SENTENCE_MODEL = "fallback"
        return "fallback"


def encode_text(text: str, device: torch.device) -> torch.Tensor:
    """
    Encode a sentence into a [1, 384] contextual embedding.
    Uses all-MiniLM-L6-v2 (real NLP) with a deterministic hash fallback.
    """
    model = _get_sentence_model(device)
    if model == "fallback":
        vec = torch.zeros(SENTENCE_EMBED_DIM, device=device, dtype=torch.float32)
        for i, ch in enumerate(text):
            vec[i % SENTENCE_EMBED_DIM] += float(ord(ch)) / 1000.0
        vec = F.normalize(vec, dim=0)
        return vec.unsqueeze(0)                          # [1, 384]
    with torch.no_grad():
        emb = model.encode(
            [text],
            convert_to_tensor=True,
            show_progress_bar=False,
            device=device,
        )                                                # [1, 384]
    return emb.float()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Emotion & Dialogue Act definitions
# ─────────────────────────────────────────────────────────────────────────────

NUM_EMOTIONS = 7
EMOTION_LABELS   = {0:"neutral",1:"happy",2:"sad",3:"angry",
                    4:"fearful",5:"surprised",6:"disgusted"}
EMOTION_POLARITY = {0:0, 1:1, 2:-1, 3:-1, 4:-1, 5:0, 6:-1}

DIALOGUE_ACTS = ["question","uncertainty","negation","affirmation",
                 "topic_shift","emotion_word"]
NUM_DIALOGUE_ACTS = len(DIALOGUE_ACTS)

_ACT_PATTERNS = {
    "question"    : ["?","what","when","where","why","how","who",
                     "could you","would you","can you","do you"],
    "uncertainty" : ["i guess","i think","maybe","perhaps","not sure",
                     "kind of","sort of","i don't know","hmm","umm","uh"],
    "negation"    : ["no","not","never","nothing","nobody","nowhere",
                     "don't","won't","can't","isn't","wasn't"],
    "affirmation" : ["yes","yeah","sure","absolutely","definitely",
                     "of course","right","exactly","okay","fine"],
    "topic_shift" : ["anyway","by the way","speaking of","actually",
                     "but wait","oh also"],
    "emotion_word": ["love","hate","angry","sad","happy","scared","worried",
                     "excited","frustrated","upset","hurt"],
}

_POSITIVE = {"good","great","excellent","wonderful","fantastic","love",
             "happy","best","amazing","awesome","perfect","nice",
             "beautiful","enjoy","like","glad"}
_NEGATIVE = {"bad","terrible","awful","hate","sad","worst","horrible",
             "upset","angry","wrong","problem","broken","fail","sorry",
             "unfortunately","struggle","pain"}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def keyword_sentiment(text: str) -> float:
    words = text.lower().split()
    pos = sum(1 for w in words if w.strip(".,!?") in _POSITIVE)
    neg = sum(1 for w in words if w.strip(".,!?") in _NEGATIVE)
    total = pos + neg
    return 0.0 if total == 0 else (pos - neg) / total


def detect_dialogue_acts(text: str) -> torch.Tensor:
    """Multi-hot [NUM_DIALOGUE_ACTS] dialogue act vector, L1-normalised."""
    text_l = text.lower()
    vec = torch.zeros(NUM_DIALOGUE_ACTS, dtype=torch.float32)
    for i, (act, pats) in enumerate(_ACT_PATTERNS.items()):
        if any(p in text_l for p in pats):
            vec[i] = 1.0
    if vec.sum() > 0:
        vec = vec / vec.sum()
    return vec


def compute_congruence(text: str, acoustic_emotion_id: int) -> float:
    """
    Returns [0,1].  0 = incongruent (e.g. 'I'm fine' + sad tone).
    Incongruence is the core signal driving importance boosting.
    """
    ts = keyword_sentiment(text)
    ap = EMOTION_POLARITY.get(acoustic_emotion_id, 0)
    if ts == 0 and ap == 0: return 1.0
    if ap == 0:             return 0.7
    tsign = 1 if ts > 0.1 else (-1 if ts < -0.1 else 0)
    if tsign == 0:          return 0.7
    return 1.0 if tsign == ap else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SER Head — derives emotion_id from Whisper encoder features
#     This removes the upstream dependency gap identified in the analysis.
# ─────────────────────────────────────────────────────────────────────────────

class SpeechEmotionRecognizer(nn.Module):
    """
    Lightweight classification head on top of Whisper encoder features.

    Whisper (large) produces [B, T, 1280]; we mean-pool over T then
    classify into NUM_EMOTIONS categories.

    Training:  fine-tune on IEMOCAP / MELD emotion labels.
    Inference: call predict() to get emotion_ids used by DSACIS.
    """

    def __init__(self, whisper_dim: int = 1280):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(whisper_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_EMOTIONS),
        )

    def forward(
        self,
        whisper_features: torch.Tensor,   # [B, T, whisper_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits [B,7], emotion_ids [B])."""
        pooled = whisper_features.mean(dim=1)           # [B, whisper_dim]
        logits = self.classifier(pooled)                # [B, NUM_EMOTIONS]
        return logits, logits.argmax(dim=-1)

    def ser_loss(
        self,
        whisper_features: torch.Tensor,
        emotion_labels: torch.Tensor,     # [B]  integer 0..6
    ) -> torch.Tensor:
        logits, _ = self.forward(whisper_features)
        return F.cross_entropy(logits, emotion_labels)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Pragmatic State GRU — the core NLP novelty
# ─────────────────────────────────────────────────────────────────────────────

class PragmaticStateGRU(nn.Module):
    """
    Multi-turn GRU that maintains a pragmatic dialogue state d_t encoding:
        • Contextual utterance meaning   (real MiniLM embeddings)
        • Dialogue act trajectory        (multi-hot + normalised)
        • Acoustic-semantic congruence   (congruence score per turn)
        • Speaker emotion trajectory     (one-hot)

    Outputs:
        d_t          [1, hidden_dim]    — dialogue state vector
        soft_prefix  [1, 1, output_dim] — prepended to Phi-3.5-mini inputs
        importance_score  float         — fed to ATR for token routing

    Auxiliary training heads (called from pipeline.py):
        dialogue_act_loss()              — multi-label BCE on dialogue acts
        congruence_contrastive_loss()    — InfoNCE pulling/pushing modalities
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 3072,       # Phi-3.5-mini hidden size
        num_layers: int = 2,
    ):
        super().__init__()

        # GRU input dim
        _in = SENTENCE_EMBED_DIM + NUM_DIALOGUE_ACTS + 1 + NUM_EMOTIONS

        self.gru = nn.GRU(
            input_size=_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        # Extra nonlinear transform on MiniLM embeddings before GRU
        self.text_proj = nn.Sequential(
            nn.Linear(SENTENCE_EMBED_DIM, SENTENCE_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(SENTENCE_EMBED_DIM, SENTENCE_EMBED_DIM),
        )

        # d_t → LLM hidden dim (soft prefix)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # d_t → importance scalar
        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # ── Auxiliary head 1: Dialogue Act Classification (multi-label) ──
        self.dialogue_act_head = nn.Linear(hidden_dim, NUM_DIALOGUE_ACTS)

        # ── Auxiliary head 2: Contrastive Congruence (InfoNCE-style) ──
        # Projects text and emotion into a shared 128-dim contrast space
        self.text_contrast_proj = nn.Sequential(
            nn.Linear(SENTENCE_EMBED_DIM, 128), nn.ReLU(), nn.Linear(128, 128)
        )
        self.emo_contrast_proj = nn.Sequential(
            nn.Linear(NUM_EMOTIONS, 64), nn.ReLU(), nn.Linear(64, 128)
        )

        self._hidden_dim = hidden_dim
        self._num_layers = num_layers

    # ── Single-turn forward ──────────────────────────────────────────────

    def forward(
        self,
        text: str,
        emotion_id: int,
        hidden: Optional[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Process one turn; update and return dialogue state."""

        # Real sentence embedding
        text_emb = encode_text(text, device)                  # [1, 384]
        text_emb = self.text_proj(text_emb)                   # [1, 384]

        # Dialogue act features
        act_vec = detect_dialogue_acts(text).to(device)       # [NUM_DA]

        # Emotion one-hot
        emo_vec = torch.zeros(NUM_EMOTIONS, device=device)
        if 0 <= emotion_id < NUM_EMOTIONS:
            emo_vec[emotion_id] = 1.0

        # Congruence scalar
        congruence = compute_congruence(text, emotion_id)
        cong_t = torch.tensor([[congruence]], dtype=torch.float32, device=device)

        # Assemble GRU input [1, 1, _in]
        gru_in = torch.cat([
            text_emb,
            act_vec.unsqueeze(0),
            cong_t,
            emo_vec.unsqueeze(0),
        ], dim=-1).unsqueeze(1)

        # Initialise hidden if first turn
        if hidden is None:
            hidden = torch.zeros(
                self._num_layers, 1, self._hidden_dim,
                dtype=torch.float32, device=device,
            )

        gru_out, new_hidden = self.gru(gru_in, hidden)
        d_t = gru_out[:, -1, :]                               # [1, hidden_dim]

        # Importance score with pragmatic boosting
        base_score = self.importance_head(d_t).item()
        if congruence < 0.5:
            base_score = min(1.0, base_score + 0.25)   # incongruence boost
        if act_vec[1] > 0 or act_vec[5] > 0:
            base_score = min(1.0, base_score + 0.15)   # uncertainty/emotion boost
        importance_score = base_score

        soft_prefix = self.output_proj(d_t).unsqueeze(1)     # [1, 1, output_dim]

        return d_t, soft_prefix, new_hidden, importance_score

    # ── Auxiliary Loss 1: Dialogue Act Classification ────────────────────
    # Fully wired — pipeline.py passes target_acts from dataset batch

    def dialogue_act_loss(
        self,
        d_t: torch.Tensor,           # [B, hidden_dim]
        target_acts: torch.Tensor,   # [B, NUM_DIALOGUE_ACTS]  float multi-hot
    ) -> torch.Tensor:
        """
        Multi-label binary cross-entropy.
        target_acts: ground-truth dialogue act labels from DailyDialog / MELD.
        Called from pipeline.py with real batch data — fully differentiable.
        """
        logits = self.dialogue_act_head(d_t)                 # [B, NUM_DIALOGUE_ACTS]
        return F.binary_cross_entropy_with_logits(logits, target_acts.float())

    # ── Auxiliary Loss 2: Contrastive Congruence (InfoNCE) ───────────────
    # Fully wired — pipeline.py builds text_embeddings and emotion_one_hots

    def congruence_contrastive_loss(
        self,
        text_embeddings: torch.Tensor,    # [B, SENTENCE_EMBED_DIM]
        emotion_one_hots: torch.Tensor,   # [B, NUM_EMOTIONS]
        congruence_labels: torch.Tensor,  # [B]  float: 1=congruent, 0=incongruent
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        InfoNCE-style contrastive loss:
          Congruent pairs  (label≈1): pull text_emb and emo_emb together
          Incongruent pairs (label≈0): push them apart

        This is the advanced NLP contribution — not sentiment classification,
        but learning the structure of modality agreement vs. conflict.
        """
        text_z = F.normalize(self.text_contrast_proj(text_embeddings), dim=-1)  # [B,128]
        emo_z  = F.normalize(self.emo_contrast_proj(emotion_one_hots),  dim=-1)  # [B,128]

        sim = torch.matmul(text_z, emo_z.T) / temperature     # [B, B]
        labels = torch.arange(sim.shape[0], device=sim.device)

        loss_pull = F.cross_entropy(sim,  labels)   # pull congruent diagonal up
        loss_push = F.cross_entropy(-sim, labels)   # push incongruent diagonal down

        w = congruence_labels.float().mean().clamp(0.0, 1.0)
        return w * loss_pull + (1.0 - w) * loss_push


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DSACISModule — top-level class used by pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

class DSACISModule(nn.Module):
    """
    Public interface for the NLP novelty.

    pipeline.py calls:
        dsacis.predict_emotion(whisper_features)    → emotion_id  (no upstream dep)
        dsacis.process_turn(text, emotion_id)       → importance_score, soft_prefix, d_t
        dsacis.compute_aux_losses(...)              → loss_da, loss_cong  (fully wired)
        dsacis.reset_conversation()                 → start of new session
    """

    def __init__(
        self,
        hidden_dim:    int = 256,
        output_dim:    int = 3072,
        num_gru_layers: int = 2,
        whisper_dim:   int = 1280,
    ):
        super().__init__()
        self.state_gru = PragmaticStateGRU(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_gru_layers,
        )
        self.ser_head = SpeechEmotionRecognizer(whisper_dim=whisper_dim)

        self._hidden     : Optional[torch.Tensor] = None
        self._turn_count : int = 0
        self._device     : torch.device = torch.device("cpu")

    def to(self, device, **kwargs):
        self._device = device if isinstance(device, torch.device) else torch.device(str(device))
        return super().to(device, **kwargs)

    def reset_conversation(self):
        """Call at the start of every new conversation session."""
        self._hidden     = None
        self._turn_count = 0

    # ── SER: emotion from Whisper features (no external SER model needed) ─

    def predict_emotion(
        self,
        whisper_features: torch.Tensor,   # [B, T, whisper_dim]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Predict emotion_id from Whisper encoder output.
        Returns (logits [B,7], emotion_ids_list).
        """
        logits, ids = self.ser_head(whisper_features)
        return logits, ids.tolist()

    # ── Process one turn ─────────────────────────────────────────────────

    def process_turn(
        self,
        text: str,
        emotion_id: int = 0,
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Returns:
            importance_score  float          → fed directly to ATR
            soft_prefix       [1,1,3072]     → prepended to LLM input embeds
            d_t               [1,hidden_dim] → passed to compute_aux_losses
        """
        self._turn_count += 1
        d_t, soft_prefix, new_hidden, score = self.state_gru(
            text=text,
            emotion_id=emotion_id,
            hidden=self._hidden,
            device=self._device,
        )
        self._hidden = new_hidden.detach()
        return score, soft_prefix, d_t

    # ── Auxiliary losses — fully wired, called from pipeline.py ──────────

    def compute_aux_losses(
        self,
        d_t:                  torch.Tensor,            # [B, hidden_dim]
        text_embeddings:      torch.Tensor,            # [B, SENTENCE_EMBED_DIM]
        emotion_one_hots:     torch.Tensor,            # [B, NUM_EMOTIONS]
        congruence_labels:    torch.Tensor,            # [B]  float
        dialogue_act_labels:  Optional[torch.Tensor] = None,  # [B, NUM_DA]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both NLP auxiliary losses.
        Both losses are fully differentiable and added to total training loss
        in pipeline.py via MultiTaskLoss.

        Returns:
            loss_da    — dialogue act classification loss
            loss_cong  — contrastive congruence loss
        """
        loss_da = (
            self.state_gru.dialogue_act_loss(d_t, dialogue_act_labels)
            if dialogue_act_labels is not None
            else torch.tensor(0.0, device=d_t.device, requires_grad=True)
        )
        loss_cong = self.state_gru.congruence_contrastive_loss(
            text_embeddings=text_embeddings,
            emotion_one_hots=emotion_one_hots,
            congruence_labels=congruence_labels,
        )
        return loss_da, loss_cong

    def forward(self, text: str, emotion_id: int = 0):
        return self.process_turn(text, emotion_id)
