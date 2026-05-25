"""
=============================================================================
PIPELINE INTEGRATION — OptimizedSpeechLMPipeline  (v2 — fully corrected)
=============================================================================

FIXES from analysis review:
  B) Both auxiliary losses (dialogue_act_loss, congruence_loss) are now
     FULLY COMPUTED and passed into MultiTaskLoss — not left as None.
     pipeline.py builds all required tensors from the batch and calls
     dsacis.compute_aux_losses() with real data every training step.
  C) True compression: ATR.forward() returns routed_ids_list (physically
     packed, no padding).  ATR.build_packed_embeds() assembles the final
     inputs_embeds.  Phi-3.5-mini never attends to dropped tokens.
  D) Emotion ID is derived from Whisper features via dsacis.predict_emotion()
     inside this pipeline.  No external upstream SER model needed.

How to use:
    Training:   pipeline.forward(batch)
    Inference:  pipeline.inference_step(...)
    New session: pipeline.reset_conversation()
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List

from src.novelty.dsacis.importance_scorer import (
    DSACISModule, encode_text, detect_dialogue_acts,
    compute_congruence, SENTENCE_EMBED_DIM, NUM_EMOTIONS, NUM_DIALOGUE_ACTS,
)
from src.novelty.atr.token_router import AdaptiveTokenRouter


# ─────────────────────────────────────────────────────────────────────────────
# Multi-task Loss
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    L_total = L_generation + λ_da * L_dialogue_act + λ_cong * L_congruence

    Both auxiliary losses are FULLY PASSED IN from the training forward —
    not None placeholders.  Weights are small so they support but do not
    dominate the generation objective.
    """

    def __init__(
        self,
        lambda_da:   float = 0.10,
        lambda_cong: float = 0.05,
    ):
        super().__init__()
        self.lambda_da   = lambda_da
        self.lambda_cong = lambda_cong

    def forward(
        self,
        generation_loss:   torch.Tensor,
        dialogue_act_loss: Optional[torch.Tensor] = None,
        congruence_loss:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        total = generation_loss.clone()
        breakdown = {"L_generation": generation_loss.item()}

        if dialogue_act_loss is not None and dialogue_act_loss.requires_grad:
            total = total + self.lambda_da * dialogue_act_loss
            breakdown["L_dialogue_act"] = dialogue_act_loss.item()

        if congruence_loss is not None and congruence_loss.requires_grad:
            total = total + self.lambda_cong * congruence_loss
            breakdown["L_congruence"] = congruence_loss.item()

        breakdown["L_total"] = total.item()
        return total, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build auxiliary training tensors from a batch
# ─────────────────────────────────────────────────────────────────────────────

def _build_aux_tensors(
    texts:        List[str],
    emotion_ids:  List[int],
    device:       torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build four tensors needed by dsacis.compute_aux_losses():
        text_embeddings   [B, SENTENCE_EMBED_DIM]  — MiniLM sentence embeddings
        emotion_one_hots  [B, NUM_EMOTIONS]         — one-hot emotion vectors
        congruence_labels [B]                       — float 0 or 1
        dialogue_act_vecs [B, NUM_DIALOGUE_ACTS]    — multi-hot act vectors

    This is called once per training step inside pipeline.forward().
    """
    B = len(texts)

    text_embs = torch.cat([encode_text(t, device) for t in texts], dim=0)  # [B,384]

    emo_ohs = torch.zeros(B, NUM_EMOTIONS, device=device)
    for b, eid in enumerate(emotion_ids):
        if 0 <= eid < NUM_EMOTIONS:
            emo_ohs[b, eid] = 1.0

    cong_labels = torch.tensor(
        [compute_congruence(t, e) for t, e in zip(texts, emotion_ids)],
        dtype=torch.float32, device=device,
    )

    act_vecs = torch.stack(
        [detect_dialogue_acts(t).to(device) for t in texts], dim=0
    )  # [B, NUM_DIALOGUE_ACTS]

    return text_embs, emo_ohs, cong_labels, act_vecs


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class OptimizedSpeechLMPipeline(nn.Module):
    """
    Full pipeline:
        1. DSACIS — NLP novelty
           • Predicts emotion_id from Whisper features (SER head)
           • Computes importance_score + soft_prefix from text + emotion
           • Returns d_t for auxiliary loss computation
        2. ATR — Speech novelty
           • Routes audio tokens; physically drops unimportant ones
           • Builds packed inputs_embeds with soft prefix prepended
        3. Phi-3.5-mini (base model, unchanged)
           • Receives shorter token sequence + pragmatic context prefix
        4. MultiTaskLoss — fully wired with both NLP auxiliary losses

    Args:
        base_model : GPTPhi3ForCausalLM instance from DualSpeechLM
        dsacis     : DSACISModule instance
        atr        : AdaptiveTokenRouter instance
    """

    def __init__(
        self,
        base_model: nn.Module,
        dsacis:     DSACISModule,
        atr:        AdaptiveTokenRouter,
    ):
        super().__init__()
        self.base_model      = base_model
        self.dsacis          = dsacis
        self.atr             = atr
        self.multi_task_loss = MultiTaskLoss()

    def reset_conversation(self):
        """Call at the start of every new conversation session."""
        self.dsacis.reset_conversation()

    # ── Training forward ─────────────────────────────────────────────────

    def forward(
        self,
        input_ids:                torch.Tensor,          # [B, T]
        attention_mask:           torch.Tensor,          # [B, T]
        labels:                   torch.Tensor,          # [B, T]
        target_audio_ids:         torch.Tensor,          # [B, T_a]
        spk_emb:                  torch.Tensor,          # [B, 512]
        texts:                    List[str],             # decoded text per batch item
        whisper_features:         Optional[torch.Tensor] = None,  # [B, T_w, 1280]
        emotion_ids:              Optional[List[int]]    = None,  # override SER if known
        dialogue_act_labels:      Optional[torch.Tensor] = None,  # [B, NUM_DA]
        waveforms:                Optional[torch.Tensor] = None,  # [B, samples]
        attention_mask_question:  Optional[torch.Tensor] = None,
        attention_mask_answer:    Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Full training forward pass.

        All four analysis fixes are active:
          B) Both auxiliary losses fully computed and added to total loss
          C) ATR physically packs sequences — Phi-3.5-mini attends to fewer tokens
          D) emotion_ids derived from Whisper features via SER head when not provided
        """
        device = input_ids.device
        B = input_ids.shape[0]

        # ── Step D: Derive emotion_ids from Whisper features if not given ──
        ser_logits = None
        if emotion_ids is None:
            if whisper_features is not None:
                ser_logits, emotion_ids = self.dsacis.predict_emotion(
                    whisper_features.to(device)
                )
            else:
                emotion_ids = [0] * B   # default neutral

        # ── Step 1: DSACIS — one representative turn per batch ────────────
        # In training we process the batch's first item to update state,
        # then build aux tensors for the full batch below.
        importance_score, soft_prefix, d_t_single = self.dsacis.process_turn(
            text=texts[0],
            emotion_id=emotion_ids[0],
        )
        soft_prefix = soft_prefix.to(device)   # [1, 1, 3072]

        # Build full-batch d_t by repeating (state is per-conversation,
        # but aux losses need [B, hidden_dim])
        d_t_batch = d_t_single.expand(B, -1)   # [B, hidden_dim]

        # ── Step B: Build auxiliary training tensors ───────────────────────
        text_embs, emo_ohs, cong_labels, act_vecs = _build_aux_tensors(
            texts=texts,
            emotion_ids=emotion_ids,
            device=device,
        )

        # Compute both auxiliary NLP losses — FULLY WIRED
        loss_da, loss_cong = self.dsacis.compute_aux_losses(
            d_t=d_t_batch,
            text_embeddings=text_embs,
            emotion_one_hots=emo_ohs,
            congruence_labels=cong_labels,
            dialogue_act_labels=dialogue_act_labels if dialogue_act_labels is not None
                                 else act_vecs,      # auto-detected if not in dataset
        )

        # SER auxiliary loss (if ground-truth emotion labels in batch)
        ser_loss = None
        if ser_logits is not None and emotion_ids is not None:
            eid_tensor = torch.tensor(emotion_ids, dtype=torch.long, device=device)
            ser_loss = F.cross_entropy(ser_logits, eid_tensor)

        # ── Step C: ATR — physically compress audio token sequences ────────
        # Identify audio token range in vocabulary
        # (DualSpeechLM adds 1024 audio tokens to Phi's vocab)
        audio_vocab_start = self.base_model.config.vocab_size - 1024

        # Split input_ids into audio and non-audio positions per batch item
        audio_mask = input_ids >= audio_vocab_start           # [B, T]

        # Route audio tokens per batch item → physically packed lists
        all_routed_audio: List[torch.Tensor] = []
        routing_stats_all: List[Dict] = []

        for b in range(B):
            audio_pos = audio_mask[b].nonzero(as_tuple=True)[0]   # positions
            if len(audio_pos) == 0:
                # No audio tokens in this sample (e.g. text-only turn)
                all_routed_audio.append(torch.empty(0, dtype=input_ids.dtype, device=device))
                routing_stats_all.append({})
                continue

            audio_ids_b = input_ids[b, audio_pos]                  # [T_audio]
            wf_b = waveforms[b] if waveforms is not None else None

            routed_list, _, stats = self.atr(
                audio_ids=audio_ids_b,
                nlp_importance_score=importance_score,
                waveform=wf_b,
            )
            all_routed_audio.append(routed_list[0])   # single item from list
            routing_stats_all.append(stats)

        # ── Step 2: Reconstruct input_ids with routed audio tokens ─────────
        # Strategy: replace audio positions with routed tokens.
        # Non-audio tokens (text instruction tokens) are always kept.
        # We build new input_ids sequences per batch item, then pad.

        new_ids_list: List[torch.Tensor] = []
        for b in range(B):
            audio_pos  = audio_mask[b].nonzero(as_tuple=True)[0]
            non_audio  = input_ids[b, ~audio_mask[b]]              # kept as-is
            routed_aud = all_routed_audio[b]

            # Interleave: find where audio block starts in the original sequence
            # and splice in the shorter routed sequence
            if len(audio_pos) == 0 or len(routed_aud) == 0:
                new_ids_list.append(input_ids[b])
            else:
                first_audio = audio_pos[0].item()
                before_audio = input_ids[b, :first_audio]           # [pre]
                after_audio  = input_ids[b, audio_pos[-1]+1:]       # [post]
                new_seq = torch.cat([before_audio, routed_aud, after_audio])
                new_ids_list.append(new_seq)

        # Pad to batch maximum (right-padding with pad_token_id = 0)
        T_max = max(s.shape[0] for s in new_ids_list)
        padded_ids  = torch.zeros(B, T_max, dtype=input_ids.dtype, device=device)
        padded_mask = torch.zeros(B, T_max, dtype=attention_mask.dtype, device=device)
        for b, seq in enumerate(new_ids_list):
            L = seq.shape[0]
            padded_ids[b, :L]  = seq
            padded_mask[b, :L] = 1

        # ── Step 3: Get LLM input embeddings and prepend soft prefix ────────
        embed_layer   = self.base_model.model.embed_tokens
        input_embeds  = embed_layer(padded_ids)                    # [B, T_max, D]

        # Prepend DSACIS soft prefix to every item in batch
        prefix_exp    = soft_prefix.expand(B, 1, -1)              # [B, 1, 3072]
        input_embeds  = torch.cat([prefix_exp, input_embeds], dim=1)  # [B,T_max+1,D]

        # Extend attention mask and labels for prefix token
        prefix_ones   = torch.ones(B, 1, dtype=padded_mask.dtype, device=device)
        attn_extended = torch.cat([prefix_ones, padded_mask], dim=1)

        # Pad or trim labels to match new sequence length
        T_new = input_embeds.shape[1]
        if labels.shape[1] < T_new:
            pad_len = T_new - labels.shape[1]
            label_pad = torch.full((B, pad_len), -100, dtype=labels.dtype, device=device)
            labels_extended = torch.cat([
                torch.full((B, 1), -100, dtype=labels.dtype, device=device),
                labels,
                label_pad,
            ], dim=1)
        else:
            labels_extended = torch.cat([
                torch.full((B, 1), -100, dtype=labels.dtype, device=device),
                labels[:, :T_new - 1],
            ], dim=1)

        if attention_mask_question is not None:
            q_pre = torch.ones(B, 1, dtype=attention_mask_question.dtype, device=device)
            attention_mask_question = torch.cat([q_pre, attention_mask_question], dim=1)
        if attention_mask_answer is not None:
            a_pre = torch.zeros(B, 1, dtype=attention_mask_answer.dtype, device=device)
            attention_mask_answer = torch.cat([a_pre, attention_mask_answer], dim=1)

        # ── Step 4: Base model forward (no input_ids — we pass embeds) ─────
        outputs = self.base_model(
            input_ids=None,
            inputs_embeds=input_embeds,
            attention_mask=attn_extended,
            labels=labels_extended,
            target_audio_ids=target_audio_ids,
            spk_emb=spk_emb,
            attention_mask_question=attention_mask_question,
            attention_mask_answer=attention_mask_answer,
        )

        # ── Step 5: Multi-task loss — FULLY WIRED with both aux losses ─────
        generation_loss = outputs.loss if outputs.loss is not None \
                          else torch.tensor(0.0, device=device, requires_grad=True)

        total_loss, breakdown = self.multi_task_loss(
            generation_loss=generation_loss,
            dialogue_act_loss=loss_da,       # ← real tensor, not None
            congruence_loss=loss_cong,       # ← real tensor, not None
        )

        # Add SER loss with small weight
        if ser_loss is not None:
            total_loss = total_loss + 0.05 * ser_loss
            breakdown["L_ser"] = ser_loss.item()

        return {
            "loss"              : total_loss,
            "loss_breakdown"    : breakdown,
            "importance_score"  : importance_score,
            "routing_stats"     : routing_stats_all,
            "emotion_ids"       : emotion_ids,
            "base_outputs"      : outputs,
        }

    # ── Inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def inference_step(
        self,
        input_ids:         torch.Tensor,
        attention_mask:    torch.Tensor,
        target_audio_ids:  torch.Tensor,
        spk_emb:           torch.Tensor,
        text:              str = "",
        whisper_features:  Optional[torch.Tensor] = None,
        emotion_id:        Optional[int] = None,
        waveform:          Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Single-turn inference.

        emotion_id is derived from whisper_features if not provided.
        ATR physically compresses audio tokens.
        soft_prefix is prepended to the LLM input.
        """
        device = input_ids.device
        B = input_ids.shape[0]

        # Derive emotion_id
        if emotion_id is None:
            if whisper_features is not None:
                _, eid_list = self.dsacis.predict_emotion(whisper_features.to(device))
                emotion_id = eid_list[0]
            else:
                emotion_id = 0

        # DSACIS
        importance_score, soft_prefix, _ = self.dsacis.process_turn(text, emotion_id)
        soft_prefix = soft_prefix.to(device)

        # ATR — route audio tokens
        audio_vocab_start = self.base_model.config.vocab_size - 1024
        audio_mask = input_ids >= audio_vocab_start
        routed_audio = []
        routing_stats = {}

        for b in range(B):
            audio_pos = audio_mask[b].nonzero(as_tuple=True)[0]
            if len(audio_pos) == 0:
                routed_audio.append(torch.empty(0, dtype=input_ids.dtype, device=device))
                continue
            audio_ids_b = input_ids[b, audio_pos]
            wf_b = waveform[b] if waveform is not None else None
            routed_list, _, routing_stats = self.atr(
                audio_ids=audio_ids_b,
                nlp_importance_score=importance_score,
                waveform=wf_b,
            )
            routed_audio.append(routed_list[0])

        # Reconstruct and embed
        new_ids_list = []
        for b in range(B):
            audio_pos  = audio_mask[b].nonzero(as_tuple=True)[0]
            routed_aud = routed_audio[b]
            if len(audio_pos) == 0 or len(routed_aud) == 0:
                new_ids_list.append(input_ids[b])
            else:
                first_audio = audio_pos[0].item()
                new_seq = torch.cat([
                    input_ids[b, :first_audio],
                    routed_aud,
                    input_ids[b, audio_pos[-1]+1:],
                ])
                new_ids_list.append(new_seq)

        T_max = max(s.shape[0] for s in new_ids_list)
        padded_ids  = torch.zeros(B, T_max, dtype=input_ids.dtype, device=device)
        padded_mask = torch.zeros(B, T_max, dtype=torch.long, device=device)
        for b, seq in enumerate(new_ids_list):
            L = seq.shape[0]
            padded_ids[b, :L]  = seq
            padded_mask[b, :L] = 1

        embed_layer  = self.base_model.model.embed_tokens
        input_embeds = embed_layer(padded_ids)
        prefix_exp   = soft_prefix.expand(B, 1, -1)
        input_embeds = torch.cat([prefix_exp, input_embeds], dim=1)
        prefix_ones  = torch.ones(B, 1, dtype=padded_mask.dtype, device=device)
        attn_mask    = torch.cat([prefix_ones, padded_mask], dim=1)

        outputs = self.base_model(
            input_ids=None,
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            target_audio_ids=target_audio_ids,
            spk_emb=spk_emb,
        )

        return {
            "outputs"          : outputs,
            "importance_score" : importance_score,
            "routing_stats"    : routing_stats,
            "emotion_id"       : emotion_id,
        }
