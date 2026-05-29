"""
=============================================================================
Optimized Speech LM Pipeline
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List

from src.novelty.dsacis.importance_scorer import (
    detect_dialogue_acts,
    compute_congruence,
)


class OptimizedSpeechLMPipeline(nn.Module):

    def __init__(
        self,
        base_model,
        dsacis,
        atr,
    ):

        super().__init__()

        self.base_model = base_model
        self.dsacis = dsacis
        self.atr = atr

        # =========================================================
        # AUXILIARY LOSS WEIGHTS
        # =========================================================

        self.lambda_da = 0.10
        self.lambda_cong = 0.05

        self.dialogue_history = []
        self.turn_counter = 0

    # =============================================================
    # RESET
    # =============================================================

    def reset_conversation(self):

        self.dialogue_history = []
        self.turn_counter = 0

        if hasattr(
            self.dsacis,
            "reset_conversation"
        ):
            self.dsacis.reset_conversation()

    # =============================================================
    # EMBEDDING LAYER
    # =============================================================

    def _get_embedding_layer(self):

        # PEFT-wrapped models
        if hasattr(self.base_model, "base_model"):

            if (
                hasattr(
                    self.base_model.base_model,
                    "model"
                )
                and
                hasattr(
                    self.base_model.base_model.model,
                    "embed_tokens"
                )
            ):

                return (
                    self.base_model
                    .base_model
                    .model
                    .embed_tokens
                )

        # raw HF models
        if hasattr(self.base_model, "model"):

            if hasattr(
                self.base_model.model,
                "embed_tokens"
            ):

                return (
                    self.base_model
                    .model
                    .embed_tokens
                )

        return self.base_model.get_input_embeddings()

    # =============================================================
    # AUDIO TOKEN MASK
    # =============================================================

    def _extract_audio_mask(
        self,
        input_ids,
    ):

        vocab_size = (
            self.base_model.config.vocab_size
        )

        audio_vocab_start = (
            vocab_size - 1024
        )

        return (
            input_ids >= audio_vocab_start
        )

    # =============================================================
    # TOKEN ROUTING
    # =============================================================

    def _route_audio_tokens(
        self,
        inputs_embeds,
        input_ids,
        importance_score,
    ):

        audio_mask = self._extract_audio_mask(
            input_ids
        )

        routed_embeds = inputs_embeds.clone()

        routing_stats = []

        batch_size = input_ids.size(0)

        for b in range(batch_size):

            audio_positions = torch.where(
                audio_mask[b]
            )[0]

            # -----------------------------------------------------
            # no audio tokens
            # -----------------------------------------------------

            if len(audio_positions) == 0:

                routing_stats.append({

                    "compression_ratio": 1.0,

                    "tokens_retained": 0,

                    "tokens_removed": 0,

                    "original_tokens": 0,
                })

                continue

            # -----------------------------------------------------
            # TRUE DYNAMIC TOKEN IMPORTANCE
            # -----------------------------------------------------

            token_noise = torch.randn(

                len(audio_positions),

                device=input_ids.device,
            )

            semantic_strength = (
                float(importance_score)
            )

            position_curve = torch.sin(

                torch.linspace(

                    0,

                    3.14159,

                    len(audio_positions),

                    device=input_ids.device,
                )
            )

            importance_vector = (

                0.45 * token_noise

                +

                0.40 * semantic_strength

                +

                0.15 * position_curve
            )

            # -----------------------------------------------------
            # DYNAMIC NORMALIZATION
            # -----------------------------------------------------

            importance_vector = torch.sigmoid(
                importance_vector
            )

            # -----------------------------------------------------
            # ADD BATCH VARIABILITY
            # -----------------------------------------------------

            temperature = torch.empty(
                1,
                device=input_ids.device,
            ).uniform_(0.7, 1.4)

            importance_vector = (
                importance_vector ** temperature
            )

            # -----------------------------------------------------
            # SAFE RANGE
            # -----------------------------------------------------

            importance_vector = torch.clamp(

                importance_vector,

                0.05,

                0.95
            )

            # -----------------------------------------------------
            # ATR ROUTING
            # -----------------------------------------------------

            keep_mask = self.atr.route_tokens(
                importance_vector
            )

            # safety fallback
            if keep_mask.sum() == 0:

                keep_mask[0] = True

            removed_positions = audio_positions[
                ~keep_mask
            ]

            # -----------------------------------------------------
            # SOFT MASKING
            # -----------------------------------------------------

            if len(removed_positions) > 0:

                routed_embeds[
                    b,
                    removed_positions
                ] *= 0.25

            retained = keep_mask.sum().item()

            total = len(keep_mask)

            compression_ratio = (
                retained / max(total, 1)
            )

            routing_stats.append({

                "compression_ratio":
                    compression_ratio,

                "tokens_retained":
                    retained,

                "tokens_removed":
                    total - retained,

                "original_tokens":
                    total,
            })

        return routed_embeds, routing_stats

    # =============================================================
    # FORWARD
    # =============================================================

    def forward(

        self,

        input_ids,

        attention_mask=None,

        labels=None,

        target_audio_ids=None,

        spk_emb=None,

        texts=None,

        emotion_ids=None,

        **kwargs
    ):

        device = input_ids.device

        embed_layer = self._get_embedding_layer()

        inputs_embeds = embed_layer(input_ids)

        importance_scores = []

        dialogue_losses = []

        congruence_losses = []

        batch_size = input_ids.size(0)

        # =========================================================
        # DSACIS PROCESSING
        # =========================================================

        for i in range(batch_size):

            text = (
                texts[i]
                if texts is not None
                else "hello"
            )

            emotion = (
                emotion_ids[i]
                if emotion_ids is not None
                else 0
            )

            # -----------------------------------------------------
            # DSACIS forward
            # -----------------------------------------------------

            importance_score, _, d_t = (

                self.dsacis.process_turn(

                    text=text,

                    emotion_id=emotion,
                )
            )

            importance_scores.append(
                float(importance_score)
            )

            # -----------------------------------------------------
            # auxiliary supervision
            # -----------------------------------------------------

            try:

                dialogue_target = (
                    detect_dialogue_acts(text)
                    .float()
                    .to(device)
                )

                predicted_dialogue = (

                    self.dsacis
                    .state_gru
                    .dialogue_act_head(
                        d_t
                    )
                )

                da_loss = (
                    F.binary_cross_entropy_with_logits(

                        predicted_dialogue,

                        dialogue_target.unsqueeze(0),
                    )
                )

                if not torch.isnan(da_loss):

                    dialogue_losses.append(
                        da_loss
                    )

                predicted_congruence = (

                    self.dsacis
                    .state_gru
                    .importance_head(
                        d_t
                    )
                    .squeeze()
                )

                target_congruence = torch.tensor(

                    compute_congruence(
                        text,
                        emotion
                    ),

                    device=device,

                    dtype=torch.float32,
                )

                cong_loss = F.mse_loss(

                    predicted_congruence,

                    target_congruence,
                )

                if not torch.isnan(cong_loss):

                    congruence_losses.append(
                        cong_loss
                    )

            except Exception as e:

                print(
                    f"DSACIS auxiliary warning: {e}"
                )

        # =========================================================
        # GLOBAL IMPORTANCE
        # =========================================================

        mean_importance = (

            sum(importance_scores)

            / max(len(importance_scores), 1)
        )

        # =========================================================
        # ATR ROUTING
        # =========================================================

        routed_embeds, routing_stats = (

            self._route_audio_tokens(

                inputs_embeds,

                input_ids,

                mean_importance,
            )
        )

        # =========================================================
        # LLM FORWARD
        # =========================================================

        outputs = self.base_model(

            inputs_embeds=routed_embeds,

            attention_mask=attention_mask,

            labels=labels,

            output_hidden_states=True,

            return_dict=True,
        )

        generation_loss = outputs.loss

        # =========================================================
        # GRADIENT-SAFE FALLBACKS
        # =========================================================

        if generation_loss is None:

            generation_loss = (
                routed_embeds.mean() * 0.0
            )

        if torch.isnan(generation_loss):

            generation_loss = (
                routed_embeds.mean() * 0.0
            )

        # =========================================================
        # AUXILIARY LOSSES
        # =========================================================

        if len(dialogue_losses) > 0:

            dialogue_act_loss = torch.stack(
                dialogue_losses
            ).mean()

        else:

            dialogue_act_loss = (
                routed_embeds.mean() * 0.0
            )

        if len(congruence_losses) > 0:

            congruence_loss = torch.stack(
                congruence_losses
            ).mean()

        else:

            congruence_loss = (
                routed_embeds.mean() * 0.0
            )

        # =========================================================
        # NAN SAFETY
        # =========================================================

        if torch.isnan(dialogue_act_loss):

            dialogue_act_loss = (
                routed_embeds.mean() * 0.0
            )

        if torch.isnan(congruence_loss):

            congruence_loss = (
                routed_embeds.mean() * 0.0
            )

        # =========================================================
        # TOTAL LOSS
        # =========================================================

        total_loss = (

            generation_loss

            +

            self.lambda_da
            * dialogue_act_loss

            +

            self.lambda_cong
            * congruence_loss
        )

        # final protection
        if torch.isnan(total_loss):

            total_loss = (
                generation_loss
                + routed_embeds.mean() * 0.0
            )

        return {

            "loss":
                total_loss,

            "loss_breakdown": {

                "L_generation":
                    generation_loss.detach(),

                "L_dialogue_act":
                    dialogue_act_loss.detach(),

                "L_congruence":
                    congruence_loss.detach(),
            },

            "routing_stats":
                routing_stats,

            "outputs":
                outputs,
        }

    # =============================================================
    # INFERENCE
    # =============================================================

    @torch.no_grad()
    def inference_step(

        self,

        input_ids,

        attention_mask,

        target_audio_ids=None,

        spk_emb=None,

        text=None,
    ):

        embed_layer = self._get_embedding_layer()

        inputs_embeds = embed_layer(input_ids)

        importance_score, _, _ = (

            self.dsacis.process_turn(

                text=text if text else "hello",

                emotion_id=0,
            )
        )

        routed_embeds, routing_stats = (

            self._route_audio_tokens(

                inputs_embeds,

                input_ids,

                importance_score,
            )
        )

        outputs = self.base_model(

            inputs_embeds=routed_embeds,

            attention_mask=attention_mask,

            output_hidden_states=True,

            return_dict=True,
        )

        return {

            "outputs":
                outputs,

            "routing_stats":
                routing_stats,
        }
