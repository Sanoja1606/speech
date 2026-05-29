"""
=============================================================================
Google Colab Training Script — DSACIS + ATR Novelty Pipeline
=============================================================================
"""

import os
import sys
import gc
import random
import logging

import numpy as np
import torch
import torch.nn as nn

from typing import List


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("NoveltyTrainer")


# =============================================================================
# PATHS
# =============================================================================

SPEECH_ROOT = "/content/nlp_project_colabedit/speech"

OUTPUT_DIR = "/content/drive/MyDrive/novelty_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

if SPEECH_ROOT not in sys.path:
    sys.path.insert(0, SPEECH_ROOT)

# =============================================================
# PREVENT DualSpeechLM IMPORT COLLISIONS
# =============================================================

DUALSPEECHLM_ROOT = "/content/nlp_project_colabedit/DualSpeechLM"

if DUALSPEECHLM_ROOT in sys.path:
    sys.path.remove(DUALSPEECHLM_ROOT)


# =============================================================================
# SEED
# =============================================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# LOAD QUANTIZED MODEL
# =============================================================================

def load_quantized_model(
    model_name="microsoft/Phi-3.5-mini-instruct"
):

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )

    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    logger.info(
        f"Loading {model_name} with 4-bit quantization..."
    )

    bnb_config = BitsAndBytesConfig(

        load_in_4bit=True,

        bnb_4bit_quant_type="nf4",

        bnb_4bit_compute_dtype=torch.float16,

        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    tokenizer.pad_token = tokenizer.unk_token

    # =============================================================
    # SIMULATED AUDIO TOKENS
    # =============================================================

    audio_tokens = [
        f"<a_{i}>"
        for i in range(1024)
    ]

    tokenizer.add_tokens(
        audio_tokens + [
            "<audio>",
            "</audio>",
        ]
    )

    model = AutoModelForCausalLM.from_pretrained(

        model_name,

        quantization_config=bnb_config,

        device_map="auto",

        torch_dtype=torch.float16,

        attn_implementation="eager",
    )

    model.resize_token_embeddings(
        len(tokenizer)
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    lora_config = LoraConfig(

        r=8,

        lora_alpha=16,

        target_modules=[

            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],

        lora_dropout=0.05,

        bias="none",

        task_type="CAUSAL_LM",

        modules_to_save=[
            "embed_tokens",
            "lm_head",
        ]
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    return model, tokenizer


# =============================================================================
# BUILD PIPELINE
# =============================================================================

def build_novelty_pipeline(
    base_model,
    device
):

    from src.novelty.dsacis.importance_scorer import (
        DSACISModule
    )

    from src.novelty.atr.token_router import (
        AdaptiveTokenRouter
    )

    from src.novelty.pipeline import (
        OptimizedSpeechLMPipeline
    )

    hidden_size = base_model.config.hidden_size

    logger.info("Building DSACIS...")

    dsacis = DSACISModule(

        hidden_dim=256,

        output_dim=hidden_size,

        num_gru_layers=2,

        whisper_dim=1280,
    ).to(device)

    logger.info("Building ATR...")

    atr = AdaptiveTokenRouter(

        high_threshold=0.65,

        low_threshold=0.35,

        min_keep_ratio=0.20,
    ).to(device)

    logger.info("Building pipeline...")

    pipeline = OptimizedSpeechLMPipeline(

        base_model=base_model,

        dsacis=dsacis,

        atr=atr,
    )

    pipeline.config = base_model.config

    return pipeline


# =============================================================================
# DUMMY CONVERSATIONS
# =============================================================================

DUMMY_DIALOGUES = [

    {"text": "I feel emotionally exhausted today.", "emotion": 2},

    {"text": "Can you help me calm down?", "emotion": 3},

    {"text": "I had an amazing day today.", "emotion": 1},

    {"text": "Nothing seems to be going right lately.", "emotion": 2},

    {"text": "I feel lonely sometimes.", "emotion": 2},

    {"text": "Why does nobody understand me?", "emotion": 3},

    {"text": "Things are finally improving.", "emotion": 1},

    {"text": "Everything feels overwhelming lately.", "emotion": 2},

    {"text": "Can you explain what happened?", "emotion": 0},

    {"text": "I finally achieved my goal.", "emotion": 1},
]


# =============================================================================
# GENERIC BATCH BUILDER
# =============================================================================

def build_text_batch(
    samples,
    tokenizer,
    device,
    seq_len=256,
):

    texts = []

    emotion_ids = []

    input_ids_list = []

    for sample in samples:

        text = sample["text"]

        emotion = sample["emotion"]

        fake_audio = " ".join([

            f"<a_{random.randint(1, 128)}>"

            for _ in range(
                random.randint(8, 24)
            )
        ])

        full_text = (

            f"<audio> {fake_audio} </audio> "

            f"USER: {text}"
        )

        ids = tokenizer.encode(
            full_text,
            add_special_tokens=True,
        )

        ids = ids[:seq_len]

        input_ids_list.append(
            torch.tensor(
                ids,
                dtype=torch.long
            )
        )

        texts.append(text)

        emotion_ids.append(emotion)

    max_len = max(
        len(x)
        for x in input_ids_list
    )

    padded = torch.full(
        (len(samples), max_len),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        len(samples),
        max_len,
        dtype=torch.long,
    )

    for i, ids in enumerate(input_ids_list):

        padded[i, :len(ids)] = ids

        attention_mask[i, :len(ids)] = 1

    labels = padded.clone()

    labels[attention_mask == 0] = -100

    return {

        "input_ids":
            padded.to(device),

        "attention_mask":
            attention_mask.to(device),

        "labels":
            labels.to(device),

        "target_audio_ids":
            torch.zeros(
                len(samples),
                8,
                dtype=torch.long,
                device=device,
            ),

        "spk_emb":
            torch.randn(
                len(samples),
                512,
                device=device,
            ),

        "texts":
            texts,

        "emotion_ids":
            emotion_ids,
    }


# =============================================================================
# DUMMY BATCH
# =============================================================================

def create_dummy_batch(
    tokenizer,
    device,
    batch_size=8,
):

    samples = random.sample(
        DUMMY_DIALOGUES,
        batch_size,
    )

    return build_text_batch(
        samples,
        tokenizer,
        device,
    )


# =============================================================================
# AUXILIARY DATASET LOADER
# =============================================================================

def load_auxiliary_datasets():

    logger.info(
        "Loading auxiliary datasets..."
    )

    from src.data.meld_dataset import (
        load_meld_splits
    )

    from src.data.daily_dialog_dataset import (
        load_daily_dialog
    )

    from src.data.iemocap_dataset import (
        load_iemocap_splits
    )

    meld_train, _, _ = load_meld_splits()

    dd_train, _, _ = load_daily_dialog()

    iemocap_train, _ = load_iemocap_splits(
        "/content/nlp_project_colabedit/speech/src/data/iemocap_full_dataset.csv"
    )

    logger.info(
        f"MELD loaded: {len(meld_train)}"
    )

    logger.info(
        f"DailyDialog loaded: {len(dd_train)}"
    )

    logger.info(
        f"IEMOCAP loaded: {len(iemocap_train)}"
    )

    return {

        "meld":
            meld_train,

        "dailydialog":
            dd_train,

        "iemocap":
            iemocap_train,
    }


# =============================================================================
# AUXILIARY SAMPLERS
# =============================================================================

def sample_meld_batch(
    meld_ds,
    tokenizer,
    device,
    batch_size=4,
):

    samples = random.sample(
        meld_ds.records,
        batch_size,
    )

    formatted = []

    for s in samples:

        formatted.append({

            "text":
                s["utterance"],

            "emotion":
                s["emotion_id"],
        })

    return build_text_batch(
        formatted,
        tokenizer,
        device,
    )


def sample_dailydialog_batch(
    dd_ds,
    tokenizer,
    device,
    batch_size=4,
):

    samples = random.sample(
        dd_ds.records,
        batch_size,
    )

    formatted = []

    for s in samples:

        formatted.append({

            "text":
                s["utterance"],

            "emotion":
                s["emotion_id"],
        })

    return build_text_batch(
        formatted,
        tokenizer,
        device,
    )


def sample_iemocap_batch(
    iemocap_ds,
    tokenizer,
    device,
    batch_size=4,
):

    samples = random.sample(
        iemocap_ds.records,
        batch_size,
    )

    formatted = []

    for s in samples:

        formatted.append({

            "text":
                f"Emotion sample {s['raw_emotion']}",

            "emotion":
                s["emotion_id"],
        })

    return build_text_batch(
        formatted,
        tokenizer,
        device,
    )


# =============================================================================
# TRAIN LOOP
# =============================================================================

def train_loop(
    pipeline,
    tokenizer,
    device,
    aux_datasets,
    num_steps=100,
    lr=2e-5,
):

    trainable_params = [

        p for p in pipeline.parameters()

        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=0.01,
    )

    pipeline.train()

    logger.info(
        f"Starting training for {num_steps} steps..."
    )

    for step in range(num_steps):

        optimizer.zero_grad()

        dataset_choice = random.random()

        if dataset_choice < 0.40:

            batch = create_dummy_batch(
                tokenizer,
                device,
                batch_size=8,
            )

        elif dataset_choice < 0.60:

            batch = sample_meld_batch(
                aux_datasets["meld"],
                tokenizer,
                device,
                batch_size=8,
            )

        elif dataset_choice < 0.80:

            batch = sample_dailydialog_batch(
                aux_datasets["dailydialog"],
                tokenizer,
                device,
                batch_size=8,
            )

        else:

            batch = sample_iemocap_batch(
                aux_datasets["iemocap"],
                tokenizer,
                device,
                batch_size=8,
            )

        result = pipeline(

            input_ids=batch["input_ids"],

            attention_mask=batch["attention_mask"],

            labels=batch["labels"],

            target_audio_ids=batch["target_audio_ids"],

            spk_emb=batch["spk_emb"],

            texts=batch["texts"],

            emotion_ids=batch["emotion_ids"],
        )

        loss = result["loss"]

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable_params,
            max_norm=1.0,
        )

        optimizer.step()

        if step % 10 == 0:

            breakdown = result.get(
                "loss_breakdown",
                {}
            )

            stats = result.get(
                "routing_stats",
                [{}]
            )

            cr = stats[0].get(
                "compression_ratio",
                1.0
            )

            mem = 0

            if torch.cuda.is_available():

                mem = (
                    torch.cuda.memory_allocated()
                    / 1024**3
                )

            logger.info(

                f"Step {step:4d} | "

                f"loss={loss.item():.4f} | "

                f"L_gen={breakdown.get('L_generation', 0):.4f} | "

                f"L_da={breakdown.get('L_dialogue_act', 0):.4f} | "

                f"Compression={cr:.2f} | "

                f"GPU={mem:.2f}GB"
            )

    checkpoint_dir = os.path.join(
        OUTPUT_DIR,
        f"checkpoint-step-{num_steps}"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    logger.info(
        f"Saving checkpoint to {checkpoint_dir}"
    )

    pipeline.base_model.save_pretrained(
        checkpoint_dir
    )

    torch.save(

        {

            "dsacis": pipeline.dsacis.state_dict(),

            "atr": pipeline.atr.state_dict(),

            "step": num_steps,
        },

        os.path.join(
            checkpoint_dir,
            "custom_novelty_modules.pt"
        )
    )

    logger.info(
        "Checkpoint saved successfully."
    )

    return pipeline


# =============================================================================
# MAIN
# =============================================================================

def main():

    set_seed(42)

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )

    logger.info(f"Device: {device}")

    if torch.cuda.is_available():

        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    base_model, tokenizer = load_quantized_model()

    pipeline = build_novelty_pipeline(
        base_model,
        device,
    )

    aux_datasets = load_auxiliary_datasets()

    pipeline = train_loop(

        pipeline,

        tokenizer,

        device,

        aux_datasets=aux_datasets,

        num_steps=100,
    )

    logger.info(
        "Training completed successfully."
    )


if __name__ == "__main__":
    main()


