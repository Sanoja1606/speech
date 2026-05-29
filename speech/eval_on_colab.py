"""
=============================================================================
Google Colab Evaluation Script — DSACIS + ATR + LibriSpeech Evaluation
=============================================================================

Features:
- DSACIS conversational importance modeling
- ATR adaptive token routing
- Dynamic compression evaluation
- Stable readable Phi-3 generation
- Simulated speech-token routing

=============================================================================
"""

import os
import gc
import re
import logging
import torch

from datasets import load_dataset

from peft import (
    PeftModel,
    PeftConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("NoveltyEval")


# =============================================================================
# PATHS
# =============================================================================

CHECKPOINT_DIR = (
    "/content/drive/MyDrive/novelty_outputs/checkpoint-step-100"
)

CUSTOM_MODULES_PATH = os.path.join(
    CHECKPOINT_DIR,
    "custom_novelty_modules.pt",
)


# =============================================================================
# LOAD EVAL PIPELINE
# =============================================================================

def load_eval_pipeline(device):

    from train_on_colab import (

        load_quantized_model,

        build_novelty_pipeline,
    )

    logger.info(
        "Loading base quantized model..."
    )

    base_model, tokenizer = load_quantized_model()

    # =========================================================================
    # REMOVE EXISTING PEFT CONFIG
    # =========================================================================

    if hasattr(base_model, "peft_config"):

        logger.warning(
            "Model already contains PEFT config. "
            "Attempting adapter unload..."
        )

        try:

            base_model = base_model.unload()

        except Exception as e:

            logger.warning(
                f"Adapter unload failed: {e}"
            )

    # =========================================================================
    # LOAD TRAINED ADAPTER
    # =========================================================================

    logger.info(
        "Loading trained LoRA adapter..."
    )

    try:

        base_model = PeftModel.from_pretrained(

            base_model,

            CHECKPOINT_DIR,

            is_trainable=False,

            torch_dtype=torch.float16,
        )

        logger.info(
            "Adapter loaded successfully."
        )

    except Exception as e:

        logger.warning(
            f"Standard PEFT load failed: {e}"
        )

        logger.warning(
            "Attempting structural fallback..."
        )

        peft_config = PeftConfig.from_pretrained(
            CHECKPOINT_DIR
        )

        base_model = PeftModel(

            model=base_model,

            peft_config=peft_config,

            adapter_name="default",
        )

    # =========================================================================
    # BUILD PIPELINE
    # =========================================================================

    logger.info(
        "Building novelty pipeline..."
    )

    pipeline = build_novelty_pipeline(
        base_model,
        device,
    )

    # =========================================================================
    # LOAD DSACIS + ATR
    # =========================================================================

    logger.info(
        "Loading DSACIS + ATR checkpoint..."
    )

    checkpoint = torch.load(

        CUSTOM_MODULES_PATH,

        map_location=device,
    )

    if "dsacis" in checkpoint:

        pipeline.dsacis.load_state_dict(
            checkpoint["dsacis"]
        )

        logger.info(
            "DSACIS weights loaded."
        )

    if "atr" in checkpoint:

        pipeline.atr.load_state_dict(
            checkpoint["atr"]
        )

        logger.info(
            "ATR weights loaded."
        )

    pipeline.eval()

    return pipeline, tokenizer


# =============================================================================
# LOAD LIBRISPEECH
# =============================================================================

def load_librispeech_subset(num_samples=5):

    logger.info(
        "Loading LibriSpeech subset..."
    )

    ds = load_dataset(

        "openslr/librispeech_asr",

        "clean",

        split="test",

        streaming=True,
    )

    ds = list(ds.take(num_samples))

    logger.info(
        f"Loaded {len(ds)} samples."
    )

    return ds


# =============================================================================
# CLEAN GENERATED TEXT
# =============================================================================

def clean_generated_text(text):

    text = re.sub(
        r"<a_\d+>",
        "",
        text
    )

    text = text.replace(
        "<audio>",
        ""
    )

    text = text.replace(
        "</audio>",
        ""
    )

    return text.strip()


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_pipeline(

    pipeline,

    tokenizer,

    device,

    num_samples=5,
):

    logger.info(
        "Starting evaluation..."
    )

    if hasattr(
        pipeline,
        "reset_conversation"
    ):

        pipeline.reset_conversation()

    ds = load_librispeech_subset(
        num_samples=num_samples
    )

    total_compression = 0.0

    total_retained = 0

    total_original = 0

    sample_outputs = []

    with torch.no_grad():

        for idx, row in enumerate(ds):

            print(
                f"\nRunning evaluation sample {idx}"
            )

            try:

                transcript = row["text"]

                # =============================================================
                # ROUTING PROMPT (FOR DSACIS + ATR ONLY)
                # =============================================================

                fake_audio = " ".join([

                    f"<a_{torch.randint(1, 128, (1,)).item()}>"

                    for _ in range(
                        torch.randint(16, 48, (1,)).item()
                    )
                ])

                routing_prompt = (

                    f"<audio> {fake_audio} </audio>\n"

                    f"USER: {transcript}\n"

                    f"ASSISTANT:"
                )

                routing_inputs = tokenizer(

                    routing_prompt,

                    return_tensors="pt",

                    truncation=True,

                    padding=True,
                )

                input_ids = routing_inputs.input_ids.to(device)

                attention_mask = (
                    routing_inputs.attention_mask.to(device)
                )

                target_audio_ids = torch.zeros(

                    (1, 8),

                    dtype=torch.long,

                    device=device,
                )

                spk_emb = torch.randn(

                    1,

                    512,

                    device=device,
                )

                # =============================================================
                # RUN DSACIS + ATR
                # =============================================================

                out = pipeline.inference_step(

                    input_ids=input_ids,

                    attention_mask=attention_mask,

                    target_audio_ids=target_audio_ids,

                    spk_emb=spk_emb,

                    text=transcript,
                )

                # =============================================================
                # CLEAN GENERATION PROMPT
                # =============================================================

                clean_prompt = (

                    f"USER: {transcript}\n"

                    f"ASSISTANT:"
                )

                clean_inputs = tokenizer(

                    clean_prompt,

                    return_tensors="pt",

                    truncation=True,

                    padding=True,
                )

                clean_input_ids = (
                    clean_inputs.input_ids.to(device)
                )

                clean_attention_mask = (
                    clean_inputs.attention_mask.to(device)
                )

                # =============================================================
                # AUTOREGRESSIVE GENERATION
                # =============================================================

                generated_ids = pipeline.base_model.generate(

                    input_ids=clean_input_ids,

                    attention_mask=clean_attention_mask,

                    max_new_tokens=40,

                    do_sample=True,

                    temperature=0.7,

                    top_p=0.9,

                    repetition_penalty=1.1,

                    pad_token_id=tokenizer.pad_token_id,
                )

                decoded = tokenizer.decode(

                    generated_ids[0],

                    skip_special_tokens=True,
                )

                hyp_text = clean_generated_text(
                    decoded
                )

                # =============================================================
                # ROUTING STATS
                # =============================================================

                stats = out.get(
                    "routing_stats",
                    [{}]
                )

                if (
                    isinstance(stats, list)
                    and
                    len(stats) > 0
                ):

                    stats = stats[0]

                compression = stats.get(
                    "compression_ratio",
                    1.0
                )

                retained = stats.get(
                    "tokens_retained",
                    input_ids.size(1)
                )

                original = stats.get(
                    "original_tokens",
                    input_ids.size(1)
                )

                total_compression += compression

                total_retained += retained

                total_original += original

                sample_outputs.append({

                    "reference": transcript,

                    "generated": hyp_text,

                    "compression": compression,
                })

                # =============================================================
                # PRINT RESULTS
                # =============================================================

                print("=" * 80)
                print(f"SAMPLE {idx}")
                print("=" * 80)

                print(
                    f"\nTRANSCRIPT:\n{transcript}\n"
                )

                print(
                    f"GENERATED:\n{hyp_text}\n"
                )

                print(
                    f"Compression Ratio: "
                    f"{compression:.4f}"
                )

                print(
                    f"Tokens Retained: "
                    f"{retained}/{original}"
                )

                print("=" * 80)

            except Exception as e:

                logger.error(
                    f"Sample {idx} failed: {e}"
                )

                continue

    # =========================================================================
    # FINAL METRICS
    # =========================================================================

    mean_compression = (

        total_compression

        / max(len(ds), 1)
    )

    token_savings = (
        total_original - total_retained
    )

    print("\n")
    print("=" * 80)
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)

    print(
        f"Mean Compression Ratio: "
        f"{mean_compression:.4f}"
    )

    print(
        f"Total Tokens Saved: "
        f"{token_savings}"
    )

    print(
        f"Original Tokens: "
        f"{total_original}"
    )

    print(
        f"Retained Tokens: "
        f"{total_retained}"
    )

    print("=" * 80)

    return {

        "mean_compression":
            mean_compression,

        "tokens_saved":
            token_savings,

        "original_tokens":
            total_original,

        "retained_tokens":
            total_retained,

        "samples":
            sample_outputs,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )

    logger.info(
        f"Using device: {device}"
    )

    if torch.cuda.is_available():

        logger.info(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    pipeline, tokenizer = load_eval_pipeline(
        device
    )

    results = evaluate_pipeline(

        pipeline,

        tokenizer,

        device,

        num_samples=5,
    )

    logger.info(
        "Evaluation completed successfully."
    )

    print("\nDONE.\n")

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    os._exit(0)


if __name__ == "__main__":
    main()
