# training/train.py

# ----------------------------
# Suppress specific warnings
# ----------------------------
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*np.object.*"
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*sentencepiece tokenizer.*byte fallback.*"
)

import os
import torch
import argparse
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

from training.dailymail import DailyMailDataset
from training.collator import SummarizationCollator
from evaluation.rouge_eval import RougeEvaluator

def train(
    model_wrapper,
    tokenizer,
    dataset_dir,
    output_dir,
    task_prefix="summarize: ",
    epochs=3,
    batch_size=8,
    lr=5e-5,
    max_input_length=1024,
    max_output_length=256,
    gradient_accumulation_steps=1,
    fp16=False,
    bf16=False,
    save_every_epoch=True
):
    """
    Train an abstractive summarization model with PEFT/LoRA and mixed precision.
    """

    # ----------------------------
    # Load datasets
    # ----------------------------
    train_dataset = DailyMailDataset(dataset_dir, split="train", max_samples = 150000)
    val_dataset = DailyMailDataset(dataset_dir, split="validation", max_samples=150)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    collator = SummarizationCollator(
        tokenizer,
        max_input_len=max_input_length,
        max_target_len=max_output_length,
        task_prefix=task_prefix
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collator, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collator, pin_memory=True
    )

    # ----------------------------
    # Prepare model
    # ----------------------------
    device = model_wrapper.device
    model = model_wrapper.model
    model_wrapper.move_to_device()
    print(f"Training on device: {device}")

    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q", "k", "v", "o"]  # mT5 recommended
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ----------------------------
    # Optimizer & scheduler
    # ----------------------------
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = (len(train_loader) * epochs) // gradient_accumulation_steps

    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # ----------------------------
    # Mixed precision setup
    # ----------------------------
    if fp16 and "mt5" in str(model_wrapper.model_name).lower():
        print("mT5 detected: disabling FP16 and enabling BF16 for stability if supported.")
        fp16 = False
        bf16 = True

    if bf16 and not torch.cuda.is_bf16_supported():
        print("BF16 not supported on this device, falling back to FP32.")
        bf16 = False

    enabled = fp16 or bf16
    dtype = torch.float16 if fp16 else torch.bfloat16 if bf16 else None
    scaler = GradScaler(enabled=fp16)

    # Enable CUDNN benchmark for faster kernels
    torch.backends.cudnn.benchmark = True

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=100)
        total_loss = 0.0
        num_steps = len(train_loader)

        for step, batch in enumerate(pbar, 1):
            # Move batch to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with autocast(enabled=enabled, dtype=dtype if dtype is not None else torch.float32):
                labels = batch["labels"].clone()
                labels[labels == tokenizer.pad_token_id] = -100

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=labels
                )


                loss = outputs.loss
                scaled_loss = loss / gradient_accumulation_steps

            # Backward
            if fp16:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            total_loss += loss.item() if not torch.isnan(loss) else 0.0

            # Gradient accumulation step
            if (step % gradient_accumulation_steps == 0) or (step == num_steps):
                if fp16:
                    scaler.unscale_(optimizer)

                clip_grad_norm_(model.parameters(), max_norm=1.0)

                if fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix({"loss": f"{total_loss/step:.4f}"})

        # ----------------------------
        # Validation / ROUGE
        # ----------------------------
        model.eval()
        preds_all, refs_all = [], []

        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                generated_ids = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=max_output_length,
                    num_beams=4,
                    length_penalty=1.0,
                    early_stopping=True,
                )

                labels = batch["labels"].clone()
                labels = labels.masked_fill(labels == -100, tokenizer.pad_token_id)

                preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                preds = [p.strip() for p in preds]
                refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

                if idx % 50 == 0:
                    print(f"Sample Prediction {idx}: Pred: {preds[0]} | Ref: {refs[0]}")

                preds_all.extend(preds)
                refs_all.extend(refs)

        evaluator = RougeEvaluator()
        scores = evaluator.evaluate(preds_all, refs_all)

        print(
            f"ROUGE-1: {scores['rouge1']:.4f} | "
            f"ROUGE-2: {scores['rouge2']:.4f} | "
            f"ROUGE-L: {scores['rougeL']:.4f}"
        )

        # ----------------------------
        # Save checkpoint
        # ----------------------------
        if save_every_epoch:
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
            print(f"Saved model checkpoint to {epoch_dir}")

    print("Training finished.")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an abstractive summarization model.")
    parser.add_argument("--config", type=str, required=True, help="Path to training config JSON file.")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        import json
        config = json.load(f)

    model_name = config.get("model_name")
    assert model_name is not None, "Model name must be specified in config."

    from models.model_factory import get_model
    wrapper, tokenizer = get_model(model_name, pretrained_lora_weights=config.get("pretrained_lora_weights", None))
    print(f"Loaded model and tokenizer for {model_name}.")

    train(
        model_wrapper=wrapper,
        tokenizer=tokenizer,
        dataset_dir=config["dataset_dir"],
        output_dir=config.get("checkpoints_dir"),
        task_prefix=config.get("task_prefix", "summarize: "),
        epochs=config.get("num_epochs", 3),
        batch_size=config.get("batch_size", 8),
        lr=config.get("learning_rate", 5e-5),
        max_input_length=config.get("max_tokens", 1024),
        max_output_length=256,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        fp16=config.get("fp16", False),
        bf16=config.get("bf16", False),
        save_every_epoch=config.get("save_checkpoints", True)
    )