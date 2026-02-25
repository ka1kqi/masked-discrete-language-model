"""Training loop for the discrete diffusion LM."""

import sys
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import load_dataset, disable_progress_bars

from .model import DiffusionTransformer
from .noise import NoiseSchedule
from .diffusion import compute_loss
from .tokenizer import get_tokenizer, tokenize_dataset


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def log(msg: str):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def train(config_path: str = "configs/default.yaml"):
    disable_progress_bars()
    cfg = load_config(config_path)
    mcfg = cfg["model"]
    dcfg = cfg["diffusion"]
    tcfg = cfg["training"]

    # Device (MPS is slower than CPU for this model, so skip it)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    log(f"Using device: {device}")

    # Tokenizer
    tokenizer = get_tokenizer()
    mask_token_id = tokenizer.mask_token_id

    # Dataset
    log(f"Loading dataset: {tcfg['dataset']}")
    raw_dataset = load_dataset(tcfg["dataset"], split="train")
    dataset = tokenize_dataset(raw_dataset, tokenizer, max_seq_len=mcfg["max_seq_len"])
    dataloader = DataLoader(
        dataset, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
        num_workers=2, pin_memory=(device == "cuda"),
    )

    # Model
    model = DiffusionTransformer(
        vocab_size=mcfg["vocab_size"],
        hidden_dim=mcfg["hidden_dim"],
        num_layers=mcfg["num_layers"],
        num_heads=mcfg["num_heads"],
        max_seq_len=mcfg["max_seq_len"],
        dropout=mcfg["dropout"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {num_params:,}")

    # Noise schedule
    schedule = NoiseSchedule(
        num_timesteps=dcfg["num_timesteps"],
        schedule=dcfg["schedule"],
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )

    # LR warmup scheduler
    def lr_lambda(step):
        if step < tcfg["warmup_steps"]:
            return step / max(tcfg["warmup_steps"], 1)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Output directory
    output_dir = Path(tcfg["output_dir"])
    output_dir.mkdir(exist_ok=True)

    # Training loop
    model.train()
    global_step = 0
    running_loss = 0.0

    log(f"Starting training for {tcfg['max_steps']} steps...")

    import time
    t0 = time.time()

    while global_step < tcfg["max_steps"]:
        for batch in dataloader:
            if global_step >= tcfg["max_steps"]:
                break

            if global_step == 0:
                log(f"First batch loaded in {time.time() - t0:.1f}s")

            x_0 = batch["input_ids"].to(device)

            loss = compute_loss(model, x_0, schedule, mask_token_id)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            global_step += 1
            log_interval = tcfg["log_every"]

            # Only sync loss from GPU at log intervals to avoid MPS stalls
            if global_step % log_interval == 0 or global_step <= 2:
                loss_val = loss.item()
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                steps_per_sec = global_step / elapsed
                log(f"step {global_step:>6d} | loss {loss_val:.4f} | lr {lr:.2e} | {steps_per_sec:.1f} steps/s")

            if global_step % tcfg["checkpoint_every"] == 0:
                ckpt_path = output_dir / f"checkpoint_{global_step}.pt"
                torch.save(
                    {
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": cfg,
                    },
                    ckpt_path,
                )
                log(f"Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = output_dir / "checkpoint_final.pt"
    torch.save(
        {
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
        },
        final_path,
    )
    log(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
