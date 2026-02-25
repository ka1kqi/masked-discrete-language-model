"""Generate text from a trained diffusion LM checkpoint."""

import yaml
import torch
import argparse

from .model import DiffusionTransformer
from .noise import NoiseSchedule
from .diffusion import sample
from .tokenizer import get_tokenizer


def generate(checkpoint_path: str, num_samples: int = 4, config_override: dict | None = None):
    """Load a checkpoint and generate text samples.

    Args:
        checkpoint_path: path to .pt checkpoint
        num_samples: number of sequences to generate
        config_override: optional overrides for sampling config
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mcfg = cfg["model"]
    dcfg = cfg["diffusion"]
    scfg = cfg["sampling"]

    if config_override:
        scfg.update(config_override)

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Model
    model = DiffusionTransformer(
        vocab_size=mcfg["vocab_size"],
        hidden_dim=mcfg["hidden_dim"],
        num_layers=mcfg["num_layers"],
        num_heads=mcfg["num_heads"],
        max_seq_len=mcfg["max_seq_len"],
        dropout=mcfg["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Noise schedule
    schedule = NoiseSchedule(
        num_timesteps=dcfg["num_timesteps"],
        schedule=dcfg["schedule"],
    )

    tokenizer = get_tokenizer()
    mask_token_id = tokenizer.mask_token_id

    print(f"Generating {num_samples} samples ({scfg['steps']} steps, temp={scfg['temperature']})...\n")

    generated_ids = sample(
        model=model,
        schedule=schedule,
        mask_token_id=mask_token_id,
        seq_len=scfg["seq_len"],
        batch_size=num_samples,
        steps=scfg["steps"],
        temperature=scfg["temperature"],
        top_k=scfg["top_k"],
        strategy=scfg["strategy"],
        device=device,
    )

    # Decode and print
    for i in range(num_samples):
        text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
        print(f"--- Sample {i + 1} ---")
        print(text)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.steps is not None:
        overrides["steps"] = args.steps
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.top_k is not None:
        overrides["top_k"] = args.top_k

    generate(args.checkpoint, args.num_samples, overrides or None)
