"""Interactive interface for the diffusion LM."""

import sys
import torch
import torch.nn.functional as F
import argparse

from .model import DiffusionTransformer
from .noise import NoiseSchedule
from .tokenizer import get_tokenizer


class DiffusionChat:
    def __init__(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        mcfg = cfg["model"]
        dcfg = cfg["diffusion"]
        self.scfg = cfg["sampling"]

        self.tokenizer = get_tokenizer()
        self.mask_id = self.tokenizer.mask_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.max_seq_len = mcfg["max_seq_len"]

        self.model = DiffusionTransformer(
            vocab_size=mcfg["vocab_size"],
            hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"],
            num_heads=mcfg["num_heads"],
            max_seq_len=mcfg["max_seq_len"],
            dropout=mcfg["dropout"],
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.schedule = NoiseSchedule(dcfg["num_timesteps"], dcfg["schedule"])

        self.temperature = self.scfg["temperature"]
        self.steps = self.scfg["steps"]
        self.top_k = self.scfg["top_k"]

    @torch.no_grad()
    def generate(self, length: int = 0, show_progress: bool = True) -> str:
        """Generate text from scratch (all masks)."""
        seq_len = length or self.max_seq_len
        x = torch.full((1, seq_len), self.mask_id, dtype=torch.long)
        return self._denoise(x, frozen=None, show_progress=show_progress)

    @torch.no_grad()
    def complete(self, prompt: str, length: int = 0, show_progress: bool = True) -> str:
        """Complete a prompt — freeze the prompt tokens, unmask the rest."""
        seq_len = length or self.max_seq_len
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) >= seq_len:
            ids = ids[: seq_len]
            return self.tokenizer.decode(ids)

        pad_len = seq_len - len(ids)
        x = torch.tensor([ids + [self.mask_id] * pad_len], dtype=torch.long)
        frozen = torch.zeros(1, seq_len, dtype=torch.bool)
        frozen[0, : len(ids)] = True
        return self._denoise(x, frozen=frozen, show_progress=show_progress)

    @torch.no_grad()
    def infill(self, text: str, show_progress: bool = True) -> str:
        """Fill in [MASK] tokens in the given text."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]

        x = torch.tensor([ids], dtype=torch.long)
        frozen = x != self.mask_id
        return self._denoise(x, frozen=frozen, show_progress=show_progress)

    def _denoise(
        self,
        x: torch.Tensor,
        frozen: torch.Tensor | None,
        show_progress: bool,
    ) -> str:
        """Core denoising loop with optional progress display."""
        seq_len = x.shape[1]
        steps = self.steps
        tokens_per_step = seq_len / steps

        for step in range(steps):
            t_val = max(int(self.schedule.T * (1 - step / steps)), 1)
            t = torch.tensor([t_val], dtype=torch.long)

            logits = self.model(x, t) / self.temperature

            if self.top_k > 0:
                topk_vals, _ = logits.topk(self.top_k, dim=-1)
                logits[logits < topk_vals[..., -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, seq_len)

            is_masked = x == self.mask_id
            if frozen is not None:
                is_masked = is_masked & ~frozen

            if not is_masked.any():
                break

            # Confidence-based unmasking
            num_to_unmask = max(int(round(tokens_per_step * (step + 1))) - (seq_len - is_masked.sum().item()), 1)
            confidence, _ = probs.max(dim=-1)
            confidence[~is_masked] = -1.0
            k = min(num_to_unmask, is_masked.sum().item())
            _, indices = confidence[0].topk(k)
            x[0, indices] = sampled[0, indices]

            if show_progress and step % max(steps // 5, 1) == 0:
                preview = self.tokenizer.decode(x[0], skip_special_tokens=False)
                for tok in ("[PAD]", "[CLS]", "[SEP]"):
                    preview = preview.replace(tok, "")
                preview = preview.replace("[MASK]", "_").strip()
                sys.stdout.write(f"\r  [{step + 1}/{steps}] {preview[:80]}")
                sys.stdout.flush()

        # Final pass for any remaining masks
        remaining = x == self.mask_id
        if frozen is not None:
            remaining = remaining & ~frozen
        if remaining.any():
            t = torch.ones(1, dtype=torch.long)
            logits = self.model(x, t) / self.temperature
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, seq_len)
            x[remaining] = sampled[remaining]

        if show_progress:
            sys.stdout.write("\r" + " " * 90 + "\r")
            sys.stdout.flush()

        return self.tokenizer.decode(x[0], skip_special_tokens=True)


HELP = """
Commands:
  /generate [length]       Generate text from scratch
  /complete <prompt>       Complete a prompt
  /infill <text>           Fill in blanks (use [MASK] for blanks)
  /temp <value>            Set temperature (current: {temp:.1f})
  /steps <value>           Set denoising steps (current: {steps})
  /topk <value>            Set top-k (current: {topk}, 0=off)
  /help                    Show this help
  /quit                    Exit

Or just type text to complete it.
""".strip()


def main():
    parser = argparse.ArgumentParser(description="Chat with your diffusion LM")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    args = parser.parse_args()

    print("Loading model...")
    chat = DiffusionChat(args.checkpoint)
    print(f"Ready! Model loaded ({chat.max_seq_len} max tokens)")
    print(HELP.format(temp=chat.temperature, steps=chat.steps, topk=chat.top_k))
    print()

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Bye!")
            break

        elif user_input == "/help":
            print(HELP.format(temp=chat.temperature, steps=chat.steps, topk=chat.top_k))

        elif user_input.startswith("/temp "):
            try:
                chat.temperature = float(user_input.split()[1])
                print(f"  Temperature set to {chat.temperature}")
            except (ValueError, IndexError):
                print("  Usage: /temp 0.8")

        elif user_input.startswith("/steps "):
            try:
                chat.steps = int(user_input.split()[1])
                print(f"  Steps set to {chat.steps}")
            except (ValueError, IndexError):
                print("  Usage: /steps 50")

        elif user_input.startswith("/topk "):
            try:
                chat.top_k = int(user_input.split()[1])
                print(f"  Top-k set to {chat.top_k}")
            except (ValueError, IndexError):
                print("  Usage: /topk 50")

        elif user_input.startswith("/generate"):
            parts = user_input.split()
            length = int(parts[1]) if len(parts) > 1 else 0
            result = chat.generate(length=length)
            print(f"  {result}")

        elif user_input.startswith("/complete "):
            prompt = user_input[len("/complete "):]
            result = chat.complete(prompt)
            print(f"  {result}")

        elif user_input.startswith("/infill "):
            text = user_input[len("/infill "):]
            result = chat.infill(text)
            print(f"  {result}")

        else:
            # Default: treat as completion
            result = chat.complete(user_input)
            print(f"  {result}")

        print()


if __name__ == "__main__":
    main()
