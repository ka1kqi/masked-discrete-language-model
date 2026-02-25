"""Forward and reverse diffusion process for discrete masked diffusion."""

import torch
import torch.nn.functional as F

from .noise import NoiseSchedule, corrupt
from .model import DiffusionTransformer


def compute_loss(
    model: DiffusionTransformer,
    x_0: torch.Tensor,
    schedule: NoiseSchedule,
    mask_token_id: int,
) -> torch.Tensor:
    """Compute masked diffusion training loss.

    Samples a random timestep, corrupts the input, and computes
    cross-entropy loss only on the masked positions.

    Args:
        model: the diffusion transformer
        x_0: (batch, seq_len) clean token IDs
        schedule: noise schedule
        mask_token_id: ID of the [MASK] token

    Returns:
        scalar loss
    """
    batch_size = x_0.shape[0]
    device = x_0.device

    # Sample random timesteps in [1, T]
    t = torch.randint(1, schedule.T + 1, (batch_size,), device=device)

    # Corrupt
    x_t, mask = corrupt(x_0, t, schedule, mask_token_id)

    # Predict
    logits = model(x_t, t)  # (batch, seq_len, vocab_size)

    # Cross-entropy only on masked positions
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = x_0.view(-1)
    mask_flat = mask.view(-1)

    loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    # Average over masked tokens only
    loss = (loss_per_token * mask_flat.float()).sum() / mask_flat.float().sum().clamp(min=1)

    return loss


@torch.no_grad()
def sample(
    model: DiffusionTransformer,
    schedule: NoiseSchedule,
    mask_token_id: int,
    seq_len: int = 128,
    batch_size: int = 1,
    steps: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    strategy: str = "confidence",
    device: str = "cpu",
) -> torch.Tensor:
    """Generate text via iterative unmasking.

    Starts from a fully masked sequence and progressively unmasks tokens
    over the given number of steps.

    Args:
        model: trained diffusion transformer
        schedule: noise schedule
        mask_token_id: ID of the [MASK] token
        seq_len: length of sequence to generate
        batch_size: number of sequences to generate
        steps: number of denoising steps
        temperature: sampling temperature
        top_k: top-k filtering (0 to disable)
        strategy: unmasking strategy — "confidence", "linear", or "random"
        device: device to run on

    Returns:
        (batch_size, seq_len) generated token IDs
    """
    model.eval()

    # Start fully masked
    x = torch.full((batch_size, seq_len), mask_token_id, dtype=torch.long, device=device)

    # Number of tokens to unmask at each step
    tokens_per_step = seq_len / steps

    for step_idx in range(steps):
        # Map step to a diffusion timestep (going from T down to 1)
        t_val = int(schedule.T * (1 - step_idx / steps))
        t_val = max(t_val, 1)
        t = torch.full((batch_size,), t_val, dtype=torch.long, device=device)

        logits = model(x, t)  # (batch, seq_len, vocab)

        # Apply temperature
        logits = logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            top_k_vals, _ = logits.topk(top_k, dim=-1)
            threshold = top_k_vals[..., -1:]
            logits[logits < threshold] = float("-inf")

        probs = F.softmax(logits, dim=-1)

        # Sample tokens for all positions
        sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, seq_len)

        # Determine which tokens are still masked
        is_masked = x == mask_token_id  # (batch, seq_len)

        if not is_masked.any():
            break

        # Decide how many tokens to unmask this step
        num_to_unmask = int(round(tokens_per_step * (step_idx + 1))) - (seq_len - is_masked[0].sum().item())
        num_to_unmask = max(num_to_unmask, 1)

        if strategy == "confidence":
            # Unmask positions where model is most confident
            confidence, _ = probs.max(dim=-1)  # (batch, seq_len)
            confidence[~is_masked] = -1.0  # ignore already-unmasked
            _, indices = confidence.topk(min(num_to_unmask, is_masked.sum(-1).max().item()), dim=-1)
            unmask = torch.zeros_like(is_masked)
            unmask.scatter_(1, indices, True)
            unmask = unmask & is_masked
        elif strategy == "random":
            # Randomly select masked positions to unmask
            unmask = torch.zeros_like(is_masked)
            for b in range(batch_size):
                masked_indices = is_masked[b].nonzero(as_tuple=True)[0]
                n = min(num_to_unmask, len(masked_indices))
                perm = torch.randperm(len(masked_indices), device=device)[:n]
                unmask[b, masked_indices[perm]] = True
        else:  # linear
            # Unmask leftmost masked positions
            unmask = torch.zeros_like(is_masked)
            for b in range(batch_size):
                masked_indices = is_masked[b].nonzero(as_tuple=True)[0]
                n = min(num_to_unmask, len(masked_indices))
                unmask[b, masked_indices[:n]] = True

        # Commit sampled tokens at unmasked positions
        x[unmask] = sampled[unmask]

    # Fill any remaining masks in a final pass
    remaining = x == mask_token_id
    if remaining.any():
        t = torch.ones((batch_size,), dtype=torch.long, device=device)
        logits = model(x, t) / temperature
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, seq_len)
        x[remaining] = sampled[remaining]

    return x
