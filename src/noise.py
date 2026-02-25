"""Noise schedule and token corruption for discrete diffusion."""

import torch
import math


class NoiseSchedule:
    """Masking noise schedule: maps timestep t -> masking probability gamma(t).

    At t=0 text is clean, at t=T text is fully masked.
    """

    def __init__(self, num_timesteps: int = 1000, schedule: str = "cosine"):
        self.T = num_timesteps
        self.schedule = schedule

        # Precompute gamma values for t=0..T
        ts = torch.arange(0, num_timesteps + 1, dtype=torch.float32)
        if schedule == "cosine":
            self.gammas = 1.0 - torch.cos(math.pi * ts / (2 * num_timesteps))
        elif schedule == "linear":
            self.gammas = ts / num_timesteps
        elif schedule == "log_linear":
            self.gammas = 1.0 - (1.0 - ts / num_timesteps) ** 2
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """Return masking probability for each timestep in batch.

        Args:
            t: (batch_size,) integer timesteps in [1, T]

        Returns:
            (batch_size,) masking probabilities in [0, 1]
        """
        return self.gammas[t.cpu()].to(t.device)


def corrupt(
    x_0: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt clean tokens by replacing with [MASK] at rate gamma(t).

    Args:
        x_0: (batch, seq_len) clean token IDs
        t: (batch,) timesteps
        schedule: noise schedule instance
        mask_token_id: token ID to use for masking

    Returns:
        x_t: (batch, seq_len) corrupted token IDs
        mask: (batch, seq_len) bool tensor, True where tokens were masked
    """
    gamma = schedule.gamma(t)  # (batch,)
    # Expand to (batch, 1) for broadcasting over seq_len
    gamma = gamma.unsqueeze(1)

    # Each token is independently masked with probability gamma
    rand = torch.rand_like(x_0, dtype=torch.float32)
    mask = rand < gamma  # (batch, seq_len)

    x_t = x_0.clone()
    x_t[mask] = mask_token_id

    return x_t, mask
