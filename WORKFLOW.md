# Small Diffusion LLM — Project Workflow

## Overview

Build a small discrete diffusion language model using masked diffusion (MDLM-style). The model learns to denoise corrupted text by predicting masked tokens, then generates text by iteratively unmasking from a fully masked sequence.

---

## Phase 1: Environment & Project Setup

- [ ] Initialize Python project with `pyproject.toml`
- [ ] Install dependencies: `torch`, `transformers`, `datasets`, `wandb` (optional logging)
- [ ] Set up directory structure:

```
diff/
├── WORKFLOW.md
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── model.py          # Transformer backbone
│   ├── noise.py           # Noise schedule & corruption
│   ├── diffusion.py       # Forward/reverse diffusion logic
│   ├── train.py           # Training loop
│   ├── sample.py          # Inference / text generation
│   └── tokenizer.py       # Tokenizer wrapper
├── configs/
│   └── default.yaml       # Hyperparameters
├── data/
│   └── README.md          # Dataset instructions
└── scripts/
    ├── train.sh
    └── sample.sh
```

- [ ] Verify GPU availability (`torch.cuda.is_available()` or MPS on Mac)

---

## Phase 2: Data Pipeline

- [ ] Choose a small dataset:
  - **TinyStories** (~2M short stories, good for small models)
  - **WikiText-2** (~2M tokens, standard benchmark)
  - **OpenWebText subset** (for more diverse text)
- [ ] Load dataset via HuggingFace `datasets`
- [ ] Tokenize with a pretrained tokenizer (e.g., `bert-base-uncased` vocab, ~30k tokens)
- [ ] Chunk text into fixed-length sequences (128 or 256 tokens)
- [ ] Build PyTorch `DataLoader` with batching and shuffling

**Output:** A dataloader yielding `(batch_size, seq_len)` token ID tensors.

---

## Phase 3: Noise Schedule

- [ ] Implement a masking noise schedule
  - At timestep `t=0`: clean text
  - At timestep `t=T`: fully masked (all `[MASK]` tokens)
  - Intermediate `t`: each token independently masked with probability `gamma(t)`
- [ ] Define `gamma(t)` schedule:
  - **Linear:** `gamma(t) = t / T`
  - **Cosine:** `gamma(t) = 1 - cos(pi * t / (2T))` (recommended, smoother)
  - **Log-linear:** `gamma(t) = 1 - (1 - t/T)^2`
- [ ] Implement `corrupt(x_0, t)` function:
  - Input: clean token IDs `x_0`, timestep `t`
  - For each token, replace with `[MASK]` with probability `gamma(t)`
  - Return: corrupted `x_t` and the binary mask of which tokens were masked

**Output:** `corrupt(x_0, t) -> (x_t, mask)`

---

## Phase 4: Model Architecture

- [ ] Build a small transformer encoder (bidirectional, BERT-style):
  - **Embedding layer:** token embeddings + positional embeddings
  - **Timestep conditioning:** embed timestep `t` and add/concat to input
  - **Transformer blocks:** 6-12 layers, 256-512 hidden dim, 4-8 heads
  - **Output head:** linear projection to vocabulary size (logits per position)
- [ ] Target model sizes:
  - **Tiny:** 6 layers, 256 dim, 4 heads (~10M params)
  - **Small:** 8 layers, 512 dim, 8 heads (~40M params)
  - **Medium:** 12 layers, 512 dim, 8 heads (~60M params)
- [ ] Timestep embedding options:
  - Sinusoidal embedding (like original diffusion models)
  - Learned embedding table
  - AdaLayerNorm conditioning (inject into each layer)

**Output:** `model(x_t, t) -> logits (batch, seq_len, vocab_size)`

---

## Phase 5: Training Loop

- [ ] Define training objective:
  ```
  For each batch:
    1. Sample clean text x_0
    2. Sample random timestep t ~ Uniform(1, T)
    3. Corrupt: x_t = corrupt(x_0, t)
    4. Predict: logits = model(x_t, t)
    5. Loss = cross_entropy(logits[masked_positions], x_0[masked_positions])
  ```
- [ ] Only compute loss on masked positions (not unmasked tokens)
- [ ] Hyperparameters:
  - Batch size: 64-128
  - Learning rate: 1e-4 to 3e-4 (AdamW)
  - Weight decay: 0.01
  - Warmup steps: 1000
  - Total steps: 50k-200k
  - T (total diffusion steps): 1000
- [ ] Add gradient clipping (max norm 1.0)
- [ ] Log loss curves (terminal, wandb, or tensorboard)
- [ ] Save checkpoints every N steps

**Output:** Trained model checkpoint.

---

## Phase 6: Sampling / Generation

- [ ] Implement iterative unmasking:
  ```
  1. Start with x_T = all [MASK] tokens (length = desired output length)
  2. For t = T, T-1, ..., 1:
     a. Predict logits = model(x_t, t)
     b. For each masked position, sample token from softmax(logits / temperature)
     c. Decide which tokens to "unmask" (commit) at this step
     d. Update x_{t-1}: committed tokens stay, rest remain [MASK]
  3. Return x_0
  ```
- [ ] Unmasking strategies:
  - **Linear unmasking:** unmask `N/T` tokens per step
  - **Confidence-based:** unmask tokens where model is most confident first
  - **Random subset:** randomly choose which masked tokens to reveal
- [ ] Sampling hyperparameters:
  - Temperature: 0.7-1.0
  - Top-k / top-p filtering on token predictions
  - Number of diffusion steps at inference (can be fewer than T)
- [ ] Support few-step generation (e.g., 10-50 steps instead of 1000)

**Output:** Generated text from noise.

---

## Phase 7: Evaluation

- [ ] **Perplexity:** Estimate via ELBO (evidence lower bound)
- [ ] **Generation quality:** Manual inspection of samples
- [ ] **Diversity:** Measure distinct n-grams across samples
- [ ] **Controllability test:** Fix some tokens, generate the rest (infilling)
- [ ] Compare against:
  - A small autoregressive model of the same size
  - Random baseline

---

## Phase 8: Experiments & Extensions

- [ ] **Text infilling:** Mask only the middle of a sequence, let the model fill it in
- [ ] **Classifier-free guidance:** Train with occasional unconditional passes (drop conditioning), use guidance scale at inference
- [ ] **Variable-length generation:** Predict sequence length, or use an EOS-aware masking scheme
- [ ] **Speed optimizations:**
  - Caching for repeated unmasked positions
  - Reduce inference steps with DDIM-style skipping
- [ ] **Scaling:** Increase model size, dataset, or sequence length

---

## Key Reference Papers

| Paper | Year | Key Idea |
|-------|------|----------|
| D3PM (Austin et al.) | 2021 | Discrete diffusion with transition matrices |
| Diffusion-LM (Li et al.) | 2022 | Continuous diffusion for controllable text |
| MDLM (Sahoo et al.) | 2024 | Masked discrete LM with simple training |
| SEDD (Lou et al.) | 2024 | Score entropy for discrete diffusion |
| Plaid 1B (Gulrajani & Hashimoto) | 2024 | Scaling discrete diffusion to 1B params |

---

## Quick-Start Checklist

1. Set up environment and install PyTorch
2. Load and tokenize TinyStories
3. Implement cosine noise schedule
4. Build a 6-layer transformer with timestep conditioning
5. Train for 50k steps on a single GPU
6. Sample text and inspect outputs
7. Iterate on architecture, schedule, and sampling strategy
