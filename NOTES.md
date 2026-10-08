# Latent Energy Matching for mel-spectrogram TTS — implementation spec

Goal: a test implementation of a text → mel-spectrogram model for a dialect continuum, using
Energy Matching (Balcerak et al., 2025, arXiv:2504.10612) in the latent space of a mel VAE.
No phonemisation; conditioning is (text, speaker embedding, dialect embedding).

Reference code: https://github.com/m1balcerak/EnergyMatching

---

## 1. Components

| # | Component | Status | Role |
|---|-----------|--------|------|
| 1 | Mel extractor | build | audio → mel, parameters must match the vocoder exactly |
| 2 | Mel VAE (encoder + decoder) | build | mel → latent `z` (T/r × C), `z` → mel |
| 3 | Tokenizer + text encoder | build | BPE/SentencePiece on dialect text → contextual token embeddings |
| 4 | Duration predictor | build | token embeddings + `s` + `d` → per-token frame counts |
| 5 | Speaker encoder | exists, frozen | audio → `s` |
| 6 | Dialect embedding | build | continuous vector `d`, learned jointly |
| 7 | Energy model `V_θ(z; c)` | build | scalar potential over latents, conditioned |
| 8 | Sampler | build | gradient descent + Langevin on `V` (+ optional extra energies) |
| 9 | Vocoder | exists, frozen | mel + `s` → audio |

Training-only: 1, 2 (encoder), negatives from the sampler.
Inference: 3 → 4 → 8 → 2 (decoder) → 9.

---

## 2. Data and shapes

- Mel: `(T, n_mels)`, log-mel, vocoder settings (sample rate, hop, n_mels, fmin/fmax).
- Latent: `z ∈ R^{T/r × C}`, e.g. `r = 4..8`, `C = 32..64`. Normalise `z` to roughly unit variance
  per channel (store mean/std, apply before energy model, invert before decoding).
- Text: BPE ids `(N,)`; encoder output `(N, D_txt)`.
- Durations: `(N,)` integers, sum = `T/r`. Expand text encoding to `(T/r, D_txt)`.
- `s ∈ R^{D_s}` from the existing speaker encoder, L2-normalised.
- `d ∈ R^{D_d}`, e.g. `D_d = 16`; initialise from region coordinates projected by a small MLP,
  then let it train.
- Conditioning bundle `c = (text_expanded, s, d)`.

---

## 3. Mel VAE

- 1-D conv encoder/decoder over time, channel dim `C`, temporal compression `r`.
- Loss: L1 mel reconstruction + small KL (β ≈ 1e-3 .. 1e-2). Optionally a mel-GAN style
  discriminator later; not needed for the test.
- Train first, then freeze. Check: decoded mel through the vocoder is intelligible.

---

## 4. Alignment / duration

For the test, bootstrap durations from an external aligner (CTC forced alignment with a
character-level ASR, or any available aligner) at the latent frame rate `T/r`.

Duration predictor: 2–3 conv or transformer layers on token embeddings, FiLM-conditioned on
`(s, d)`, predicts `log(duration)`. Loss: MSE in log domain. At inference: round, clamp ≥ 1.

---

## 5. Energy model `V_θ(z; c)`

Architecture (time-independent — no `t` input anywhere):

```
z (L, C) ──► conv-in ──► [N × Block] ──► transformer head ──► masked sum over time ──► scalar
                           │
                           ├── FiLM(s, d): h ← γ(s,d)·h + β(s,d)   (per block, per channel)
                           └── cross-attention: queries = latent frames, keys/values = text_expanded
```

- Blocks: 1-D residual conv blocks (UNet-style with 2–3 down/up levels is fine) or a plain
  transformer over latent frames. Start small: ~5–10 M params for the test.
- Head: linear per frame → masked sum over valid frames → scalar. Do not pool through a global
  bottleneck before the per-frame contribution, or per-frame gradients wash out.
- Output scale: multiply the scalar by a constant (paper uses 1000 on images); tune so that
  `‖∇_z V‖` matches `‖z_data − z_0‖` at initialisation order of magnitude.
- Gradient: `g = torch.autograd.grad(V.sum(), z, create_graph=True)[0]` during training.
  Use `create_graph=False` in the sampler.

Masking: pad to the batch max length; multiply per-frame contributions by the mask; also mask
the OT loss to valid frames.

---

## 6. Training objectives

Notation: `z_1` = data latent (from the VAE encoder), `z_0 ~ N(0, I)` same shape,
`z_t = (1 − t) z_0 + t z_1`.

### 6.1 Phase 1: OT loss (warm-up, most of the compute)

```
t ~ U(0, τ*)                     # τ* = 1.0
L_OT = ‖ ∇_z V_θ(z_t; c) + (z_1 − z_0) ‖²   (masked mean over frames and channels)
```

Minibatch OT coupling (POT) between `{z_0}` and `{z_1}` is optional. For the test: independent
coupling first; if adding OT, bucket by length so tensors match in shape.

### 6.2 Phase 2: add contrastive divergence (short, after Phase 1 converges)

Negatives: run the sampler (Section 7) under the **same `c` as the positive**. Half the chains
start from `z_1` (data), half from noise. Stop gradient through sampling.

```
L_pos = mean_batch  V_θ(z_1; c)
L_neg = trimmed_mean_batch V_θ(z_neg; c)     # drop top α fraction, α ≈ 0.1–0.2
L_CD  = (L_pos − L_neg) / ε_max
L_CD  = max(L_CD, −β)                        # clamp, β ≈ 0.01–0.02
L     = L_OT + λ_CD · L_CD                   # λ_CD ≈ 1e-4 .. 1e-3
```

Temperature schedule used by the sampler (and by negatives):

```
ε(t) = 0                                   for t < τ*
     = ε_max · (t − τ*) / (1 − τ*)         for τ* ≤ t ≤ 1    (if τ* < 1; else jump)
     = ε_max                               for t ≥ 1
```

With `τ* = 1.0`, Phase 1 is pure transport and the Langevin noise switches on at `t = 1`.

### 6.3 Optional regularisers

- Dialect smoothness: for pairs of regions `(i, j)` with geographic distance `g_ij`,
  penalise `‖d_i − d_j‖² / g_ij` (or a Laplacian on the region graph).
- Duration loss (Section 4) trained jointly or separately.

---

## 7. Sampler (inference and negatives)

```
z ← N(0, I)  of shape (sum(durations), C)
Δt = 0.01;  N = τ_s / Δt                 # τ_s ≈ 1.5–3.25; sweep it
for n in 0..N−1:
    t = n·Δt
    g = ∇_z U(z)                         # U = V_θ(z; c) + extra energies (Section 8)
    η = N(0, I)
    z ← z − Δt·g + sqrt(2·ε(t)·Δt)·η      # Euler; Heun for the first regime is slightly better
mel = VAE.decode(z)
audio = vocoder(mel, s)
```

Reduce `N` once it works (try `Δt = 0.05` with Heun); step count is the main cost driver.

---

## 8. Extra energies (what the scalar buys you)

All are summed into `U(z)` at sampling time, no retraining:

| Energy | Formula | Use |
|--------|---------|-----|
| Infilling / SSL fidelity | `ε(t)/ζ² · ‖M ⊙ (z − z_obs)‖²` | continue or edit a real recording; masked-latent pretraining check |
| Speaker consistency | `λ · ‖f_spk(decode(z)) − s‖²` | requires a differentiable (mel-domain) speaker encoder |
| Dialect push | `λ · ‖f_dial(z) − d‖²` or `−λ · log p_dial(region | z)` | push toward a region via a classifier |
| Dialect mixing | `α·V(z; d_A) + (1−α)·V(z; d_B)` | product-of-experts interpolation |
| Repulsion (M chains) | `−ε/σ² · Σ_{k≠m} ‖B(z_m − z_k)‖²` | diverse prosody for the same text |

Dialect transfer of a real utterance: `z ← encode(mel)`, swap `d`, run Langevin only
(`ε = ε_max`) for a few hundred steps.

---

## 9. Hyperparameters to start from (adapted from the paper)

| Name | Value | Notes |
|------|-------|-------|
| `τ*` | 1.0 | switch to Langevin at t = 1 |
| `τ_s` | 2.0 | sweep 1.0–3.25 |
| `Δt` | 0.01 | |
| `ε_max` | 0.01–0.05 | relative to latent scale; too high = noisy mel |
| `M_Langevin` (negatives) | 50–200 | |
| `λ_CD` | 1e-4 – 1e-3 | |
| `α` (trim) | 0.1 | |
| `β` (clamp) | 0.02 | |
| lr | 1e-4 – 3e-4, Adam | |
| EMA | 0.999 | use EMA weights for sampling |
| Phase 1 / Phase 2 iters | e.g. 100k / 2k | Phase 2 is short |

---

## 10. Test plan

Run in order; each step has a pass criterion.

1. **Toy check.** Reproduce 2-moons with the reference code to understand `τ*, τ_s, ε_max, λ_CD`.
2. **Mel VAE.** Reconstruction → vocoder → listen. Pass: intelligible, speaker preserved.
3. **Unconditional latent EM, Phase 1 only.** Condition on nothing (or speaker only).
   Pass: samples at `τ_s = 1` decode to speech-like mel; quality *degrades* for `τ_s > 1`
   (expected signature before Phase 2).
4. **Add Phase 2.** Pass: (a) `V(z_data) < V(z_data + noise) < V(z_noise)` on held-out data;
   (b) a Fréchet distance on wav2vec/HuBERT features or UTMOS plateaus instead of diverging as
   `τ_s` grows.
5. **Speaker conditioning (SSL stage, audio only).** Pass: speaker-encoder cosine similarity
   between target `s` and `f_spk(vocoder(decode(z)))` > a chosen threshold; infilling via the
   fidelity energy reconstructs masked regions.
6. **Text + durations + dialect (paired data).** Pass: WER from a dialect-robust ASR on
   synthesised speech; dialect classifier accuracy on synthesised speech ≥ on-target.
7. **Energy-specific features.** Pass: dialect mixing gives monotone classifier probability vs.
   `α`; `V(z; d_A) − V(z; d_B)` separates held-out A/B utterances; dialect transfer of a real
   recording keeps WER and speaker similarity.
8. **Latency.** Sweep step count and `Δt`; report real-time factor.

---

## 11. Known pitfalls

- Negatives under a different `c` than the positive make `L_CD` rank conditions, not data.
- Forgetting `create_graph=True` in training gives zero gradient to `θ` through `∇_z V`.
- `V` summed over frames: longer utterances have larger `|V|`; compare energies per frame when
  diagnosing, and keep the clamp/trim on per-frame-normalised values.
- No time input: the sampler cannot know its progress; `τ_s` must be tuned, and Phase 2 is what
  keeps samples from drifting past the data manifold.
- Latent normalisation mismatch between VAE training and energy training silently breaks
  `ε_max` scaling.
- Expect inference to be 1–2 orders of magnitude slower per utterance than a 10-step flow model
  until step count is reduced or a flow is distilled from the energy model.
