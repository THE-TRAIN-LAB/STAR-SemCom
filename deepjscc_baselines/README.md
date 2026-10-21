# DeepJSCC baselines

Learned baselines for **STAR: Semantic-Criticality Tiered Allocation of Radio Resources via Zero-Shot
Learning**, by Christo Kurisummoottil Thomas.

STAR delivers the whole image to a remote control center and protects only its task-causal pixels, which a
frozen vision-language model selects zero-shot. This folder contains the **trained, task-specific
alternatives** that STAR is compared with. These are three DeepJSCC variants that send complex symbols
straight onto the OFDM resource elements (REs) of the **same 5G NR link** as STAR. The paper's
*hybrid DeepJSCC* (Sec. IV, Figs. 4–6) is the semantic-aware model below.

| Baseline | File | Receiver gets | REs per image |
|---|---|---|---|
| Reconstruction DeepJSCC | `djscc_baseline.py`, `train_djscc.py` | lossy image | 0.79 M |
| Task-oriented DeepJSCC | `task_djscc.py` | 7 answers, no image | 256 |
| Semantic-aware (hybrid) DeepJSCC | `hybrid_djscc.py` | lossy image **and** 7 answers | 0.79 M |

No VLM or LLM is used here. The task-oriented and hybrid models answer the seven factory-monitoring
questions with trained classification heads, so semantic accuracy (S-Acc) is a classification accuracy.

## Channel (identical for all baselines and for STAR)

Sionna 2.0 (PyTorch) with the following link:

- 30 kHz subcarrier spacing, 14 OFDM symbols × 128 subcarriers (1792 data REs per slot), 3.5 GHz carrier.
- 3GPP TDL-A channel with 100 ns delay spread, drawn independently per slot.
- Perfect CSI, LMMSE equalization, fixed $E_s/N_0$ with $E_s = 1$ per complex symbol.

The encoder output is power-normalized per image and mapped directly onto the REs (`SionnaLink` in
`djscc_baseline.py`). The link is differentiable, so all models are trained **through** it, with
$E_s/N_0 \sim \mathcal U[-8, 18]$ dB drawn at every step.

## Models

**Reconstruction DeepJSCC** (Bourtsoulatze et al. style, 2.58 M parameters).
- Encoder: Conv 5×5/2 (3→128), Conv 5×5/2, 2 × Conv 5×5, Conv 5×5 (128→16), PReLU. The 16 × H/4 × W/4
  latent is read as complex pairs, giving a bandwidth ratio of 1/6.
- Decoder: a mirror of the encoder with two transposed convolutions and a sigmoid.
- Training: MSE loss, Adam 1e-4, 256 × 256 crops, batch 16, held-out factory panels 300–499. The released
  checkpoint is at step 5,000.

**Task-oriented DeepJSCC** (11.07 M parameters).
- Encoder: input downsampled by 2, then 5 stages of [Conv 3×3/2, BN, ReLU, Conv 3×3, BN, ReLU] with 64, 64,
  128, 128 and 256 channels, average pooling to 6 × 10, and a linear layer to $k = 256$ complex symbols.
- Receiver: an MLP (512 → 1024 → 1024) and seven linear heads with 3, 41, 56, 2, 6, 2 and 2 classes.
- Training: sum of cross-entropies, AdamW 1e-3, warm-up + cosine schedule, batch 16, 20,000 steps.
- The optional `--digits` flag reads pressure and temperature digit by digit. It was not used for the
  released checkpoint.

**Semantic-aware (hybrid) DeepJSCC** (12.18 M parameters).
- The reconstruction DeepJSCC above, frozen, delivers the image.
- A CNN on the **received** latent answers the seven tasks: 4 stages with GroupNorm (64–128 channels),
  pooling to 6 × 10, then an MLP (1024 → 1024). Pressure and temperature are read digit by digit.
- Training: heads trained with cross-entropy, AdamW 1e-3, batch 16, full resolution, 10,000 steps
  (`--freeze_jscc`).
- Joint fine-tuning of image and answers is supported but was not used for the released results.

Training images for the task heads are rendered on the fly from a seed that never produces a test image.
Evaluation uses the paper's 60 factory test images (20 per event, `SEED_RUN = 20240612`, panels 0–299).

## Results (factory-monitoring dataset, 60 test images)

S-Acc (mean over the seven tasks of Table I of the paper):

| $E_s/N_0$ (dB) | −8 | −4 | 0 | 4 | 8 | 12 | 18 |
|---|---|---|---|---|---|---|---|
| Task-oriented DeepJSCC (256 REs) | 0.74 | 0.86 | 0.92 | 0.97 | 0.97 | 0.99 | 0.99 |
| Semantic-aware DeepJSCC (0.79 M REs) | 0.77 | 0.84 | 0.86 | 0.86 | 0.87 | 0.86 | 0.87 |

- **Hybrid:** at 18 dB it reads status, pressure, power, room, fire and technician (1.00 each) but not the
  temperature (≈ 0.07), which caps its S-Acc at 0.87. Its image reaches 28.2 dB PSNR and 25.8 dB on the
  ground-truth text regions.
- **Task-oriented:** it degrades at low SNR, first on the numeric readings.

Per-task numbers are in `results/`, training logs in `logs/`, and released weights in `checkpoints/`.

## Run

The factory-monitoring generator and scene templates live at the root of this repository, so run every
command **from the repository root**. If you want to train, first render the images (see the main README),
for example with `python factory_dataset/generate_factory_dataset.py --n 500 --seed 2027`.

```bash
pip install -r deepjscc_baselines/requirements.txt   # install PyTorch with CUDA for your platform first
```

Every training script saves a checkpoint periodically and accepts `--resume`. Checkpoints in `checkpoints/`
contain weights only.

```bash
# Reconstruction DeepJSCC (ratio 1/6)
python deepjscc_baselines/train_djscc.py --root . --factory --c 16 --out deepjscc_baselines/checkpoints/djscc_factory_c16.pt

# Task-oriented DeepJSCC (k = 256 REs), then evaluate on the 60 test images
python deepjscc_baselines/task_djscc.py train --k 256 --steps 20000 --out deepjscc_baselines/checkpoints/task_k256.pt
python deepjscc_baselines/task_djscc.py eval  --ckpt deepjscc_baselines/checkpoints/task_k256.pt \
                                              --out deepjscc_baselines/results/task_k256.json

# Semantic-aware DeepJSCC: heads on the frozen reconstruction model, then evaluate
python deepjscc_baselines/hybrid_djscc.py train --init deepjscc_baselines/checkpoints/djscc_factory_c16.pt \
       --freeze_jscc --batch 16 --steps 10000 --out deepjscc_baselines/checkpoints/hybrid_head.pt
python deepjscc_baselines/hybrid_djscc.py eval  --ckpt deepjscc_baselines/checkpoints/hybrid_head.pt \
                                                --out deepjscc_baselines/results/hybrid_head.json

# Curves (add --star base_runs_final.pkl to overlay STAR from the factory notebook)
python deepjscc_baselines/plot_baselines.py --run "Task JSCC=deepjscc_baselines/results/task_k256.json" \
       --run "Semantic DeepJSCC=deepjscc_baselines/results/hybrid_head.json" --out deepjscc_baselines/figs/baselines
```

To evaluate the released models without training, run only the `eval` and plotting commands.

Approximate speed on a single Colab GPU: 0.38 s/step (reconstruction), 0.61 s/step (task-oriented),
1.8 s/step (hybrid heads, full resolution).

**Google Colab.** `pip install sionna==2.0.1` upgrades NumPy inside a running session. Restart the session
(Runtime → Restart session) before importing the modules in a notebook. Commands started with `!python`
are unaffected.

## Requirements

Python ≥ 3.11, PyTorch ≥ 2.4 (CUDA recommended), Sionna 2.0.1, NumPy, Pillow, Matplotlib.
