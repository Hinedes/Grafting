# Grafting

Train small, swappable domain patches for frozen language models.  
One base model, many grafts. No inference overhead.

## What it does

A graft is a file. Install it on a base model and the model gets better at a specific domain: medicine, law, coding, niche slang. Remove it and the base model is untouched. Stack a few together and they mostly stay out of each other's way.

The graft only learns what the base model doesn't already know. The base handles ordinary English. The graft picks up the rest. This happens automatically because the base is frozen and the graft is confined to a fixed set of FFN channels.

## Why it works

Earlier versions used rotated subspaces. That was elegant but broke inside the SwiGLU gate. The element-wise multiply scattered the signal and created cross-talk between grafts.

Axis ARW drops the rotations. Each domain gets its own slice of the FFN intermediate channels. Gate and up projections for domain A never touch the same channels as domain B. The SiLU mixer has nothing to mix. Cross-talk drops roughly 25x compared to the rotated version.

## Install

```bash
pip install torch transformers datasets
```

## Quick start

Pull some training data, or bring your own JSONL files:

```bash
python dataset.py
```

Train a graft:

```bash
python train.py \
  --model HuggingFaceTB/SmolLM3-3B \
  --domain_data medical.jsonl \
  --ood_data minipile.jsonl \
  --domain_index 0 \
  --max_domains 4 \
  --output medical.graft.pt
```

Train more grafts by changing `--domain_index` (1, 2, 3) and the data files.

Check one graft:

```bash
python eval.py eval --graft medical.graft.pt --data medical.jsonl
```

Stack four grafts and test for interference:

```bash
python eval.py stack-test \
  --grafts medical.graft.pt legal.graft.pt coding.graft.pt niche.graft.pt \
  --data medical.jsonl legal.jsonl coding.jsonl niche.jsonl
```

Bake grafts permanently into a model:

```bash
python eval.py install \
  --graft medical.graft.pt legal.graft.pt coding.graft.pt niche.graft.pt \
  --output smol-grafted
```

## Preliminary Results (SmolLM3-3B, delta-only silence loss)

| metric | value |
|--------|-------|
| single-graft PPL (medical) | 1.09 |
| stacked PPL (medical, with legal) | 1.11 |
| stacked PPL (legal, with medical) | 2.94 |
| rotated ARW stacked degradation | +25.8 PPL |
| axis ARW stacked degradation | +0.02 PPL |

The remaining crosstalk comes from `down_proj` mapping all grafts back into the shared hidden size. The SiLU gate problem is solved. The residual stream is the next target.

## Hardware

The scripts pick up whatever PyTorch can see: CUDA, ROCm, MPS, or CPU. On CUDA and ROCm it uses bfloat16 where available. Weights are kept in float32 for training stability.

## Files

- `dataset.py` — pulls training data, wraps it for the loader
- `engine.py` — finds FFN layers, computes channel slices, attaches the graft during forward
- `train.py` — training loop
- `eval.py` — single-graft eval, stacked eval, model install

## Safety checks

The eval and install commands check that grafts:

- are the right artifact version
- match the model's layer names and weight shapes
- use the same `max_domains` setting when stacked
- don't overlap slices or duplicate domain indices
- get one test file per graft in `stack-test`

## Experiments

Ran on AMD MI300X hardware with support from the AMD Developer Cloud.