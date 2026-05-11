# tent-ad-domain-shift

> **Does test-time adaptation actually help in autonomous driving?**  
> A systematic ablation of TENT and variants on BDD100K domain shift scenarios.

---

## Motivation

Object detectors deployed in autonomous driving fail silently under distribution shift — a model trained on clear daytime footage degrades significantly at night or in adverse weather. Test-time adaptation (TTA) promises to fix this at inference, without retraining.

This project benchmarks whether that promise holds up. Using BDD100K as a realistic AD dataset, we evaluate TENT and ablate which components actually drive accuracy recovery — and where adaptation actively hurts.

---

## Problem Setup

| Split | Description |
|---|---|
| Source | BDD100K daytime, clear weather |
| Target 1 | BDD100K nighttime |
| Target 2 | BDD100K rain / fog |
| Backbone | FCOS (ResNet-50) |

No target labels are used at any point. Adaptation happens purely at inference time.

---

## Methods

| Method | What changes at test time | Our hypothesis |
|---|---|---|
| No adaptation (baseline) | Nothing | Lower bound |
| BN stats only | Running mean/var updated from test batch | Cheap, often underrated |
| TENT | BN params (γ, β) optimized via entropy minimization | Standard TTA baseline |
| TENT + augmentation consistency | Entropy loss over augmented views | More stable gradient signal |
| Train-time aug only | Heavy aug at train time, no TTA | Is TTA even necessary? |

The last row is the uncomfortable control — strong train-time augmentation may close most of the gap without any test-time overhead.

---

## Repository Structure

```
tent-ad-domain-shift/
├── data/
│   └── bdd100k/              # symlink or download instructions
├── src/
│   ├── models/               # FCOS backbone + head
│   ├── adaptation/
│   │   ├── tent.py           # core TENT implementation
│   │   ├── bn_adapt.py       # BN statistics adaptation
│   │   └── consistency.py    # augmentation consistency variant
│   ├── eval/
│   │   ├── benchmark.py      # domain shift evaluation loop
│   │   └── metrics.py        # mAP per domain split
│   └── train.py              # source domain training
├── configs/
│   └── bdd100k_fcos.yaml
├── notebooks/
│   └── results_analysis.ipynb
├── requirements.txt
└── README.md
```

---

## Results (preliminary)

| Method | mAP day | mAP night | mAP rain | Latency overhead |
|---|---|---|---|---|
| No adaptation | — | — | — | — |
| BN stats only | — | — | — | — |
| TENT | — | — | — | — |
| TENT + consistency | — | — | — | — |
| Train-time aug only | — | — | — | — |

*Results to be filled in as experiments complete.*

---

## Key Questions This Project Answers

1. How much accuracy is lost under realistic AD domain shift?
2. Does TENT recover meaningful performance, or just add inference complexity?
3. Is BN statistics adaptation alone sufficient — and cheaper?
4. When does entropy minimization make things worse?

---

## Setup

```bash
git clone https://github.com/Nickvanhurck/tent-ad-domain-shift
cd tent-ad-domain-shift
pip install -r requirements.txt
```

Download BDD100K from [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu/) and symlink to `data/bdd100k/`.

```bash
# Run baseline evaluation
python src/eval/benchmark.py --method none --split night

# Run TENT
python src/eval/benchmark.py --method tent --split night

# Full ablation
python src/eval/benchmark.py --method all --split all
```

---

## Requirements

```
torch>=2.0
torchvision>=0.15
numpy
opencv-python
pycocotools
onnxruntime
```

---

## References

- [TENT: Fully Test-Time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726) — Wang et al., ICLR 2021
- [BDD100K: A Diverse Driving Dataset](https://arxiv.org/abs/1805.04687) — Yu et al., CVPR 2020
- [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355) — Tian et al., ICCV 2019

---

## Status

- [ ] Source domain training pipeline
- [ ] Domain shift benchmark splits
- [ ] TENT implementation
- [ ] BN adaptation variant
- [ ] Consistency variant
- [ ] Full ablation results
- [ ] Write-up / findings

---

*Part of a 3-project portfolio on robust perception for autonomous driving.*  
*Related: [ssl-detection-bdd](https://github.com/Nickvanhurck/ssl-detection-bdd) · [detector-distillation-analysis](https://github.com/Nickvanhurck/detector-distillation-analysis)*
