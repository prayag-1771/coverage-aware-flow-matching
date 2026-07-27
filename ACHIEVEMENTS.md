# Achievements

Running record of what has been established, built, and measured.
Every number here was computed from public data on this machine and is
reproducible from the scripts in `experiments/`.

**Last updated:** 2026-07-28

This is a living record. Everything measured goes in — results that supported the
hypothesis and results that did not, bugs found in our own code, claims of ours
that later proved wrong, and practical limits hit along the way. The failures are
recorded as carefully as the findings: they are what makes it possible to answer
"did you check X?" with evidence rather than recollection.

**Raw numbers live in `results/` (gitignored, regenerable). Anything needed to
defend a claim is copied into this file**, because re-running the experiments
costs hours and a result that exists only in a log is a result that will be lost.

---

## 1. Findings

Results that did not exist before this work. Listed strongest first.

### 1.0 SMOTE and ADASYN can be worse than no augmentation at all

On UNSW-NB15, both standard resamplers score **below the unaugmented baseline**
on macro-F1, and the damage on individual rare classes is large.

| arm | macro-F1 | Shellcode F1 | Worms F1 |
|---|---|---|---|
| none | 0.511 | 0.495 | 0.514 |
| random oversample | **0.526** | 0.422 | **0.720** |
| SMOTE | **0.487** ↓ | **0.357** ↓ | **0.382** ↓ |
| ADASYN | **0.487** ↓ | **0.352** ↓ | **0.412** ↓ |
| flowmatch | 0.511 | 0.507 | 0.556 |
| flowmatch_pertype | 0.510 | **0.513** | 0.549 |

SMOTE costs 0.139 F1 on Shellcode and 0.132 on Worms relative to doing nothing.

**This directly contradicts the same experiment on NSL-KDD**, where SMOTE improved
every rare class (U2R 0.219 → 0.428). The *direction* of the SMOTE effect is
dataset-dependent. Papers reporting a SMOTE gain on one benchmark and
generalising are not entitled to.

**Two further results in the same table:**

- **Random oversampling — plain duplication — wins Worms outright at 0.720**,
  well clear of flow matching's 0.556 and nearly double SMOTE's 0.382. Worms has
  130 training rows. Copying preserves them exactly; every synthesis method
  distorts them. Below some sample count, repeating data beats inventing it, and
  locating that threshold predicts when a generative augmenter is the wrong tool.
- **Flow matching scores exactly 0.000 on Analysis**, on every seed, in both
  variants — the classifier never emits the class. Analysis has 2,000 training
  rows, *more* than Shellcode's 1,133. Not sample starvation: Analysis and
  Backdoor overlap Exploits and Normal in feature space, and smooth synthetic
  samples spread further into that overlap than SMOTE's interpolations between
  real points, which cannot leave the manifold.

*Reproduce:* `experiments/05_unsw_comparison.py`

### 1.0b Undersampling the majority makes rare-class detection worse

The first step of nearly every imbalance pipeline is to cut the majority class
down. Measured on CICIDS2017, one seed, identical in every other respect:

| | rows | Bot imbalance | macro-F1 | **Bot F1** |
|---|---|---|---|---|
| **uncapped** (full BENIGN) | 1,979,513 | **1:1,161** | **0.9605** | **0.822** |
| BENIGN capped at 100,000 | 489,589 | 1:129 | 0.9455 | 0.761 |

**The version with ninefold worse nominal imbalance detects the rare class
better.** Discarding majority data to improve a class ratio costs 0.06 F1 on the
rarest usable class — the extra majority data sharpens the decision boundary,
and the ratio is not what the classifier is limited by.

This finding invalidated our own design: the first CICIDS2017 run capped BENIGN
purely for tractability and was measuring an artifact of that choice. Replaced
with a cap on the *oversampling target* instead, which keeps every real row.

*Reproduce:* `src/augment/resampling.py::_target_counts`, `experiments/06`

### 1.1 Synthetic-sample fidelity does not predict downstream utility

The implicit justification for preferring a generative model over simple
interpolation is that its samples look more realistic, and that realism should
translate into better detection. Measured directly, it does not.

Training a discriminator to separate synthetic minority samples from real ones:

| method | class | detection AUC | NN ratio | KS median | resulting R2L F1 |
|---|---|---|---|---|---|
| SMOTE | r2l | **0.535** (indistinguishable) | 0.63 | 0.011 | 0.246 |
| ADASYN | r2l | 0.989 (separable) | 9.54 | 0.203 | **0.251 — best of all arms** |
| ADASYN | u2r | **1.000** (perfectly separable) | 1.21 | 0.103 | — |

ADASYN's synthetic rows are identified as fake **100 % of the time** on U2R, and
ADASYN still produces the highest R2L F1 of any method tested.

**Why it matters.** A synthetic-data evaluation reporting only fidelity metrics
can rank methods in the wrong order. It also removes the justification for
preferring an expensive generator on realism grounds alone — the choice becomes
empirical, not principled.

**Status.** Replaces an uncited claim in the paper Introduction that was drawn
from a search snippet and could not be sourced. Now rests on our own
measurement, on a cleaner demonstration than the literature provides.

*Reproduce:* `experiments/03_flow_matching_gate.py`, `04_per_type_gate.py`

### 1.2 NSL-KDD's rare-class failure is distribution shift, not class imbalance

The R2L failure on this benchmark is quoted throughout the literature as
evidence of an imbalance problem. It is largely not one.

| attack type | train | test |
|---|---|---|
| `warezclient` | **890** (89 % of R2L train) | **0** |
| `guess_passwd` | 53 | 1,231 |
| `warezmaster` | 20 | 944 |
| `snmpguess` | **0** | 331 |
| `snmpgetattack` | **0** | 178 |
| `httptunnel` | **0** | 133 |

- 89 % of R2L training data is a single attack type absent from the test set.
- 642 of 2,887 R2L test samples (22 %) are attack types with **no** training
  representation.
- 30 of 67 U2R test samples (45 %) are likewise unseen types.
- R2L is 0.79 % of training data but 12.8 % of test data.

**Consequence.** The *effective* R2L training size is ≈ **105 rows**, not 995 —
an effective imbalance near **1 : 641** rather than the 1 : 68 implied by the
class counts. A generator fitted to R2L learns to reproduce `warezclient`, which
cannot help detect `snmpguess`. No augmentation method can close this gap.

**Consequence for the project.** NSL-KDD was demoted to a *diagnostic* dataset.
Headline claims move to CICIDS2017 and UNSW-NB15.

*Reproduce:* `experiments/00_class_distribution.py`

### 1.3 The base paper's results table has rotated class labels

Alsubaei (*Scientific Reports*, 2025, Q1) evaluates on `KDDTrain+_20Percent`
(25,192 records — confirmed against our copy). Its reported per-class counts
against the true distribution we computed from that file:

| their label | their count | true class at that count | true count |
|---|---|---|---|
| normal | 9,181 | dos | 9,234 |
| dos | 2,357 | probe | 2,289 |
| probe | 224 | r2l | 209 |
| r2l | **11** | **u2r** | **11** |
| u2r | **13,422** | normal | 13,449 |

Every reported class matches the *next* class's true count, within 1 %.

**Certain:** 13,422 U2R records cannot exist. NSL-KDD contains 119 U2R records
in total; this subset contains 11. The figure is wrong by a factor over 1,000.

**Strongly implied:** the widely-cited "R2L F1 = 0.15" is actually **U2R**, and
the reported "perfect U2R detection (1.00)" is actually the **normal** class —
trivially easy, which explains the perfect score.

**Consequence.** The motivating premise survives — a rare class does fail at
F1 ≈ 0.15 — but the class name attached to it is wrong. Our own baseline is used
instead. *Before publication, verify against the typeset table and state the
observation neutrally, evidenced by counts impossible for the dataset.*

*Reproduce:* comparison in `PLAN.md` §7

### 1.4 Standard practice generates structurally invalid network records

60 % of NSL-KDD's nominally numeric columns are not continuous:

| kind | count | examples |
|---|---|---|
| binary flags | 6 | `land`, `logged_in`, `root_shell` |
| low-cardinality counts | 6 | `num_shells`, `su_attempted` |
| integer counts | 11 | `src_bytes`, `duration`, `count` |
| genuinely continuous | 15 | `serror_rate`, `same_srv_rate` |

Interpolating or generating these as continuous produces values like
`land = 0.37` and `root_shell = 0.62`, which correspond to no real connection.

**This is not only our bug.** `SMOTENC` protects only the three declared
categorical columns and interpolates the binary ones. As far as we can determine
the practice is universal in this literature.

**Effect of fixing it in our generator:** U2R KS median fell **0.424 → 0.056**;
detection AUC **1.0000 → 0.9468**.

---

## 2. Built and verified

| component | file | status |
|---|---|---|
| NSL-KDD loader, explicit 23→5 mapping, errors on unmapped labels | `src/data/nsl_kdd.py` | working |
| Leakage-safe preprocessing (fitted on train only) | `src/data/preprocess.py` | working |
| Per-class metrics: precision/recall/F1/PR-AUC/support | `src/eval/metrics.py` | working |
| Centralised classifier config | `src/models/classifier.py` | working |
| Classical arms: none / ROS / SMOTE(NC) / ADASYN | `src/augment/resampling.py` | working |
| Flow-matching generator | `src/augment/flow_matching.py` | working, fidelity below SMOTE |
| Per-attack-type generation | `src/augment/per_type.py` | working |
| Sample-quality diagnostics | `experiments/03`, `04` | working |
| Multi-seed comparison runner | `experiments/02_resampling_comparison.py` | 6 arms, running |
| Paper: Introduction + bibliography | `paper/main.tex`, `references.bib` | drafted |

Environment: Python 3.11, pinned `requirements.txt`. pandas held at 2.2.3 —
3.0.5 is blocked by Windows Application Control.

---

## 3. Measured results

All figures: XGBoost, mean ± sd over 5 seeds, synthetic data added to the
training split only, test split evaluated once.

### 3.1 NSL-KDD — official `KDDTrain+` / `KDDTest+` split (CPU)

| arm | R2L F1 | U2R F1 | macro-F1 | accuracy |
|---|---|---|---|---|
| none | 0.192 ± .025 | 0.219 ± .018 | 0.569 ± .011 | 0.781 |
| random oversample | 0.192 ± .040 | 0.355 ± .027 | 0.601 ± .015 | 0.785 |
| SMOTE | 0.246 ± .010 | **0.428 ± .027** | **0.627 ± .008** | 0.788 |
| ADASYN | 0.251 ± .015 | 0.331 ± .036 | 0.618 ± .009 | 0.798 |
| flowmatch | 0.238 ± .027 | 0.387 ± .021 | 0.629 ± .010 | 0.798 |
| flowmatch_pertype | **0.263 ± .029** | 0.278 ± .022 | 0.617 ± .010 | **0.802** |

- Accuracy 0.781 against macro-F1 0.569 — the gap the project is about.
- R2L precision 0.974 with recall 0.107: the classifier is almost never wrong
  when it flags R2L, it simply declines to flag it.
- **Flow matching does not beat SMOTE here.** Per-type takes the best R2L score
  but by 0.013 against a standard deviation of 0.029 — not a significant win.
  SMOTE holds U2R decisively (0.428 vs 0.278, roughly six times the pooled sd).
- **Coverage explains the split between the two flow variants.** Per-type models
  97% of R2L rows (3 of 8 attack types) and wins that class; it models 58% of
  U2R (1 of 4 types — only `buffer_overflow` clears the sample threshold) and
  comes last of the augmented arms.

### 3.2 UNSW-NB15 — official split (GPU)

| arm | Analysis | Backdoor | Shellcode | Worms | macro-F1 |
|---|---|---|---|---|---|
| none | 0.045 | 0.068 | 0.495 | 0.514 | 0.511 |
| random oversample | 0.071 | 0.104 | 0.422 | **0.720** | **0.526** |
| SMOTE | 0.049 | 0.078 | 0.357 | 0.382 | 0.487 |
| ADASYN | **0.074** | 0.072 | 0.352 | 0.412 | 0.487 |
| flowmatch | 0.000 | 0.112 | 0.507 | 0.556 | 0.511 |
| flowmatch_pertype | 0.000 | **0.112** | **0.513** | 0.549 | 0.510 |

Test support: Analysis 677, Backdoor 583, Shellcode 378, Worms 44.

- **Flow matching beats both classical resamplers on 3 of 4 rare classes** —
  Shellcode +0.156, Worms +0.174, Backdoor +0.034 over SMOTE — and is the only
  arm that does not degrade Shellcode.
- **It still does not win outright.** Four classes, three different winners.
- `flowmatch_pertype` matches `flowmatch` within noise, exactly as predicted:
  UNSW-NB15 has no labels below `attack_cat`, so per-type degenerates to
  per-class. The control behaving as expected supports coverage being the real
  mechanism behind their divergence on NSL-KDD.

### 3.3 CICIDS2017 — stratified 70/30 split on the fine label (GPU)

Train 1,979,513 / test 848,363. BENIGN never downsampled; minority classes lifted
to 176,198 (see §1.0b). Augmented training sets are 2,999,508 rows.

**Bot F1** — the only class with meaningful headroom. WebAttack sits at
0.980 and BruteForce at 0.999 before any augmentation, so neither can move.

| arm | Bot F1 (clean seeds) | vs baseline |
|---|---|---|
| flowmatch_pertype | **0.827** | +0.005 |
| flowmatch | 0.825 | +0.003 |
| none | 0.822 | — |
| random oversample | 0.817 | −0.005 |
| SMOTE | 0.783 | **−0.039** |
| ADASYN | 0.780 | **−0.042** |

**Flow matching ties the unaugmented baseline.** It does not improve on doing
nothing, on the dataset offering the best conditions it has had. Both
interpolation methods degrade the rare class by ~0.04, tightly and consistently
(ADASYN across three seeds: 0.780, 0.779, 0.781).

The defensible statement here is not that flow matching wins, but that **the
standard resamplers actively hurt and flow matching does not.**

**Training instability, ~1 seed in 5.** Two distinct failure modes, both silent —
no exception, no warning, GPU neither idle nor out of memory:

| arm | seeds | note |
|---|---|---|
| none | 0.822 ×4, **macro-F1 0.0996** ×1 | total collapse: predicts only BENIGN |
| flowmatch | 0.831, **0.640**, 0.817, 0.824, 0.826 | partial collapse |
| flowmatch_pertype | 0.823, 0.833, 0.817, 0.833, **0.649** | partial collapse |

The `none` collapse reproduces exactly across three independent runs, so it is a
property of that fit, not noise. **A single-seed study would either miss this
entirely or publish the failed run as a result.** Detected by
`is_degenerate()` in `src/eval/metrics.py` and reported as a failure rate rather
than averaged in — including it drags the `none` arm from 0.944 to 0.775.

**macro-F1 is not trustworthy on this dataset.** It swung 0.9698 → 0.8351 between
two `flowmatch` seeds while Bot moved 0.002. The swing is Infiltration (11 test
rows) or Heartbleed (3) flipping between ~1.0 and 0. Per-class figures with
support reported are the only reliable readout.

### 3.4 The three datasets disagree

| | NSL-KDD | UNSW-NB15 | CICIDS2017 |
|---|---|---|---|
| SMOTE vs no augmentation | **improves** every rare class | **degrades** 2 of 4 | **degrades** Bot (−0.039) |
| best macro-F1 | SMOTE | random oversample | ADASYN (worst on Bot) |
| flow matching vs SMOTE | loses | wins 3 of 4 rare classes | wins (+0.042) |
| flow matching vs doing nothing | wins | wins 3 of 4 | **ties** |
| why rare classes fail | train/test attack types disjoint | classes overlap in feature space | they mostly don't fail |

**SMOTE helps on one benchmark and hurts on the other two, measured through an
identical pipeline with identical seeds.** Any claim of the form "method X helps
with class imbalance in IDS" drawn from a single benchmark is therefore
unsupported — and single-benchmark evaluation is the norm in this literature.

Note the last row. Across three datasets the rare-class failure has three
different causes, and **none of them is a shortage of training examples.** The
field's standard remedy addresses a cause that, on this evidence, is rarely the
operative one.

---

## 3b. Generator diagnostics (raw measurements)

Kept in full because several are needed to defend claims elsewhere, and because
re-running them costs hours.

### Flow-matching sample quality, NSL-KDD

Detection AUC is a random forest separating synthetic from real (0.5 =
indistinguishable). NN ratio is mean synthetic→real nearest-neighbour distance
divided by real→real; below 1 means samples sit closer to real points than real
points sit to each other.

| class | n_real | detection AUC | NN ratio | KS median | verdict |
|---|---|---|---|---|---|
| r2l | 995 | 0.9996 | 21.40 | 0.113 | FAIL |
| u2r | 52 | 0.9834 | 2.61 | 0.183 | FAIL |
| probe | 11,656 | 1.0000 | 26.68 | 0.119 | FAIL |

For comparison, on the same test:

| method | class | detection AUC | NN ratio | KS median |
|---|---|---|---|---|
| SMOTE | r2l | **0.5354** | **0.634** | 0.011 |
| SMOTE | u2r | 0.7934 | **0.524** | 0.058 |
| ADASYN | r2l | 0.9894 | 9.54 | 0.203 |
| ADASYN | u2r | 1.0000 | 1.21 | 0.103 |

**SMOTE's sub-1.0 NN ratio is structural**: its samples are interpolations
between two real points and therefore cannot leave the data manifold. That is the
most plausible explanation for why a 2002 method remains competitive against
learned generators.

### Per-type generation vs per-class, NSL-KDD

| class | metric | whole-class | per-type | coverage |
|---|---|---|---|---|
| r2l | detection AUC | 0.9996 | 0.9835 | 97% (3 of 8 types) |
| r2l | NN ratio | 21.40 | **7.14** | |
| r2l | KS median | 0.113 | **0.061** | |
| u2r | detection AUC | 0.9834 | 0.9642 | 58% (1 of 4 types) |
| u2r | NN ratio | 2.61 | **1.98** | |

Fitting per attack type improved every quality measure — confirming the
mixture-fitting diagnosis — but did not close the gap. Types below 20 training
rows are not modelled; on U2R only `buffer_overflow` clears that bar, leaving
42% of the class ungenerated.

### Training converges; the gap is structural, not undertraining

Flow-matching loss on NSL-KDD u2r, against an irreducible floor of ~1.31 (the
target `x1 - x0` is stochastic, so the model can only predict its conditional
mean):

    epoch    1   loss 1.266      epoch  600   loss 0.373
    epoch   50   loss 1.022      epoch  900   loss 0.377
    epoch  150   loss 0.633      epoch 1200   loss 0.347
    epoch  300   loss 0.446

Longer training improves distance and marginals substantially — on r2l, 300→4800
epochs moved NN ratio 23.0→12.8 and KS median 0.146→0.063 — **but detection AUC
stayed at 0.9996 throughout.** Samples get closer to the real distribution
without becoming any harder to identify. That rules out undertraining as the
explanation and supports the structural reading: a continuous flow cannot
concentrate mass onto the dense, near-duplicate clusters network traffic forms.

### GPU batch-size scaling (probe class, 50 epochs)

| batch | ms/epoch | speedup | peak VRAM |
|---|---|---|---|
| 256 | 680 | — | 39 MiB |
| 1,024 | 148 | 4.6× | 52 MiB |
| **4,096** | **47** | **14.5×** | 105 MiB |
| 11,656 (full) | 34 | 20× | 237 MiB |

CPU baseline ~1,300 ms/epoch, so batch 4,096 on GPU is ~28×. The initial
measurement of only 1.4× was batch size, not the GPU: at batch 256 a 512-wide MLP
spends almost all its time on kernel launches — 180 per epoch on the DoS class.

### Runtime, CICIDS2017 (1.98M train rows, ~3M augmented)

| arm | seconds/seed |
|---|---|
| none | 93 |
| random_oversample | 141–151 |
| smote | 245–325 |
| flowmatch (chunked) | 230–475 |
| flowmatch_pertype | 227–457 |
| **adasyn** | **2,408–4,779** |

Before chunked sampling, `flowmatch` took 20,898s / 962s / 4,287s for identical
cached-generator work.

---

## 4. Methodological corrections made to our own work

Recorded because each was caught by measurement rather than assumed.

| # | Problem | Consequence if missed | Fix |
|---|---|---|---|
| 1 | Deterministic classifier reported sd = 0.0000 across 5 seeds | Error bars measuring nothing; false confidence | `subsample`/`colsample_bytree` → 0.8 so the seed perturbs the model |
| 2 | Quality gate specified as a **blocking** checkpoint | Would have discarded ADASYN, the best-performing arm | Demoted to reported diagnostic (§1.1) |
| 3 | Base paper designated as Diff-IDS (Q2) | Weaker narrative; building on a competitor | Corrected to Alsubaei (Q1); Diff-IDS is a rival method |
| 4 | Discrete columns transported as continuous | Invalid records, detection AUC 1.0000 | Type inference + rounding (§1.4) |
| 5 | Categorical decode used `argmax` | Mode collapse: 2 of 7 `flag` values | Sample from the block instead |
| 6 | Memorisation check broadcast an n×n×d array | 59 GiB allocation, crash | `NearestNeighbors` index |
| 7 | Generators refitted per seed | Hours of duplicated training | Cache on (arm, class, generator seed, n) |
| 8 | `.gitignore` rule `data/` also matched `src/data/` | Source package silently untracked | Anchored to `/data/` |
| 9 | UNSW loader kept the CSV's own binary `label` column as a feature | Classifier handed the answer; a meaningless ~99% | Take target from `attack_cat`, drop binary label |
| 10 | UNSW CSV has a UTF-8 BOM | First column parses as `﻿id`, intended drop misses it | `encoding="utf-8-sig"` |
| 11 | CICIDS2017 `Web Attack` labels are cp1252-encoded | Three classes fail to map, fragment or raise | Normalise `U+FFFD` on load |
| 12 | `Flow Bytes/s` contains `Infinity` where duration is 0 | Passes pandas silently, breaks StandardScaler | Replace with NaN, drop (~0.1% rows) |
| 13 | `SMOTENC` hard-coded; CICIDS2017 is all-numeric | Run died 15 min in with a `ValueError` | Select plain `SMOTE` when no categoricals |
| 14 | Majority capped at 100k for tractability | Measured an artifact of our own choice (§1.0b) | Cap the oversampling target instead |
| 15 | Flow sampling built all ~176k rows in one tensor | 94% VRAM; identical work took 20,898s / 962s / 4,287s | Chunk at 32,768 rows, free cache between |
| 16 | Results written only after all 30 runs | A kill during a slow arm discarded every completed run | Flush both CSVs after every run |
| 17 | Ran a diagnostic alongside the main job on 8 GB RAM | Free RAM hit 1.8 GB; main run died at 5/30 | One heavy job at a time |

---

## 4b. Wrong turns and corrections to our own claims

Recorded because the reasoning matters as much as the result, and because each
was caught by measurement rather than argument.

**"The DiffIDS idea is novel."** It was not. ~25 searches found diffusion for IDS
imbalance occupied by ~10 papers (2023–2026), including one named Diff-IDS on the
same three datasets, plus TabPFN for IDS already taken. Cost: one search session.
Value: avoided months building a duplicate.

**"Diff-IDS is the base paper."** Wrong choice — it is a competing method with no
released code, in a Q2/Q3 venue. Corrected to Alsubaei (Sci. Reports, Q1), which
uses the same datasets and whose own results show tuning does not fix rare
classes. Caught only because the user asked which paper we were building on.

**"Augmentation cannot help NSL-KDD R2L at all."** Overstated. 89% of R2L training
data is `warezclient` with zero test samples, which is true — but the remaining
105 rows (`guess_passwd`, `warezmaster`) map to 75% of the R2L test set, and
SMOTE/ADASYN did improve R2L by ~0.05. The sharper and correct claim is that
R2L's *effective* training size is ~105 rows, not 995, so the real imbalance is
~1:641 rather than 1:68.

**"The quality gate should block bad generators."** Wrong, and following it would
have discarded ADASYN — the best R2L performer — since its synthetic data is
identified as fake 100% of the time. Gate demoted from blocking checkpoint to
reported diagnostic.

**"GPU will give flow matching 5–10×."** It gave 1.4×, because a 512-wide MLP at
batch 256 cannot saturate the device — peak VRAM was 39 MiB of 4,096. After
raising the batch to 4,096 the speedup was 14× and the original estimate was
recovered. The estimate was right; the configuration was wrong.

**"The Bot 0.640 outlier is a memory-saturation artifact."** Wrong. It fit the
timing data, so it was the convenient explanation — then `flowmatch_pertype` seed
4 produced 0.649 under normal memory conditions and fast runtime. The instability
is real and occurs in ~1 seed of 5 in both flow arms.

---

## 5. Verification work

- **~25 targeted literature searches.** Established that diffusion for IDS
  imbalance is occupied by ~10 papers (2023–2026), that TabPFN for IDS is taken,
  and that **flow matching for IDS class imbalance appears unoccupied**.
  Absence of search evidence is not proof — re-verify before submission.
- **Base paper reproduced** under published conditions. The rare-class collapse
  reproduces (our R2L recall 0.107 vs their 0.09); the 99 % accuracy does not,
  because they evaluate on a subset split rather than the official test split.
- **Dataset integrity confirmed** by row count: `KDDTrain+` 125,973,
  `KDDTest+` 22,544, `KDDTrain+_20Percent` 25,192 — all canonical.

---

## 5b. Practical limits found the hard way

Worth reporting: these are constraints anyone reproducing this work will hit, and
none of them appear in the papers that use these methods.

**ADASYN does not scale to 2M rows.** It fits a nearest-neighbour index over the
whole training set and queries it per minority sample; in 78 dimensions spatial
trees degrade to brute force. 150s per seed on UNSW's 175k rows, **2,400–4,800s
per seed on CICIDS2017's 1.98M** — and one aborted attempt exceeded 25 minutes
without finishing. Papers applying ADASYN to CICIDS2017 are either subsampling
heavily or not saying so.

**Flow matching saturates a 4 GB GPU on CICIDS2017.** Unchunked sampling held the
card at 94% and made runtimes meaningless: 20,898s, 962s and 4,287s for identical
cached-generator work. Chunking fixed it — the same work then took 475s and 230s.

**Windows Application Control blocks recent binary wheels.** pandas 3.0.5 and the
current pyarrow both fail with `DLL load failed ... blocked by Application
Control`. Pinned to pandas 2.2.3 and pyarrow 15.0.2.

**`pip install torch` yields a CPU build silently.** No error, no warning — flow
matching simply runs on CPU. Cost most of a session before it was noticed. The
CUDA build needs `--index-url https://download.pytorch.org/whl/cu124`, and pip
will not swap builds when version numbers match, so uninstall first. The 2.4 GB
download failed twice under pip (no resume); `curl -C -` at ~2 MB/s worked.

---

## 6. Honest status

**Working:** the pipeline, the measurement infrastructure, and all four
classical baselines.

**Not yet working:** flow matching produces samples a discriminator identifies
~98 % of the time — materially worse than SMOTE's 0.535. Per-attack-type
generation improved this substantially (R2L NN ratio 21.40 → 7.14, KS median
0.113 → 0.061) which confirmed the mixture-fitting diagnosis, but did not close
the gap.

**Unknown:** whether flow matching improves rare-class F1. Given §1.1, poor
fidelity does not rule it out — ADASYN scores worst on fidelity and best on R2L.
The six-arm comparison is running and will answer this.

**Coverage limit to state plainly:** per-type generation cannot model attack
types with too few samples. R2L covers 97 % of rows (3 of 8 types); **U2R covers
only 58 % (1 of 4 types)**.

---

## 7. What is not done

Roughly 40% of the experiments and 90% of the writing remain.

**Experiments**
- **CTGAN and TabDDPM arms — the largest gap.** The stated novelty is that
  diffusion has been applied ~10 times and flow matching has not. **Without a
  diffusion baseline that comparison cannot be made at all**, and a reviewer will
  ask for it first. 3 datasets × 2 arms × 5 seeds = 30 runs.
- Ablations: synthetic ratio (25/50/100%), ODE step count. Everything currently
  runs at full rebalancing and 50 steps with no justification beyond default.
- Significance testing (Wilcoxon signed-rank). Several reported differences are
  currently inside the noise and a reviewer can dismiss them.
- SHAP — not started.
- Re-run CICIDS2017 `flowmatch` seeds 0–2, which were measured under GPU memory
  saturation while seeds 3–4 were not. Mixing regimes within one arm is not
  defensible even though it does not change the conclusion.
- Diagnose the ~1-in-5 degenerate fits rather than only detecting them.

**Paper**
- Only the Introduction exists. Related Work, Method, Experiments, Results,
  Discussion, Conclusion unwritten.
- Bibliography verification — entries assembled from landing pages, not publisher
  exports; several fields marked `% CHECK`.
- Re-run the novelty search before submission. The field is moving: a directly
  relevant paper appeared 9 days before this project started.
- Repo cleanup and reproduction instructions.
- Cross-dataset transfer (deferred; see `PLAN.md` §9).

**Decision, not work:** the results no longer support a method paper. Flow
matching loses on NSL-KDD, wins 3 of 4 rare classes on UNSW, and ties the
baseline on CICIDS2017. The strongest material is now §1.0–§1.4 and §3.4. Which
paper gets written changes what is still needed — a method paper needs TabDDPM
urgently; an empirical paper needs the significance tests and less of the
baseline zoo. **That is a supervisor decision and should be taken before more
compute is spent.**

---

## 8. Where this stands relative to the plan

Roughly one week of the 16-week plan's Phase 0–2 work, completed in a single
session, plus four findings that were not in the plan. The main technical risk —
that the generative method might not work — surfaced in week 1 rather than
month 2, which is the outcome the gate was designed to produce.

The largest remaining lever is not technical: **securing a supervisor.**
`PLAN.md` §11 records the honest odds — roughly 25–35 % for Q1 on a first
attempt, ~90 % for publication somewhere given the work is finished and
submitted down the tiers.
