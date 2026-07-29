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

### 1.0a Fourteen of 63 augmentations are statistically significant *harms* ⭐

**Updated 2026-07-28 with all eight arms (CTGAN and diffusion added).** 63
comparisons across three datasets and nine rare classes; **14 are harms** and
**27 are indistinguishable from no effect**. Fewer than half do anything
demonstrably useful.

Harms now include every family: SMOTE, ADASYN, random oversampling, CTGAN,
diffusion, and both flow-matching variants. **No method is exempt.**

| dataset | class | arm | effect | d | Holm-p |
|---|---|---|---|---|---|
| UNSW | Shellcode | ADASYN | −0.1434 | −6.64 | 0.0007 |
| UNSW | Shellcode | **SMOTE** | **−0.1385** | **−9.59** | **0.0002** |
| UNSW | Worms | SMOTE | −0.1320 | −2.64 | 0.0246 |
| UNSW | Worms | ADASYN | −0.1020 | −1.70 | 0.0946 |
| UNSW | Shellcode | random oversample | −0.0738 | −2.89 | 0.0148 |
| CICIDS | WebAttack | CTGAN | −0.0586 | −0.66 | 1.0000 |
| CICIDS | WebAttack | diffusion | −0.0496 | −1.10 | 0.5762 |
| UNSW | Analysis | flowmatch | −0.0453 | −2.08 | 0.0679 |
| UNSW | Analysis | flowmatch_pertype | −0.0453 | −2.08 | 0.0679 |
| CICIDS | Bot | ADASYN | −0.0415 | −6.35 | 0.0067 |
| CICIDS | Bot | SMOTE | −0.0388 | −6.54 | 0.0067 |
| NSL-KDD | U2R | CTGAN | −0.0371 | −1.05 | 0.0778 |
| UNSW | Analysis | diffusion | −0.0363 | −1.73 | 0.0909 |
| UNSW | Backdoor | CTGAN | −0.0114 | −0.90 | 0.4548 |

*(Original 10-of-51 table retained below for the record.)*

### 1.0a-prev Ten of 51 augmentations are statistically significant *harms*

Paired tests against no augmentation, Holm-corrected within each (dataset, class)
family, degenerate fits excluded, bootstrap 95% CIs.

**The largest effect measured anywhere in this project is a harm.**

| dataset | class | arm | effect | Cohen's d | Holm-p |
|---|---|---|---|---|---|
| UNSW | Shellcode | **SMOTE** | **−0.1385** | **−9.59** | 0.0002 |
| UNSW | Shellcode | ADASYN | −0.1434 | −6.64 | 0.0006 |
| UNSW | Worms | SMOTE | −0.1320 | −2.64 | 0.0205 |
| UNSW | Worms | ADASYN | −0.1020 | −1.70 | 0.0757 |
| UNSW | Shellcode | random oversample | −0.0738 | −2.89 | 0.0118 |
| UNSW | Analysis | flowmatch | −0.0453 | −2.08 | 0.0582 |
| UNSW | Analysis | flowmatch_pertype | −0.0453 | −2.08 | 0.0582 |
| CICIDS | Bot | ADASYN | −0.0415 | −6.35 | 0.0048 |
| CICIDS | Bot | SMOTE | −0.0388 | −6.54 | 0.0048 |
| UNSW | Analysis | diffusion | −0.0363 | −1.73 | 0.0728 |

SMOTE damaging Shellcode at **d = −9.59** is a larger effect than any benefit
SMOTE produces on any dataset. These are not failures to help; they are
measurably worse than doing nothing, with corrected p-values and CIs excluding
zero.

**20 of 51 comparisons do not survive** — their 95% CI contains zero, so at five
seeds they are indistinguishable from no effect. Several differences discussed
earlier as real fall here, including flow matching's R2L gain on NSL-KDD
(Holm-p 0.0613).

#### A statistical limit that had to be worked around honestly

Wilcoxon signed-rank over n=5 pairs has 2⁵ = 32 sign assignments, so its smallest
attainable two-sided p-value is **0.0625**. **No comparison in this work can reach
p < 0.05 by Wilcoxon, regardless of effect size.** Reporting Wilcoxon alone —
the obvious choice, and what the plan originally specified — would have made
every result appear non-significant for a reason having nothing to do with the
data.

Reported instead: paired t-tests (Holm-corrected), Cohen's d, and bootstrap CIs,
with **"CI excludes zero" as the primary criterion** rather than any p-threshold.
The limitation is documented in `experiments/08_significance.py` rather than
worked around silently.

*Reproduce:* `experiments/08_significance.py`, `results/significance.csv`

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

### 3.2b Diffusion vs flow matching — the comparison the project is named for

Added 2026-07-28. Until this arm existed there was **no diffusion baseline at
all**, so the stated contribution — flow matching rather than diffusion — could
not be evaluated.

`TabularDiffusion` subclasses `TabularFlowMatcher` and overrides only `fit` and
`sample`. Identical encoder, decoder, discrete-column inference, log1p handling,
network architecture, width, and categorical decode. **The only difference is the
generative mechanism.** An off-the-shelf TabDDPM would have confounded the
comparison with preprocessing and tuning differences.

Sampling uses DDIM striding at 50 steps, matching flow matching exactly, so both
arms cost the same at generation. Full 1000-step ancestral sampling was not
tractable at CICIDS2017 scale and was not run — stated rather than implied.

**NSL-KDD** (5 seeds): diffusion wins R2L outright, second on U2R, above both
flow variants on each.

| arm | R2L F1 | U2R F1 |
|---|---|---|
| **diffusion** | **0.2893 ± .017** | 0.4070 ± .035 |
| flowmatch_pertype | 0.2634 ± .029 | 0.2784 ± .022 |
| ADASYN | 0.2505 ± .015 | 0.3314 ± .036 |
| SMOTE | 0.2462 ± .010 | **0.4279 ± .027** |
| flowmatch | 0.2382 ± .027 | 0.3869 ± .021 |
| none | 0.1923 ± .025 | 0.2193 ± .018 |

The mechanism is a precision/recall trade. Diffusion's R2L precision is **0.624**
against everyone else's 0.97–0.98, but recall is **0.190** against their
0.107–0.152. It flags far more R2L attacks and is wrong more often, and that
lands ahead on F1. **Whether that is preferable is a deployment question F1
cannot answer** — a system that raises three false alarms to catch one more
intrusion may or may not be wanted, and the paper should say so rather than let
F1 decide silently.

**UNSW-NB15** (5 seeds): mixed rather than dominant.

| class | best arm | diffusion | flowmatch | pertype |
|---|---|---|---|---|
| Analysis | ADASYN 0.074 | 0.009 | 0.000 | 0.000 |
| Backdoor | pertype 0.112 | 0.065 | 0.112 | **0.112** |
| Shellcode | **diffusion 0.517** | **0.517** | 0.507 | 0.513 |
| Worms | ROS 0.720 | 0.493 | **0.556** | 0.549 |

**Across both datasets diffusion beats flow matching on three rare classes and
loses on two.** The project's stated contribution is not supported.

#### CTGAN — the GAN family

Added 2026-07-28, completing the comparison across GAN / diffusion / flow rather
than two closely related continuous-time methods.

**Not architecture-matched, deliberately.** Diffusion subclasses the flow matcher
so the only difference is the generative mechanism. CTGAN is the published
implementation (Xu et al., NeurIPS 2019) with its own preprocessing, conditional
sampler and PacGAN discriminator — it is the baseline reviewers expect cited, and
a hand-rolled GAN would invite the objection that we did not use the real thing.
The confound is that a CTGAN-vs-flow difference could come from preprocessing
rather than the GAN objective, and any claim drawn from it must say so.

**NSL-KDD, 5 seeds — CTGAN is the weakest generative arm and harms U2R:**

| arm | R2L F1 | U2R F1 |
|---|---|---|
| diffusion | **0.2893** | 0.4070 |
| flowmatch_pertype | 0.2634 | 0.2784 |
| ADASYN | 0.2505 | 0.3314 |
| SMOTE | 0.2462 | **0.4279** |
| flowmatch | 0.2382 | 0.3869 |
| **CTGAN** | 0.2107 | **0.1821** ↓ |
| none | 0.1923 | 0.2193 |

CTGAN's U2R falls **below the unaugmented baseline** (0.182 vs 0.219).

**Speed.** CTGAN's default batch of 500 left the GPU at 23% and the CPU at 0.9
cores. Raising it to 4,000 gave 3.1× (219s → 71s per 10 epochs on probe). Larger
batches are faster still — full-batch ran in 62s — but only because they perform
one gradient update per epoch, which is not speed but absence of training. 4,000
was chosen as the largest batch that keeps a sane update count: **weakening a
baseline to save compute would reproduce exactly the flaw this project
criticises.**

#### The two generator families fail on opposite kinds of class

**Shellcode — the cleanest family separation in this work:**

    diffusion 0.517 > pertype 0.513 > flowmatch 0.507 > none 0.495
       >> ROS 0.422 > SMOTE 0.357 > ADASYN 0.352

Every generative method beats doing nothing; every interpolation method loses to
it, by up to 0.14. Well supported — 1,133 training rows, 378 test rows.

**Analysis — the mirror image, but the split is not generative-vs-interpolation:**

| arm | Analysis F1 | Shellcode F1 |
|---|---|---|
| ADASYN | **0.0744** | 0.3520 |
| random oversample | 0.0707 | 0.4216 |
| **CTGAN** | **0.0513** | **0.5402** |
| SMOTE | 0.0488 | 0.3568 |
| none | 0.0453 | 0.4954 |
| diffusion | 0.0090 | 0.5172 |
| flowmatch | **0.0000** | 0.5068 |
| flowmatch_pertype | **0.0000** | 0.5127 |

**This corrects an earlier claim in this file.** It previously read "all three
generative methods collapse to ~0 on Analysis", attributing the failure to
learned generators spreading mass into class overlap. **CTGAN does not collapse**
— it reaches 0.0513, in line with the classical methods and above the
unaugmented baseline, while also taking Shellcode outright at 0.5402.

The failure is therefore specific to the **continuous-time** generators.
Diffusion and flow matching both transport a Gaussian to the data distribution
through a smooth field, and both collapse on the overlapping class. CTGAN's
mode-specific normalisation and conditional sampler evidently preserve the
minority mode where continuous transport does not.

**Revised mechanism:** *continuous-time generative models fail on classes that
overlap another class in feature space; adversarial and interpolation methods do
not.* Sharper than the original claim, and still predictive from a property
measurable before training — but it now separates the generative family rather
than lumping it together, which the CTGAN data forced.

### 3.3b All 90 runs: best arm per rare class

Three datasets, six arms, five seeds, one pipeline. Nine rare classes with usable
test support.

| dataset | class | best arm | F1 | vs no augmentation |
|---|---|---|---|---|
| NSL-KDD | R2L | flowmatch_pertype | 0.2634 | +0.071 |
| NSL-KDD | U2R | **SMOTE** | 0.4279 | +0.209 |
| UNSW | Analysis | **ADASYN** | 0.0744 | +0.029 |
| UNSW | Backdoor | flowmatch_pertype | 0.1123 | +0.044 |
| UNSW | Shellcode | flowmatch_pertype | 0.5127 | +0.017 |
| UNSW | Worms | **random oversample** | 0.7196 | +0.205 |
| CICIDS | **Bot** | **none** | **0.8218** | **+0.000** |
| CICIDS | WebAttack | random oversample | 0.9920 | +0.008 |
| CICIDS | BruteForce | random oversample | 0.9997 | +0.000 |

**Five different methods win across nine classes.** Plain duplication — the
crudest possible approach — wins three. On CICIDS2017's only class with real
headroom, **doing nothing wins outright**.

CICIDS2017 Bot in full, since it is the cleanest test in the project (1,369
training rows, 587 test rows, 1:1,161 imbalance):

| arm | Bot F1 | sd |
|---|---|---|
| **none** | **0.8218** | 0.0072 |
| random oversample | 0.8167 | 0.0060 |
| flowmatch_pertype | 0.7910 | **0.0796** |
| flowmatch | 0.7876 | **0.0828** |
| SMOTE | 0.7829 | 0.0016 |
| ADASYN | 0.7806 | 0.0011 |

Every augmentation method underperforms the unaugmented baseline. The two flow
arms carry standard deviations ~50× ADASYN's — the 1-in-5 instability surfacing
as a headline number.

**The degenerate-fit detector mattered.** It flagged `none` seed 4 automatically
and reported "20% of seeds failed to train" rather than folding it into the mean.
Averaged in, the `none` arm reads macro-F1 0.775 instead of 0.943 — which would
have made doing nothing look far worse than it is and **quietly reversed the main
conclusion of the table above.**

### 3.3c Generative methods have a higher ceiling and a 20% failure rate

**This corrects §3.3.** That section reported that every augmentation method
underperforms the unaugmented baseline on CICIDS2017 Bot. That was an artifact of
our own analysis, not a property of the methods.

The degenerate filter excludes *total* collapses — macro-F1 below ~1/n_classes.
The baseline's seed 4 qualifies (0.0996) and was removed. But generative arms
fail *partially*: Bot drops to ~0.62 while macro-F1 stays near 0.85, so those
seeds slipped through and stayed in the mean. **The baseline had its worst seed
removed and the generative arms did not**, which biased the comparison toward
doing nothing.

Reporting the median alongside the mean fixes it without any further decisions
about which runs to discard:

| arm | Bot mean | **Bot median** | gap |
|---|---|---|---|
| **diffusion** | 0.7886 | **0.8303** | −0.042 |
| flowmatch | 0.7876 | **0.8243** | −0.037 |
| flowmatch_pertype | 0.7910 | **0.8233** | −0.032 |
| none | 0.8218 | 0.8228 | −0.001 |
| random oversample | 0.8167 | 0.8174 | −0.001 |
| SMOTE | 0.7829 | 0.7830 | −0.000 |
| ADASYN | 0.7806 | 0.7806 | 0.000 |

**By median all three generative methods beat the baseline and diffusion is
best. By mean all three lose to it.** Both are true and they answer different
questions:

- **median** — what you get when it works
- **mean** — what you get including failures

A deployment decision needs both. Reporting either alone misleads, in opposite
directions.

**The mean-minus-median gap is the cleanest instability metric in this work:**
~0.000 for every classical arm, −0.032 to −0.042 for every generative one. That
single column separates the two families.

**Failure rates on CICIDS2017 Bot** (seed F1 below 85% of that arm's median):

| family | arms with collapses |
|---|---|
| generative | **3 of 3** — diffusion, flowmatch, pertype, one seed each (20%) |
| classical | **0 of 4** — none, ROS, SMOTE, ADASYN |

Perfect separation. Every learned generator fails one run in five; no
interpolation method fails at all.

*Caveat on the detector:* an 85%-of-median threshold is only informative where F1
is high. On UNSW's Analysis and Backdoor (F1 0.01–0.11) it fires for nearly every
arm including `none`, because small absolute differences are large relative ones.
Read it on Bot and Shellcode, not on classes that are already near zero.

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
| CTGAN | r2l | 1.0000 | 11.32 | **0.413** |
| diffusion | r2l | 0.9992 | 43.24 | 0.306 |

**The three fidelity metrics disagree about which generator is best.** On r2l,
CTGAN has the closest samples in Euclidean terms (NN ratio 11.3) but the worst
marginals (KS 0.413) and is perfectly separable; diffusion has the worst NN ratio
(43.2) but better marginals than CTGAN; flow matching has the best marginals
(0.146) and middling distance.

**"Sample quality" is therefore not one property**, and any paper reporting a
single fidelity number to argue its generator is better has chosen which metric
to report. Combined with §1.1 — that fidelity does not predict utility at all —
the case for selecting a generator on sample-quality grounds is weak on two
independent counts.

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

## 3b-ii. Diagnosed: collapses follow the synthetic batch, and no quality metric sees them ⭐

CICIDS2017 Bot, flow matching. Five generator seeds, each evaluated with **two
classifier seeds on identical synthetic data** — the test that separates a bad
generated batch from an unlucky classifier fit.

| gen seed | clf seed | F1 | recall | PR-AUC | synth NN dist | leak→BENIGN |
|---|---|---|---|---|---|---|
| 3 | 3 | **0.6016** | 0.4617 | 0.7724 | 20.961 | **0.54** |
| 3 | 103 | **0.6043** | 0.4736 | 0.7843 | 20.961 | **0.53** |
| 0,1,2,4 | both | 0.82–0.84 | 0.74–0.94 | 0.934–0.941 | 20.76–21.37 | 0.06–0.26 |

**Both classifier fits collapsed on generator seed 3; no other seed collapsed
under either fit.** The failure follows the synthetic sample, so the fix has to be
in the generator, not the classifier.

| | collapsed | healthy | what it rules out |
|---|---|---|---|
| PR-AUC | 0.778 | 0.938 | not a threshold/argmax artifact — ranking degrades |
| recall | 0.468 | 0.809 | recall halves |
| precision | 0.849 | 0.861 | **unchanged** — not false alarms, missed attacks |
| leak → BENIGN | 0.54 | 0.06–0.26 | missed Bot absorbed into benign traffic |
| **synthetic NN distance** | **20.961** | **21.069** | **detects nothing** |

**The quality metric cannot see the bad batch.** Nearest-neighbour distance —
the standard pre-training check on synthetic data — is 20.961 for the batch that
halved recall and 21.069 for the batches that worked. A 0.5% difference, in the
wrong direction.

**Consequence.** "Validate your synthetic data before using it" is standard
advice and, on this evidence, does not work: the batch that destroys performance
is indistinguishable beforehand from the batches that do not. This is the fifth
independent fidelity measure that fails to predict utility (§1.1, §3b, §3c) and
the most practically damaging, because it removes the obvious mitigation for
generative instability.

*Reproduce:* `experiments/11_collapse_diagnosis.py`,
`results/collapse_diagnosis.csv`

## 3c. SHAP: augmentation rewrites the decision rule

NSL-KDD, TreeSHAP on XGBoost, 2,000 test rows. L1 distance between normalised
mean-|SHAP| profiles against the unaugmented model — 0 means identical reliance,
2 means no overlap.

| arm | R2L L1 | R2L top-5 overlap | R2L F1 | U2R L1 | U2R F1 |
|---|---|---|---|---|---|
| SMOTE | **0.326** | 4/5 | 0.246 | 0.541 | **0.428** |
| ADASYN | 0.462 | 3/5 | 0.251 | 0.699 | 0.331 |
| flowmatch_pertype | 0.609 | 3/5 | 0.263 | 0.547 | 0.278 |
| flowmatch | 0.686 | **2/5** | 0.238 | 0.633 | 0.387 |
| **diffusion** | **0.827** | 3/5 | **0.289** | **0.783** | 0.407 |

**Augmentation substantially rewrites what the classifier attends to.** L1 reaches
0.83 of a possible 2.0, and top-5 feature overlap falls to **2 of 5**. Synthetic
data is not "more of the same": models trained with it use different features.

Concretely, on R2L the unaugmented model leads on `src_bytes` and `count`;
diffusion promotes `hot` and `protocol_type_tcp`; flow matching promotes
`dst_host_same_src_port_rate` to first. On U2R, `service_telnet` is a top-5
feature for none / SMOTE / both flow variants but **not** for diffusion or ADASYN.

**The framing this analysis was built on turned out to be backwards.** It assumed
faithful synthetic data should *preserve* the decision rule, making divergence a
warning sign. But the baseline's R2L rule is bad — 0.107 recall — so preserving
it is not a virtue. On R2L the arm that diverges most (diffusion) also scores
best, and the arm that stays closest (SMOTE) scores near the bottom.

**Does divergence predict performance? No, not detectably.**

| class | Pearson r | p | Spearman | p |
|---|---|---|---|---|
| R2L | +0.622 | 0.263 | +0.400 | 0.505 |
| U2R | +0.226 | 0.715 | −0.100 | 0.873 |

Five arms gives no power; these are descriptive, not evidence. The suggestive
positive trend on R2L does not appear on U2R at all.

**This is the fourth independent measure that fails to predict utility**, after
detection AUC, nearest-neighbour distance, and marginal KS (§1.1, §3b). Sample
realism, distributional closeness, marginal fidelity, and now decision-rule
similarity — none of them tells you whether a generator will help.

*Reproduce:* `experiments/10_shap.py`, `results/shap_divergence.csv`,
`results/shap_profiles.csv`

## 3d. Ablations: both defaults are wrong

NSL-KDD, 3 seeds. A sensitivity sweep, not a significance test.

### Synthetic ratio — full rebalancing buys nothing

| ratio | train rows | flowmatch R2L | diffusion R2L | SMOTE R2L |
|---|---|---|---|---|
| 0.25 | 163,775 | 0.3046 | 0.2476 | 0.2624 |
| 0.50 | 214,283 | 0.3070 | 0.2750 | 0.2383 |
| 1.00 | 336,715 | 0.3032 | 0.2660 | 0.2586 |

Flow matching is flat to within 0.004 across a **fourfold** change in synthetic
volume. Quarter-rebalancing matches full rebalancing on half the training rows.

**Full parity to the majority class is the near-universal default in this
literature.** It doubles the training set, makes NSL-KDD's U2R 99.92% synthetic,
and on this evidence produces no benefit.

### Integration steps — fewer is better, not merely sufficient

| steps | R2L F1 | U2R F1 | seconds |
|---|---|---|---|
| **10** | **0.3179** | **0.4517** | 90 |
| 50 | 0.3018 | 0.4385 | 87 |
| 100 | 0.3028 | 0.4479 | 131 |

Ten steps is best on both rare classes; 100 steps costs 45% more time for
nothing. At three seeds a 0.016 difference is not significant, so the defensible
claim is that **performance is flat or slightly better at 10 steps and more
integration does not help.**

This strengthens flow matching's practical case. Its advantage over diffusion is
sampling cost, and 10 steps is **100× fewer network evaluations than standard
1000-step DDPM** rather than the 20× implied by our default of 50.

*Both defaults were chosen because they are common, not because they were tested.
Both turn out to be wrong in the same direction — more is not better.*

---

## 3e. Class overlap predicts difficulty, not what augmentation will do about it

**A prediction that failed, recorded because it failed.** The comparison threw up
a pattern nobody went looking for: generative augmentation helps some rare
classes and does nothing for others, and the split is not by dataset, not by
class size, and not by generator fit quality.

| | train rows | every arm's best F1 |
|---|---|---|
| UNSW-NB15 Analysis | 2,000 | 0.047 → 0.073 |
| UNSW-NB15 Shellcode | 1,133 | 0.495 → 0.538 |
| CICIDS2017 Bot | 1,369 | 0.822 → 0.830 |

Analysis has **more** training data than Shellcode and still cannot be learned.
The hypothesis was geometric: Analysis rows sit inside the region occupied by
other classes, so no synthetic data drawn from them can create a boundary that
does not exist. If true, an overlap statistic computed on real training data
**before any generator is fitted** would tell a practitioner in advance whether
augmenting a class is worth the GPU hours.

Three standard complexity measures (Lorena et al. 2019; Oh 2011), computed on the
encoded training split with exact blocked k-NN on GPU — 5,000 queries against
CICIDS2017's full 1.98M rows, not a subsample, because subsampling bias depends
on class size and class size is the variable under test.

| measure | vs baseline F1 | vs best gain | vs generative advantage |
|---|---|---|---|
| N3 (1-NN error) | **−0.770** (p=0.0001) | +0.568 (p=0.011) | −0.150 (p=0.54) |
| R-value (k=10) | **−0.770** (p=0.0001) | +0.637 (p=0.003) | −0.154 (p=0.53) |
| nearest-enemy ratio | +0.416 (p=0.077) | −0.312 (p=0.19) | −0.254 (p=0.29) |

Spearman rho over 19 augmented classes; 3 majority classes and 2 below the
test-support threshold excluded before testing.

**The middle column does not survive contact with the confound.** Overlap drives
the baseline down, and a low baseline mechanically leaves more room to gain.
Controlling for baseline F1 by partial correlation:

| | raw rho | controlling for baseline F1 |
|---|---|---|
| N3 vs best gain | +0.568 | **−0.105** |
| R-value vs best gain | +0.637 | **+0.070** |

**Three conclusions, one of them the opposite of what was predicted.**

1. Overlap is an excellent predictor of *difficulty* — rho −0.77 at p=0.0001. That
   is a sanity check passing, not a result.
2. Overlap carries **no information about how much augmentation recovers** once
   difficulty is divided out. The intuitive rule "spend your compute on the
   overlapping classes" is not supported.
3. Overlap does **not** predict which family to reach for (rho −0.15, p=0.54).
   Shellcode (N3 0.654) is where generative methods win by the largest margin
   (+0.114); Worms (N3 0.769) is where they lose by the largest (−0.159). Both
   are heavily overlapped. What separates them is training rows — 1,133 versus
   130 — not geometry.

**This is the third cheap predictor of utility to fail.** The quality gate
(§1.1), SHAP divergence (§3c) and now class geometry all correlate with something
plausible and none of them predicts whether augmentation will help. That
convergence is a stronger and more coherent claim than the positive mechanism
originally expected, and it is a claim about measurement practice rather than
about any one generator.

**Stated limit:** 19 classes, of which 13 have enough headroom for `recovery` to
be defined. A null at that sample size is weak evidence of absence, not proof.

*A metric bug was caught here and is worth recording. The nearest-enemy ratio
first came back at values around 10⁸. These datasets contain large numbers of
exact duplicate rows, so the distance to the nearest same-class row is frequently
exactly zero, and a per-row ratio diverges. Replaced with a ratio of means and
the duplicate fraction reported alongside. The measure still says the least of
the three, because for a well-separated class almost no query has any
out-of-class row among its ten neighbours, so it rests on a handful of boundary
points.* — `experiments/13_class_overlap.py`, `results/class_overlap.csv`

---

## 3f. The conclusions are classifier-dependent ⭐

*Complete: 120 runs, 8 arms × 5 seeds × 3 datasets.*

Every number in this document up to §3e came from XGBoost. That leaves one
sentence able to dismiss the lot: gradient-boosted trees are piecewise-constant
and scale-invariant, so a synthetic row landing inside an existing leaf changes
nothing for them. "SMOTE and flow matching are indistinguishable" might be a fact
about trees rather than about synthetic data.

An MLP was built as the contrast — same splits, same seeds, same augmentation,
same preprocessing, same untuned-and-unweighted treatment, **only the model family
changed**. The answer is not the reassuring one.

**Across all nine rare classes, the two classifiers never once agree on the best
arm.**

| dataset | class | rank rho | XGBoost's best | MLP's best |
|---|---|---|---|---|
| NSL-KDD | r2l | −0.119 | diffusion | ctgan |
| NSL-KDD | u2r | −0.452 | smote | ctgan |
| UNSW-NB15 | Analysis | +0.686 | adasyn | random_oversample |
| UNSW-NB15 | Backdoor | +0.429 | flowmatch_pertype | random_oversample |
| UNSW-NB15 | Shellcode | +0.611 | ctgan | **none** |
| UNSW-NB15 | Worms | +0.240 | random_oversample | **none** |
| CICIDS2017 | Bot | +0.381 | diffusion | ctgan |
| CICIDS2017 | WebAttack | +0.467 | random_oversample | smote |
| CICIDS2017 | BruteForce | −0.334 | random_oversample | flowmatch_pertype |

Mean Spearman rho +0.212, **not significant for any class**; **0 of 9 share a best
arm**. On two UNSW classes the MLP's best option is no augmentation at all.

**Sign agreement is 27 of 63 (43%)** — whether an arm helps or harms is worse than
a coin flip between classifiers.

**This is not a weak model disagreeing with a strong one.** The MLP *beats*
XGBoost on NSL-KDD and loses on the other two:

| best macro-F1 over all arms | XGBoost | MLP |
|---|---|---|
| NSL-KDD | 0.6373 | **0.6832** |
| UNSW-NB15 | **0.5261** | 0.4335 |
| CICIDS2017 | **0.9731** | 0.7909 |

It also **never collapsed**: 0 degenerate fits in 120 runs, against the ~1-in-5
rate XGBoost shows on CICIDS2017 generative arms (§3.3c). Whatever is driving the
disagreement, "the network could not train" is not it.

**The disagreement is directional, not noise.** Every generative arm that helps
XGBoost harms the MLP:

| UNSW-NB15 | ctgan | diffusion | flowmatch | pertype |
|---|---|---|---|---|
| Shellcode, XGBoost | +0.044 | +0.022 | +0.003 | +0.016 |
| Shellcode, MLP | −0.007 | −0.119 | −0.082 | −0.082 |
| Worms, XGBoost | +0.060 | −0.007 | +0.060 | +0.026 |
| Worms, MLP | −0.076 | −0.073 | −0.095 | −0.084 |

Seven of eight cells flip sign, all in the same direction. This is consistent with
the mechanism the experiment was designed around: synthetic rows that fall inside
an existing leaf merely reweight it for a tree, while for a network they move a
decision boundary it then has to fit.

**CICIDS2017 adds a second pattern the tree hides entirely.** Under the MLP, the
interpolation methods do violent and *opposite* things to two classes at once:

| CICIDS2017, MLP (median over seeds) | Bot | WebAttack | BruteForce | macro-F1 |
|---|---|---|---|---|
| none | 0.559 | 0.142 | 0.982 | 0.646 |
| smote | **0.235** | **0.373** | 0.964 | 0.762 |
| random_oversample | **0.212** | **0.352** | 0.975 | 0.718 |
| adasyn | **0.208** | **0.296** | **0.783** | 0.714 |
| flowmatch | 0.548 | 0.196 | 0.984 | 0.769 |
| diffusion | 0.549 | 0.193 | 0.985 | 0.776 |

SMOTE cuts Bot by more than half and simultaneously more than doubles WebAttack.
ADASYN does the same and additionally destroys BruteForce, a class every other arm
leaves at 0.98. The generative arms move neither. **Macro-F1 conceals all of it** —
every augmented arm looks like a clean improvement over the 0.646 baseline, which
is exactly the summary statistic this literature reports.

Under XGBoost these same CICIDS effects are near-invisible: ADASYN moves BruteForce
by +0.0001 and WebAttack by −0.0014. The tree is simply not sensitive to what the
interpolation is doing to the geometry.

**What survives both classifiers is almost entirely the classical methods.** Of 36
effects with a bootstrap CI excluding zero under XGBoost and 47 under the MLP,
**16 are found under both — and 14 of those 16 involve SMOTE, ADASYN or random
oversampling.**

Only two robust effects involve a generative method at all: `flowmatch_pertype`
helping NSL-KDD R2L, and CTGAN *harming* UNSW Backdoor. Split by direction, the 16
are 8 helps and 8 harms.

**Stated plainly: after 240 runs across two classifiers, generative augmentation
produces one reproducible improvement.** Interpolation methods produce fourteen
reproducible effects, half of which are damage. §1.0a's central negative result is
the part robust to the classifier; the positive results are not.

### The gap between the two protocols is the finding — not the second number alone

Reporting only "one reproducible improvement" understates what happened, and an
earlier draft of this section did exactly that. **Under XGBoost alone — the
protocol this field actually uses — generative augmentation produces 11
statistically significant improvements**, six of them surviving Holm-Bonferroni
correction:

| dataset | class | arm | gain | Holm p |
|---|---|---|---|---|
| NSL-KDD | u2r | diffusion | **+0.1878** | 0.0021 |
| NSL-KDD | u2r | flowmatch | **+0.1677** | 0.0004 |
| NSL-KDD | r2l | diffusion | **+0.0971** | 0.0037 |
| NSL-KDD | r2l | flowmatch_pertype | +0.0712 | 0.0219 |
| NSL-KDD | u2r | flowmatch_pertype | +0.0591 | 0.0021 |
| UNSW-NB15 | Shellcode | ctgan | +0.0449 | 0.0217 |
| UNSW-NB15 | Backdoor | flowmatch_pertype | +0.0442 | 0.1328 |
| UNSW-NB15 | Backdoor | flowmatch | +0.0440 | 0.1038 |
| UNSW-NB15 | Worms | flowmatch | +0.0419 | 0.4629 |
| UNSW-NB15 | Shellcode | diffusion | +0.0218 | 0.2267 |
| NSL-KDD | r2l | flowmatch | +0.0459 | 0.0919 |

Against 7 significant harms. Generative arms also beat the best classical arm
outright on **4 of 9** rare classes, Shellcode by +0.114.

**These are real, correctly computed, and would be publishable on their own terms.**
A +0.19 F1 gain at Holm-corrected p=0.002 is not noise, and nothing in this project
shows the arithmetic is wrong.

What the second classifier shows is that **1 of the 11 survives**. The contribution
is therefore not "generative augmentation fails" — it is:

> Under the standard single-classifier protocol we obtain 11 significant
> improvements, 6 surviving multiple-comparison correction, with gains to +0.19 F1.
> Changing only the classifier leaves 1.

Both halves are needed. The first alone is the paper this field keeps publishing;
the second alone reads as a failed project and discards the evidence that makes
the first one interesting.

### Ruled out: this is not the scaler

The obvious objection had to be eliminated first. The pipeline fits
`StandardScaler` on the **augmented** training set, as the usual recipe does. If a
generator emits outliers the fitted mean and variance shift, and every *real* row
is re-encoded through a distorted transform — which a tree would not notice and a
network would. That would make the whole finding a pipeline artefact.

**The distortion is real and large:**

| arm | max mean shift (real sd) | max sd ratio | worst column |
|---|---|---|---|
| smote | 0.386 | 2.55 | `sttl` |
| flowmatch | 1.220 | 4.90 | `ct_flw_http_mthd` |
| diffusion | **1.901** | **5.77** | `ct_flw_http_mthd` |

Diffusion moves one column's fitted mean by nearly two standard deviations of the
real data and inflates its scale almost sixfold. **And it is not the cause.**
Re-fitting the scaler on real rows only (regime B), everything else identical:

| macro-F1 delta vs none | regime A (augmented) | regime B (real only) |
|---|---|---|
| smote | −0.0123 | −0.0115 |
| diffusion | −0.0311 | −0.0268 |
| flowmatch | −0.0238 | −0.0213 |

Correcting the scaler recovers at most 0.004 of a 0.031 loss — about 13%, and on
Worms flow matching gets *worse* under B (−0.072 → −0.111). **The harm is the
synthetic rows themselves entering the loss, not the encoding of the real ones.**

*Recorded as a distinct result regardless: generative augmentation measurably
corrupts a fitted standard scaler, by up to 5.8× on a real column. It happens to
not be what is driving the F1 loss here, but anyone pairing these generators with
a scale-sensitive model is carrying it unknowingly.*

**Why this matters more than the result it complicates.** Every paper in this area
fixes one classifier and reports the arm ranking as though it were a property of
the augmentation method. On this evidence it is not. A recommendation of the form
"use method X for rare-class IDS" is incomplete without naming the classifier, and
the field states it unconditionally.

### Limits of this result, stated before a reviewer states them

- **Two classifiers is two points.** "Rankings are classifier-dependent" is
  established; *which* classifier to prefer is not, and cannot be from n=2. A
  third family (linear, or a tabular transformer) would say whether trees or
  networks are the outlier.
- **The MLP is untuned by design.** Fixing one architecture across all arms is
  what makes the arms comparable, but it means the MLP's absolute numbers are a
  floor rather than the best a network can do. Its lower macro-F1 on UNSW-NB15 and
  CICIDS2017 should not be read as a claim about neural networks on tabular data.
- **Five seeds.** Same Wilcoxon floor as everywhere else in this project (§1.0a);
  the bootstrap CI carries the argument, not the p-values.
- **The mechanism is inferred, not proven.** The leaf-versus-boundary explanation
  fits the direction of every flip and is supported by the scaler test ruling out
  the alternative, but it has not been demonstrated directly.

— `experiments/12,14,15`, `results/mlp_*.csv`, `results/scaler_confound.csv`

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

*Profiled properly when it became the largest single line item in the second
classifier sweep. The cost is not where it looks: the expensive queries come from
the **large** minority classes, not the rare ones. imblearn queries the index with
every row of each class it augments, so PortScan sends 111,163 queries and DDoS
89,617, while Bot sends 1,369. Measured at CICIDS2017 scale, 1.98M references ×
78 dimensions:*

| configuration | 20k queries | projected per ADASYN pass |
|---|---|---|
| `auto`, 1 job (imblearn default) | 269s | 48 min |
| `brute`, all 8 cores | 243s | 43 min |
| exact blocked search on GPU | 44s | 8 min |

*`n_jobs=-1` buys 1.1×, not the 8× the core count suggests, because sklearn's
brute-force path is already threaded through BLAS — the job-level knob is close to
a no-op. **The GPU port was measured, verified to return identical neighbours, and
then deliberately not used:** it computes in float32 while the already-completed
XGBoost ADASYN runs used float64 on CPU, and swapping precision mid-comparison
would buy 2.8 hours by making the two classifiers' ADASYN arms non-comparable.
The point of the sweep is comparability.*

**A 60-epoch cap on the MLP is not binding, which had to be checked rather than
assumed.** Early stopping never fires on NSL-KDD (max 53 of 60) but hits the cap
in 13 of 20 UNSW runs, which looks like an under-trained model on one of three
datasets. Re-tested at cap 200:

| arm | cap 60 | cap 200 | macro-F1 |
|---|---|---|---|
| `none` | ran 60 | ran 65 | 0.4327 both |
| `random_oversample` | ran 56 | ran 56 | 0.4251 both |

*Identical to four decimal places, including every rare-class F1. `fit` restores
the best-validation checkpoint rather than the final epoch, so hitting the cap
does not mean the reported model is the epoch-60 model — the extra epochs only
confirm the earlier checkpoint was already the best. No re-run needed. Recorded
because "the budget ran out" and "the model converged" produce the same log line
and only one of them is a problem.*

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

*Rewritten 2026-07-28. The previous version of this section described the six-arm
comparison as "running" and listed CTGAN, ablations, significance testing and
SHAP as not started; all had been finished for some time. Left stale sections are
worse than absent ones, so this and §7 are now dated.*

**Working:** the pipeline, the measurement infrastructure, all four classical
baselines, all four generative arms, and the full comparison at 8 arms × 5 seeds
× 3 datasets under XGBoost.

**Settled:** flow matching produces samples a discriminator identifies ~98 % of
the time, materially worse than SMOTE's 0.535 — and it does not matter. Fidelity
does not predict utility (§1.1), the quality gate does not predict utility
(§1.1), SHAP divergence does not predict utility (§3c), and class geometry does
not predict utility (§3e). Four independent cheap predictors, all plausible, none
of them informative about whether augmentation will help.

**Settled and negative:** no generative arm beats the best interpolation arm by a
margin that survives a bootstrap CI on any rare class. 14 of 63 augmentations are
statistically significant *harms* (§1.0a). This is not the result the project set
out to produce and it is the result it has.

**Settled, and it reframes the paper:** none of the above is a property of
augmentation alone. Re-run with an MLP, the two classifiers agree on the best arm
for **0 of 9** rare classes, agree on help-vs-harm 43% of the time, and share only
16 of 83 statistically solid effects — 14 of those 16 belonging to SMOTE, ADASYN
or random oversampling. **Across 240 runs and two classifiers, generative
augmentation produces exactly one reproducible improvement.** See §3f.

**Coverage limit to state plainly:** per-type generation cannot model attack
types with too few samples. R2L covers 97 % of rows (3 of 8 types); **U2R covers
only 58 % (1 of 4 types)**.

---

## 7. What is not done

*Dated 2026-07-28.* The experimental content is close to complete; the writing is
not.

**Experiments — remaining**
- A **third** classifier family. Two points establish that rankings are
  classifier-dependent but cannot say which classifier is the outlier (§3f).
- More than 5 seeds. The Wilcoxon signed-rank floor at n=5 is 0.0625, so no rank
  test in this project can reach p<0.05 whatever the effect size (§1.0a). 20 seeds
  would remove that ceiling. Optional: the bootstrap CI already carries the
  argument.
- Cross-dataset transfer — train the generator on one dataset, test on another.
  Deferred deliberately (`PLAN.md` §9); it is a different paper.

**Experiments — done since this section was last accurate**
CTGAN on all three datasets; diffusion as an architecture-matched arm; the ratio
and integration-step ablations (§3d); paired significance testing with
Holm-Bonferroni and bootstrap CIs (§1.0a); SHAP attribution divergence (§3c); the
CICIDS2017 `flowmatch` re-run under un-saturated GPU memory; root-cause diagnosis
of the ~1-in-5 collapses (§3b-ii); class-overlap complexity measures (§3e).

**Paper**
- Only the Introduction exists. Related Work, Method, Experiments, Results,
  Discussion, Conclusion unwritten.
- Bibliography verification — entries assembled from landing pages, not publisher
  exports; several fields marked `% CHECK`.
- Re-run the novelty search before submission. The field is moving: a directly
  relevant paper appeared 9 days before this project started. A search on
  2026-07-27 found that fidelity≠utility is **already published** in the general
  tabular literature (arXiv 2503.05954), so that claim cannot carry the paper on
  its own; the IDS-specific version and the undetectable-bad-batch result (§3b-ii)
  are what remain distinctive.
- Repo cleanup and reproduction instructions.

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
