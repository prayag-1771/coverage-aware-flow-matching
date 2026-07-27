# Achievements

Running record of what has been established, built, and measured.
Every number here was computed from public data on this machine and is
reproducible from the scripts in `experiments/`.

**Last updated:** 2026-07-26

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

### 3.3 The two datasets disagree

| | NSL-KDD | UNSW-NB15 |
|---|---|---|
| SMOTE vs no augmentation | **improves** every rare class | **degrades** macro-F1 and 2 of 4 rare classes |
| best macro-F1 | SMOTE (0.627) | random oversample (0.526) |
| flow matching vs SMOTE | loses | wins 3 of 4 rare classes |

Any claim of the form "method X helps with class imbalance in IDS", drawn from a
single benchmark, is unsupported. This is measurable with the pipeline here and
is a result in its own right.

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

- Flow-matching downstream result — running
- CTGAN and TabDDPM comparison arms — not built
- UNSW-NB15 and CICIDS2017 — not loaded; **headline claims depend on these**
- Ablations, significance testing, SHAP
- All paper sections except the Introduction
- Bibliography verification — entries assembled from landing pages, not
  publisher exports
- Cross-dataset transfer (deferred; see `PLAN.md` §9)

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
