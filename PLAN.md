# Research Plan — Flow-Matching Augmentation for Rare Attack Detection in NIDS

**Author:** Prayag
**Status:** Planning → Phase 0
**Target submission:** ~4 months from start
**Last updated:** 2026-07-26

---

## 1. One-paragraph summary

Network intrusion datasets are severely imbalanced: rare attack classes (R2L, U2R,
Infiltration, Heartbleed, Web Attack) make up a fraction of a percent of traffic, and
classifiers reporting 99% accuracy routinely detect almost none of them. The standard
fixes — SMOTE, ADASYN, GANs — have known weaknesses, and diffusion models have recently
been applied to this problem by roughly ten papers. **Flow matching**, the successor
generation of generative models, has not been applied to intrusion detection class
imbalance. This project builds a flow-matching augmenter for rare attack classes,
benchmarks it honestly against every standard alternative with per-class metrics on
three datasets, and uses SHAP to explain what the synthetic data actually teaches the
classifier.

---

## 2. Contribution statement

Two contributions, matching the two accepted forms of novelty:

**C1 — New technique, new setting.** Flow matching (proven in image and general tabular
generation) applied to NIDS rare-class augmentation for the first time. Lighter than
diffusion: fewer sampling steps, simpler training objective, no noise schedule to tune.

**C2 — Improved rare-class metrics.** Head-to-head against no-augmentation, random
oversampling, SMOTE, ADASYN, CTGAN, and TabDDPM, with per-class precision/recall/F1
across three datasets, five seeds, and significance testing. The published work in this
space largely skips this comparison — the representative recent paper (Diff-IDS, CMC
2025) reports 99.93% and never compares against SMOTE.

**C3 (stretch, deferred) — Cross-dataset transfer.** Does augmentation-driven rare-class
gain survive a change of dataset? Held in reserve; see §9.

---

## 3. Positioning — what is already taken

Verified by literature search on 2026-07-26.

| Idea | Status | Evidence |
|---|---|---|
| Diffusion for IDS imbalance | **TAKEN** (~10 papers, 2023–2026) | Diff-IDS (CMC 2025), DID-IDS, CDDPM, MAGE-ID, latent-diffusion IoT (arXiv 2601.16976) |
| TabPFN / tabular foundation models for IDS | **TAKEN** | Electronics 2025, 14(19), 3792 — explicitly covers rare-class recall |
| GAN augmentation for IDS | **HEAVILY TAKEN** | CTGAN, WGAN-GP, BGAN-TabTransformer (arXiv 2607.16348, July 2026) |
| SMOTE / ADASYN / focal loss for IDS | **SATURATED** | 7+ papers; Electronics 2025 survey maps the space |
| **Flow matching for IDS imbalance** | **APPEARS OPEN** | No paper found. Tabular flow-matching survey (arXiv 2502.17119) covers general tabular only |

**Caveat:** "appears open" is absence of search evidence, not proof. Re-verify before
writing the related-work section, and again before submission.

### 3.1 Base paper

**Diff-IDS — Yang et al., *Computers, Materials & Continua* 82(3), 2025.**

This is the paper this work improves on. It is the closest published approach: a
generative model (diffusion) applied to NIDS class imbalance, evaluated on CICIDS2017,
KDD99, and NSL-KDD.

**The improvement, in one sentence:** Diff-IDS uses diffusion; this work uses flow
matching, and supplies the baseline comparisons and per-class rare-attack metrics that
Diff-IDS omitted.

**Important practical caveat.** Diff-IDS has no public code, and its method converts
network traffic into grayscale images for a U-Net denoiser. It is not reproducible and
should not be reproduced. Therefore:

- **Cite** Diff-IDS as the base paper and the motivating work
- **Benchmark against TabDDPM** (public code, tabular-native) as the diffusion-family
  representative

Improving on a published *approach* while benchmarking against a reproducible
implementation of that approach is standard practice and defensible under review.

**Other key reference papers**
- Alsubaei — *Scientific Reports*, 2025. Reports NSL-KDD R2L F1 = 0.15 (on 11 test samples — see §7).
- Feb 2026 *Scientific Reports* few-shot LSTM imbalance paper. Structural template for a Q1 imbalance paper.
- Cantone et al. — *IEEE Access*, 2024. Cross-dataset collapse; relevant to C3.
- Gupta et al. — arXiv 2503.03022. Imbalance + drift; must cite, adjacent question.

---

## 4. Datasets

| Dataset | Role | Size | Rare classes |
|---|---|---|---|
| **NSL-KDD** | Development / smoke test | 125k train, 22.5k test | R2L, U2R (tiny — demo only) |
| **UNSW-NB15** | Second schema | 175k train, 82k test | Worms, Shellcode, Backdoor, Analysis |
| **CICIDS2017** | **Primary claims** | 2.8M rows, 78 features | Infiltration, Heartbleed, Web Attack (hundreds–thousands) |

Use the **standard published splits** (NSL-KDD `KDDTrain+`/`KDDTest+`, UNSW-NB15
train/test) so numbers are comparable to prior work. NSL-KDD's official split has
built-in distribution shift between train and test — that is a feature, not a bug, and
worth commenting on.

**Label granularity: multi-class, always.** Binary attack/normal defeats the entire
purpose. Map NSL-KDD's 23 attack labels to the 5 standard classes
(normal / DoS / Probe / R2L / U2R).

**Memory:** CICIDS2017 at float64 is ~1.7 GB. Downcast to float32 on load (~850 MB).
This is a RAM constraint, not a GPU constraint.

---

## 5. Method

### 5.1 Pipeline

```
raw CSV
  → clean (drop NaN/Inf, dedupe, strip whitespace in column names)
  → split (use official train/test; carve validation out of TRAIN only)
  → fit preprocessing on TRAIN ONLY (one-hot categoricals, scale numerics)
  → [AUGMENTATION ARM] applied to TRAIN ONLY
  → train classifier
  → evaluate on untouched TEST
  → SHAP on trained classifier
```

### 5.2 Augmentation arms

| Arm | Purpose |
|---|---|
| `none` | Floor. Shows the rare-class failure. |
| `random_oversample` | Trivial baseline. Often surprisingly competitive — include it. |
| `smote` | The standard. The number to beat. |
| `adasyn` | Standard variant. |
| `ctgan` | Generative baseline (GAN family). |
| `tabddpm` | Generative baseline (diffusion family) — the current state of the art in this space. |
| `flowmatch` | **Ours.** |

### 5.3 Flow matching design

Conditional flow matching on tabular data. Minimum viable implementation:

- One-hot encode categoricals; min-max or quantile-transform numerics to a bounded range
- Treat the full vector as continuous; learn a velocity field `v_θ(x, t)` with an MLP
- Training objective: regress `v_θ(x_t, t)` onto `(x_1 − x_0)` where
  `x_t = (1−t)·x_0 + t·x_1`, `x_0 ~ N(0,I)`, `x_1` = real minority sample
- Generation: integrate the ODE from `t=0` to `t=1` (Euler, 50–100 steps)
- Post-process: argmax over each categorical's one-hot block; clip numerics to valid range
- **One generator per rare class per dataset** (class-conditional by training separately —
  simpler than conditioning, and the classes are few)

Denoiser MLP: ~3–4 hidden layers, 256–512 units. A few hundred thousand parameters.
Trains in 10–30 min per class on a GTX 1650.

**Compute saver:** train the generator ONCE per class per dataset, then vary only the
classifier seed. Generation is cheap; retraining generators per seed is not. This cuts
total generator trainings from ~75 to ~15.

### 5.4 Classifiers

- **XGBoost** — primary. Fast, strong on tabular, and TreeSHAP works on it.
- **MLP** — secondary, to show the effect is not classifier-specific.

Two is enough. Do not add more; it multiplies runtime and adds nothing to the argument.

---

## 6. Evaluation protocol

**This section is where the paper is won or lost. Treat every item as mandatory.**

### 6.1 Leakage rules (non-negotiable)

1. Synthetic data is generated from the **training split only** and added to the
   **training split only**.
2. Scalers, encoders, and imputers are **fitted on train**, then applied to test.
3. The test set is touched exactly once per experiment, at the end.
4. Hyperparameter choices come from a **validation split carved out of train** — never
   from test.

### 6.2 Metrics

- **Per-class precision / recall / F1** for every class. This is the headline table.
- **Macro-F1** (unweighted mean over classes) — the summary number that respects rare classes.
- **PR-AUC** for rare classes specifically — more informative than ROC-AUC under extreme imbalance.
- **Balanced accuracy.**
- Overall accuracy reported **only** to demonstrate that it is misleading.

### 6.3 Statistics

- **5 seeds** per configuration; report mean ± standard deviation.
- Paired comparison of `flowmatch` vs `smote` using **Wilcoxon signed-rank** across
  seeds × classes × datasets.
- Single-run numbers are a standard rejection reason. Never report one.

### 6.4 Ablation

| Variant | Question answered |
|---|---|
| No augmentation | What is the baseline failure? |
| Flow matching, varying synthetic ratio (25/50/100% of majority) | How much synthetic data is optimal? |
| Flow matching, varying ODE steps (10/50/100) | Is the extra compute justified? |
| Flow matching with vs without categorical post-processing | Does validity enforcement matter? |

### 6.5 Synthetic-quality gate — run BEFORE any classifier results are trusted

If the synthetic data is garbage, no downstream number means anything.

- **TSTR** (train on synthetic, test on real) vs **TRTR** (train real, test real)
- **Detection test**: train a random forest to distinguish real from synthetic. AUC near
  0.5 = indistinguishable (good). AUC near 1.0 = trivially separable (bad — and this is
  the known failure mode of CTGAN).
- **Per-feature marginal comparison**: Kolmogorov–Smirnov statistic per feature
- **Correlation-structure preservation**: compare correlation matrices

Report these. They are also the honest answer to "is fidelity even the right target?" —
existing work has found fidelity and downstream utility only weakly related, sometimes
negatively.

### 6.6 Reproducibility

- Fixed seeds everywhere, logged
- `environment.yml` / `requirements.txt` pinned
- Public GitHub repo with instructions to reproduce every table
- Release the multi-class preprocessing code — no such clean harness currently exists
  publicly, and this is itself a minor contribution

---

## 7. Known traps

**Tiny-sample classes.** Per-class metrics on a handful of samples are dominated by
noise: with 11 samples, one flipped prediction moves F1 by ~9 points.

→ **Rule:** NSL-KDD is for development and illustration. Put the real claims on
CICIDS2017 rare classes, where sample counts are in the hundreds to thousands. Always
report per-class test-set support alongside metrics. Never write "solved R2L."

**Do not cite "Alsubaei R2L F1 = 0.15" without qualification — the class labels in that
table appear to be rotated by one position.**

Verified 2026-07-26. Alsubaei (Sci. Reports 2025) evaluates on `KDDTrain+_20Percent`
(25,192 records — confirmed, our copy matches exactly). Their Table 4 class counts vs.
the true distribution we computed from that file:

| their label | their count | true class at that count | true count |
|---|---|---|---|
| normal | 9,181 | dos | 9,234 |
| dos | 2,357 | probe | 2,289 |
| probe | 224 | r2l | 209 |
| r2l | **11** | **u2r** | **11** |
| u2r | **13,422** | normal | 13,449 |

Each reported class matches the *next* class's true count, within 1%.

**Certain:** 13,422 U2R records cannot exist. NSL-KDD contains 119 U2R records in total
(52 in KDDTrain+, 67 in KDDTest+); `KDDTrain+_20Percent` contains 11. The reported figure
is off by a factor of over 1,000.

**Strongly implied:** the widely-quoted "R2L F1 = 0.15" is actually **U2R**, and the
"perfect U2R detection (1.00 precision/recall/F1)" is actually the **normal** class —
which is trivially easy and explains the perfect score.

**Consequences for this project:**
1. The motivating premise survives — a rare class does fail badly at F1 ≈ 0.15. The
   class name attached to it is wrong.
2. Do not reproduce or cite that number as an R2L result; doing so propagates the error.
3. Use our own baseline instead (§8 Phase 0, `results/nsl_kdd_baseline_per_class.csv`),
   computed on the official KDDTrain+/KDDTest+ split with full per-class support reported.
4. Before putting this in the paper, read Alsubaei's Table 4 directly to confirm the
   counts, and state the observation neutrally — a labelling discrepancy in a published
   table, evidenced by counts that are physically impossible for the dataset.

**NSL-KDD's rare classes are a distribution-shift problem, not (only) an imbalance
problem.** Measured directly from the official split on 2026-07-26 — see
`results/nsl_kdd_class_distribution.csv`:

- R2L training data is **89% `warezclient` (890 of 995) — and `warezclient` has zero
  test samples.**
- The three largest *unseen* R2L test types — `snmpguess` (331), `snmpgetattack` (178),
  `httptunnel` (133) — have **zero training samples**. 642 of 2,887 R2L test samples
  (22%) come from attack types the model has never seen.
- U2R is the same: `ps` (15), `xterm` (13), `sqlattack` (2) appear only in test.
  30 of 67 U2R test samples (45%) are unseen types.
- R2L is also *more* common in test (12.8%) than train (0.79%).

**Consequence.** A generator trained on R2L training data learns `warezclient`. No amount
of synthetic `warezclient` improves detection of `snmpguess`. This is why published R2L
F1 hovers around 0.15, and **no augmentation method can fix it** — the failure is
zero-shot generalization, not class balance.

→ **This is a finding, not an obstacle.** Report it. It explains a number the field
keeps quoting without diagnosing, and it justifies the dataset strategy below.

→ **Revised dataset roles:** NSL-KDD is a *diagnostic* dataset — used to demonstrate the
imbalance problem and to show where augmentation cannot help and why. All headline
augmentation claims go on CICIDS2017 and UNSW-NB15, where rare-class attack types are
present in both splits.

**Overclaiming.** The honest framing is "improves rare-class detection under these
conditions," never "solves class imbalance."

**Generator memorization.** With very few minority samples, a generative model will
memorize them and output near-copies — functionally identical to random oversampling.
Check this explicitly: measure nearest-neighbour distance from each synthetic sample to
its closest real training sample. Report it.

---

## 8. Timeline

Working consistently. Weeks, not calendar months.

### Phase 0 — Infrastructure (Week 1)
- [ ] NSL-KDD loader + multi-class label mapping
- [ ] Class distribution table for all 3 datasets ← **the first real artifact**
- [ ] XGBoost baseline, per-class metrics, no augmentation
- [ ] Metrics module (per-class P/R/F1, macro-F1, PR-AUC)
- **Deliverable:** the imbalance table + baseline failure numbers. Show the professor.

### Phase 1 — Resampling baselines (Weeks 2–3)
- [ ] SMOTE, ADASYN, random oversampling arms
- [ ] Experiment runner: (dataset × arm × classifier × seed) grid
- [ ] Results logging to CSV; table generation
- **Deliverable:** first full comparison table.

### Phase 2 — Generative methods (Weeks 4–6)
- [ ] CTGAN arm
- [ ] TabDDPM arm
- [ ] **Flow matching implementation** ← main technical risk
- [ ] Synthetic-quality gate (§6.5) on all three
- **GATE (end of Week 6):** Does flow matching produce valid, non-memorized samples that
  pass the detection test? If no → debug for 2 weeks max, then fall back to a thorough
  comparative study of the other arms (still publishable).

### Phase 3 — Full experiments (Weeks 7–10)
- [ ] All arms × 3 datasets × 2 classifiers × 5 seeds
- [ ] Significance testing
- [ ] Ablations (§6.4)
- **GATE (end of Week 8):** Does flow matching beat SMOTE on rare-class F1?
  - Yes → proceed as planned
  - No → pivot framing to "when does generative augmentation help?" (honest comparative
    study — still a paper, and arguably a more interesting one)

### Phase 4 — Explainability & freeze (Weeks 11–12)
- [ ] TreeSHAP on XGBoost, global + per-rare-class attributions
- [ ] Compare attributions: real-trained vs synthetic-augmented models
- [ ] Freeze all results. No more experiments after this point.

### Phase 5 — Writing (Weeks 13–16)
- [ ] Related work (re-run the novelty search first — the field moves fast)
- [ ] Method, experiments, discussion
- [ ] Figures and tables
- [ ] Clean and document the public repo
- [ ] Supervisor review
- [ ] Format to journal template, submit

---

## 9. The upgrade path (C3 — held in reserve)

If the paper needs strengthening after review, or if results are strong and time allows:

**Cross-dataset transfer.** Train with each augmentation arm on dataset A, test on
dataset B. Cantone showed IDS models collapse to near-chance across datasets; nobody has
asked whether synthetic minority data helps or hurts that collapse.

**Important:** stay inside the CICFlowMeter family (CICIDS2017 → CSE-CIC-IDS2018 →
CIC-IoT2023) where feature schemas already match. Cantone did **not** solve cross-schema
harmonization — they picked compatible datasets. Harmonizing NSL-KDD ↔ UNSW-NB15 ↔
CICIDS2017 is a much larger job and is not on the critical path.

Cost: ~4–6 weeks. Reuses the entire existing pipeline. Nothing is wasted by deferring it.

---

## 10. Risks

| Risk | Response |
|---|---|
| Flow matching turns out to be published for IDS | Re-verify now and again at Week 12. If taken, the comparative-study framing survives intact. |
| Flow matching doesn't beat SMOTE | Reportable finding. Pivot framing at the Week 8 gate. |
| Someone scoops the idea mid-project | Real risk — a relevant paper appeared 9 days before this plan was written. Mitigation: move fast, and keep C3 in reserve as differentiation. |
| Results are mushy/inconclusive | The synthetic-quality gate (§6.5) surfaces this by Week 6, not Month 4. |
| CICIDS2017 memory issues | Downcast to float32; subsample majority classes; chunked loading. |
| Q1 rejection | Expected. Revise on reviewer feedback, resubmit down the tier ladder. |

**Hardware is not a risk.** Total GPU time for the whole project is roughly a weekend on
a GTX 1650. Tabular generative models are small.

---

## 11. Target venues

1. **IEEE Access** (Q1) — fast review, publishes exactly this kind of applied work. Primary.
2. **Scientific Reports** (Q1) — published both the Alsubaei paper and the Feb 2026 template. Secondary.
3. **Computers & Security** (Elsevier, Q1) — security-specific, no APC, slower review. Fallback.
4. Q2 tier if needed: Applied Sciences, Sensors, Electronics, Cluster Computing.

**Honest odds:** Q1 on first attempt ~25–35%. Publishing somewhere indexed, given the
work is finished and you're willing to submit down the tiers, ~90%.

**Largest single lever: get a supervisor who has published in indexed journals.** This
matters more than any technical choice in this document.

---

## 12. Repo structure

```
DiffIDS/
├── PLAN.md                  # this file
├── README.md
├── requirements.txt
├── data/                    # gitignored — raw datasets
│   ├── nsl-kdd/
│   ├── unsw-nb15/
│   └── cicids2017/
├── src/
│   ├── data/                # loaders, label mapping, preprocessing
│   ├── augment/             # smote, adasyn, ctgan, tabddpm, flowmatch
│   ├── models/              # xgboost, mlp wrappers
│   ├── eval/                # per-class metrics, quality gate, stats
│   └── explain/             # SHAP
├── experiments/             # runner scripts, configs
├── results/                 # logged CSVs, generated tables
└── notebooks/               # exploration only, not the source of truth
```

---

## 13. Immediate next actions

1. Download NSL-KDD from Kaggle into `data/nsl-kdd/`
2. Write the loader + multi-class label mapping
3. Produce the class distribution table
4. XGBoost baseline with per-class metrics

Step 3 is the artifact this whole project is built on. Everything else follows from
seeing those numbers.
