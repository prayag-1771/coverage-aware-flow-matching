# The paper, explained plainly

A walkthrough of `paper/main.tex` in the order it is written, using the same
headings. The aim is that you could defend any part of it in a viva. Nothing
technical is removed — the jargon is introduced rather than avoided.

**Before starting, five words that appear constantly:**

| Word | What it means here |
|---|---|
| **Class** | A category of network traffic. "Benign", "DoS", "U2R" are classes. |
| **Rare class** | A class with very few examples. U2R has 52 out of 125,973. |
| **Augmentation** | Manufacturing extra fake training examples of a rare class. |
| **F1** | A score from 0 to 1 combining "did you catch the attacks?" (recall) and "were you right when you shouted?" (precision). |
| **Arm** | One method being compared. We compare 8 arms. |

---

# Title — Coverage-Aware Flow Matching for Rare-Attack Detection

Three ideas in six words.

**Flow matching** is the AI technique we use to manufacture fake attack records.
**Rare-attack detection** is the problem. **Coverage-aware** is our contribution: we
found a number you can compute *beforehand* that tells you whether the technique will
help. That word is doing the real work in the title.

---

# Abstract

The 200-word summary a reviewer reads before deciding whether to read the rest.

It says: intrusion detection systems miss rare attacks; the standard fix is to
manufacture fake examples; we bring in a technique never used here before; it
produces 11 statistically significant improvements, 7 of which survive a stricter
second test; and we identify a number ("coverage") that predicts where it will work.
Then two honest caveats — 14 augmentations actually made detection *worse*, and four
common ways of judging fake-data quality fail to predict whether it helps.

---

# I. Introduction

## (opening)

A firewall's machine-learning model reports **99% accuracy**, and intrusions still
succeed. How?

Because accuracy counts every record equally. If 99.9% of traffic is normal, a model
that labels *everything* normal scores 99.9% — while catching zero attacks.

The concrete case: NSL-KDD has **52 U2R records out of 125,973**, a ratio of about
**1 in 1,295**. U2R means *user-to-root* — an attacker who already has an ordinary
account escalating to administrator. Exactly the attack you most need to catch, and
statistically invisible.

Our own baseline shows it: **accuracy 0.781, macro-F1 0.569.** ("Macro-F1" averages
the F1 of each class equally, so a rare class counts as much as a common one — it
refuses to let the majority hide the failure.)

The most telling number: **R2L precision 0.974 with recall 0.107.** When the model
does flag an R2L attack it is almost always right — it simply refuses to flag them.
It isn't confused; it's silent.

## A. Existing responses and their limits

Two families of fix exist.

**Interpolation.** SMOTE takes two real rare-attack records and invents one halfway
between them. Cheap, twenty years old, still the default. Its limit is structural: a
point between two existing points contains no *new* information, and if the two
points straddle a region belonging to a different class, the invented point lands in
the wrong territory and actively teaches the model something false.

**Learned generators.** Train a neural network on the rare class until it can produce
new samples that look like it. GANs came first, then diffusion (the family behind
image generators). Reported gains are large.

We give three reasons those gains are hard to interpret:

1. **Missing controls.** Many papers compare their generator against no augmentation
   and declare victory. That shows augmentation helps — not that *generative*
   augmentation helps. You need SMOTE in the comparison, and it is often absent.
2. **Missing information.** Per-class support is often unreported, so you can't tell
   a stable result from one computed on eleven test records. Single runs are common,
   which confuses method effect with luck.
3. **No mechanism.** Nobody has established *why* it works when it works. The implicit
   argument is "more realistic samples → better detection." We measured that
   directly and it is false.

## B. This work

We introduce **flow matching** to this problem — as far as we can tell, its first use
here.

An analogy: imagine a cloud of random dots and a photograph. Flow matching learns a
set of currents that push the random dots until they form the photograph. Diffusion
does something related but by removing noise in many small steps; flow matching
learns the current directly. It needs far fewer steps and has no noise schedule to
tune.

We also state something about NSL-KDD that reframes decades of results on it. **890
of its 995 R2L training records are one attack type, `warezclient`, which appears
zero times in the test set.** Meanwhile several R2L types in the test set have no
training examples at all.

So the *usable* R2L signal is about **105 records, not 995** — an effective imbalance
near **1:641** rather than 1:68. Training on `warezclient` cannot help you detect
`snmpguess`. This is not an imbalance problem; it is a **distribution shift** problem
— the training and test data are about different things.

## C. Contributions

Six numbered claims. In plain terms:

1. **A method** — flow matching, applied per attack sub-type, with column types
   inferred from data. 11 significant improvements, largest +85.6%.
2. **A scope condition** — sub-class *coverage*, computable before training, predicts
   where the method works.
3. **An evaluation protocol** — 8 methods × 3 datasets × 9 rare classes × 5 seeds ×
   2 classifiers, with proper statistics.
4. **Evidence augmentation can harm** — 14 of 63 comparisons are significant
   *degradations*.
5. **Evidence quality doesn't predict usefulness** — four measures, all fail.
6. **A reproducible artefact** — the code, and scripts that generate every table and
   figure from the saved results.

---

# II. Related Work

Establishes what already exists, so the gap you fill is visible.

## A. Resampling for imbalanced intrusion detection

SMOTE and ADASYN explained, and their known limits. The new part: we quantify that
they can push rare-class F1 *below* doing nothing, with corrected statistics and
large effect sizes.

## B. Generative models for tabular data

**CTGAN** — a GAN adapted to spreadsheet-shaped data. **TabDDPM** — diffusion adapted
the same way, using a *different mathematical process* for categorical columns than
for numeric ones, because a category isn't a point on a number line.

**Flow matching** — described above. Widely used for images and audio; not previously
applied here.

## C. Generative augmentation for intrusion detection

Three habits recur in this literature, and all three are consequential:

- one downstream classifier
- aggregate scores rather than per-class
- one run, no significance testing

Our data shows each one matters: a second classifier disagrees on every rare class,
aggregate scores hide opposite-signed effects happening simultaneously, and one
generative run in five fails silently.

## D. Evaluating synthetic data

The closest competing paper (IEEE Access 2026) benchmarks twelve generators on two of
our three datasets. We must position against it carefully — and the distinction is
real:

|  | They ask | We ask |
|---|---|---|
| Setup | Can fake data **replace** real data? | Does fake data **added to** real data help? |
| Labels | Binary: attack or not | Multi-class: nine named attack types |
| Metrics | Aggregate | Per class |

**Under a binary label, rare classes are invisible** — U2R gets absorbed into "attack".
So they cannot have found what we found.

## E. Data complexity measures

There's an existing literature on measuring how "hard" a classification problem is
from the geometry of the data. We borrow two measures and report both what they
predict (difficulty) and what they don't (whether augmentation will help).

---

# III. Method

## A. Problem setting

Formal statement. Two design choices worth understanding:

**Real records are never thrown away.** We only add the shortfall. So every arm
trains on the same real data plus a different synthetic supplement — the comparison
isolates the augmentation.

**The majority class is never cut down.** The usual first step is to *delete* benign
traffic to improve the ratio. We measured that: capping benign traffic at 100,000
records on CICIDS2017 dropped Bot F1 from **0.822 to 0.761** — worse detection despite
a nine-times better ratio. So we cap how far we *add*, not how much we delete.

## B. Augmentation arms

The eight methods. Four classical: none, random oversampling (just duplicate),
SMOTE, ADASYN. Four learned: CTGAN, diffusion, flow matching, per-type flow matching.

"None" is listed as an arm, not a reference line, **because on one dataset it wins.**

## C. Flow matching for tabular records

The mathematics, in words.

Take a real record `x₁` and a random noise vector `x₀`. Draw a straight line between
them. At time *t* you're at `xₜ = (1−t)x₀ + t·x₁`. The direction of travel is always
`x₁ − x₀`. Train a network to predict that direction from any point on any such line.

To generate: start from noise and follow the predicted directions to *t* = 1.

**The loss can never reach zero, and that's expected.** From a given midpoint, many
different real records could have been the destination, so the best possible
prediction is the *average* direction. On U2R that floor is around **1.31**, and our
models converge to about **0.35**. We say this so a reader comparing loss curves has
a reference for what "converged" looks like.

## D. Type-aware encoding — *our first contribution*

Flow matching assumes every column is a smooth number. Network records aren't.

Of NSL-KDD's 38 "numeric" columns:
- **6 are yes/no flags** (`land`, `root_shell`)
- **6 are small counts** (`num_shells`)
- **11 are unbounded integers** (`src_bytes`)
- **only 15 are genuinely continuous**

Treat a yes/no flag as smooth and the model produces **`land = 0.37`** — a connection
that cannot exist. Like a form recording "married: 0.37".

We infer each column's true type from the data and handle it accordingly. Result:
the mismatch between real and fake distributions on U2R falls from **0.424 to 0.056**
— roughly eight times better.

We are honest that this idea isn't new (TabDDPM does it) — but it hasn't reached this
application area, and the size of the effect means a comparison ignoring it is
measuring the *encoding*, not the generator.

## E. Per-attack-type generation — *our main contribution*

"R2L" isn't one attack. It's eight different ones sharing a label. Train one
generator on that mixture and it learns something *in between* them — matching none.

So we train one generator per attack type. But small types can't be modelled at all
(too few examples), so we only cover part of the class. **That fraction turns out to
be the key.**

**Coverage** (Equation 5) = the share of a class's records belonging to types big
enough to model.

- R2L: 3 of 8 types modelled, but those 3 are **97%** of the records → our method wins
- U2R: 1 of 4 types, only **58%** of records → our method loses

Same code, same settings, opposite outcomes, ordered by coverage. And you can compute
it from the labels **before training anything.**

**The control that proves it's coverage.** UNSW-NB15 has no sub-labels, so coverage is
100% by definition and per-type collapses into per-class. The two arms should then be
identical — and they are, within noise. If they'd differed, coverage couldn't be the
explanation.

## F. An architecture-matched diffusion baseline

To compare flow matching against diffusion fairly, our diffusion arm **inherits from
our flow matcher** and overrides only the generative mechanism. Same encoder, same
network, same everything else. So a difference between them is the *mechanism*, not
the plumbing.

CTGAN deliberately isn't matched — we use the published implementation, because
substituting our own GAN would invite "you didn't use the real thing." The cost is
that CTGAN-vs-flow differences might come from preprocessing, so we draw no mechanism
claims from that pair.

## G. Preventing leakage

**Leakage** = the model accidentally seeing test information during training,
producing scores that don't survive deployment.

Three rules: every transformation is fitted on training data only; synthetic records
never touch the test set; generators only see training records of their own class.

---

# IV. Experimental Setup

## A. Datasets

Three benchmarks spanning three decades, chosen to match the base paper.

NSL-KDD (1999-era, official split), UNSW-NB15 (2015, official split), CICIDS2017
(2017, 2.8M records, our own 70/30 split).

We repaired two real defects in CICIDS2017: infinity values where flow duration is
zero, and mis-encoded web-attack labels.

**Two classes are excluded from all claims** — Infiltration (11 test records) and
Heartbleed (3). With that few, F1 swings wildly between seeds. Including them moved
macro-F1 by up to 0.13 while every other class moved 0.002.

## B. Why NSL-KDD is reported as a diagnostic

The `warezclient` problem from §I-B, stated formally. Results on this benchmark
measure behaviour under **distribution shift**, not imbalance — so we report it, but
don't build headline claims on it.

## C. Classifiers

Two, deliberately chosen at opposite ends:

**XGBoost** — a tree ensemble. Asks yes/no questions about thresholds. Indifferent to
scale. A fake record landing inside an existing "leaf" changes almost nothing.

**MLP** — a neural network. Draws smooth boundaries, sensitive to scale. The *same*
fake record shifts a boundary it must then fit.

**Neither is tuned per method.** Tuning one arm more than another would confound the
method's effect with the effort spent on it — the most common way this comparison is
gotten wrong.

One detail worth knowing: XGBoost's row-sampling is set to 0.8, not 1.0. At 1.0 it is
fully deterministic — every "seed" returns an identical model and the reported
standard deviation is exactly zero, which measures nothing.

## D. Metrics and statistical protocol

Per-class precision, recall, F1, with test support attached.

**A limit we report rather than hide.** The Wilcoxon signed-rank test with 5 paired
samples has 2⁵ = 32 possible outcomes, so its smallest possible p-value is **0.0625**.
No result in this study can reach p < 0.05 by that test *regardless of effect size*.
Reporting Wilcoxon alone would make everything look insignificant for a reason
unrelated to the data.

So we use **bootstrap confidence intervals** as the primary criterion: resample the
five results 10,000 times; if the middle 95% of outcomes never includes zero, the
effect is real. Plus paired t-tests with **Holm–Bonferroni correction** — run seven
tests and one will look significant by chance, so the threshold is tightened.

## E. Degenerate fits

Sometimes training simply fails — the model predicts one class for the entire test
set. No error, no warning. On CICIDS2017 the unaugmented arm did this at one seed,
scoring macro-F1 0.0996.

That isn't a bad result; it's a *failure to train*. We detect these, count them, and
report the rate rather than averaging them in. Averaging that one seed would move the
arm from 0.943 to 0.775 and reverse the comparison it's part of.

Learned generators also fail *partially* — one rare class collapses while the overall
score looks fine. Those evade any threshold, so we report **medians alongside means**
throughout.

## F. Implementation

One laptop, GTX 1650 (4 GB), 14 GB RAM. Every result is written to disk as produced,
and every table and figure is generated from those files by script. **No number in
the paper is typed by hand.**

---

# V. Results

## A. Rare-class detection improves substantially

**The headline.** 11 improvements whose confidence interval excludes zero; 6 survive
Holm correction.

Biggest: **U2R from 0.219 to 0.407 — +85.6%**, p = 0.0021, Cohen's *d* = 4.68.

(*Cohen's d* measures effect size in standard deviations. Above 0.8 is "large". **4.68
is enormous** — it means the improvement dwarfs the run-to-run noise.)

What 0.219 means operationally: recall 0.131, so **seven of eight privilege-escalation
attempts go undetected.** Raising it to 0.407 moves that to roughly three in five.
Still not solved — but a different proposition.

### Which are gains in ranking, and which in threshold

**The most important subsection in the paper, and the one that keeps you honest.**

F1 is computed after the model commits to an answer. A method can raise F1 simply by
becoming more trigger-happy — flagging more traffic, catching a few more real attacks,
raising more false alarms.

**PR-AUC** doesn't work that way. It ignores the cutoff and asks whether the model
*ranks* attacks above normal traffic — genuinely better discrimination.

We computed both. Result: **7 of 11 improve on both.**

- All five UNSW improvements: ✓ both
- **Per-type flow matching: ✓ both, on both NSL-KDD classes**
- Diffusion and plain flow matching on NSL-KDD: **F1 up, PR-AUC down**

The mechanism is visible: diffusion's R2L precision falls from 0.974 to 0.624 while
recall rises from 0.107 to 0.190. It shouts more often, catches more, is wrong more.

**Why this helps you.** The four that fail are *existing* methods. The one that passes
everywhere is *yours*. Your contribution is the only generative variant that improves
discrimination itself — which is exactly what the coverage mechanism predicts.

## B. Coverage determines where per-type generation wins

The 97% vs 58% result, with the UNSW control. Covered in §III-E above.

The usable rule: **compute coverage first. If it's low, don't spend the GPU hours.**

## C. Generative versus classical, head to head

The stricter question: does a learned generator beat SMOTE?

**Four of nine classes — and we report all nine.**

Worst case: Worms, where plain duplication beats every generator by 0.159. With 130
training records, copying preserves them exactly and every synthesis method distorts
them.

We report the losses deliberately. It's what allows §V-B to read as a *scope
condition* rather than cherry-picking — and a reviewer who spots an omitted row
discounts everything else.

## D. Type-aware encoding

The 0.424 → 0.056 result, and honest acknowledgement that the idea exists elsewhere.

## E. No quality measure predicts utility

Four ways to judge fake data: can a detector spot it? how far is it from real
records? do the distributions match? does it change which features matter?

**None ranks the methods the way actual detection does.**

Sharpest case: ADASYN's fake U2R records are detected as fake **100% of the time** —
and ADASYN still produces one of the best R2L results. SMOTE's are indistinguishable
from real, and SMOTE ranks below it.

And it isn't undertraining: 16× longer training improved every distance measure while
the detector's ability to spot fakes stayed at 0.9996.

## F. Both standard settings can be reduced

Two conventions nobody had tested:

**Integration steps.** Everyone uses 50–1000. **10 works just as well.** Since each
step costs exactly one network evaluation, that's **100× cheaper** — arithmetic, not a
measurement, so nobody can dispute it.

**How much fake data.** Everyone fills to full balance. **A quarter is identical**, on
half the training rows.

## G. Where augmentation degrades detection

**14 of 63 comparisons are significant degradations.**

The largest single effect anywhere in this study is a *harm*: **SMOTE reduces UNSW
Shellcode F1 by 0.139, d = −9.59** — bigger in magnitude than any benefit SMOTE
produces anywhere.

This is a contribution, not an apology. These are the default choices in the field.

## H. Transfer across classifier families

We re-ran everything with the neural network. **The two classifiers never agree on
which method is best — 0 out of 9 classes.**

We also rule out the obvious objection: the scaler is fitted on augmented data, so
generator outliers could distort the encoding of real records — invisible to a tree,
harmful to a network. The distortion is real (diffusion shifts one column's mean by
1.9 standard deviations) but **fixing it recovers at most 13% of the loss.** So it's
the synthetic records themselves, not the encoding.

We frame this as a **scope condition**: the improvements are real and reproducible;
they are a property of the augmentation *and* the downstream model together.

---

# VI. Discussion

## A. Coverage, not capacity, decides where per-type generation helps

Why per-type wins R2L and loses U2R: coverage, not model size, data volume, or fit
quality. Confirmed by the UNSW control. Gives a practitioner something computable in
advance.

## B. Why the two generative families fail on different classes

Shellcode and Analysis are mirror images. On Shellcode every learned generator beats
doing nothing and every interpolation method loses. On Analysis, diffusion and flow
matching collapse to **exactly 0.000** — while CTGAN survives.

Analysis has *more* data (2,000 vs 1,133). The difference is geometry: 77.5% of
Analysis records have a nearest neighbour of a different class, vs 65.4% for
Shellcode. Analysis sits *inside* other classes' territory.

Both continuous-time generators spread mass smoothly, so mass near an overlapped
class leaks into neighbours. CTGAN's design apparently preserves the mode.

We also tested whether overlap predicts *gain* — it doesn't (partial correlation
−0.11 once difficulty is accounted for). Coverage is the statistic that works.

## C. Why fidelity and utility come apart

Fidelity measures reward closeness to the dense clusters real traffic forms.
Detection depends on whether fake records land on the *useful side of a boundary*.
Different quantities.

## D. Learned generators fail silently, at a measurable rate

**Every learned generator loses one seed in five on CICIDS Bot. No classical method
ever does. 3 of 3 versus 0 of 4.**

The mean-minus-median gap is the cleanest single indicator: ~0.000 for every
classical arm, −0.032 to −0.042 for every learned one.

By median, all three learned generators beat the baseline. By mean, all three lose.
Both are true — the median says what you get *when it works*, the mean includes the
failures. A deployment decision needs both.

## E. The three benchmarks disagree about the same method

SMOTE **improves every rare class** on NSL-KDD, **degrades two of four** on UNSW, and
**degrades** the only CICIDS class with room to move.

And the three datasets fail for three *different* reasons — distribution shift, class
overlap, and (mostly) not failing at all. **None is a shortage of examples**, which is
what the standard remedy addresses. The ablation confirms it: 4× more fake data moves
F1 by less than 0.004.

## F. What this means for evaluation practice

"Which augmentation method is best?" isn't well posed. Five methods win across nine
classes; the classifier changes the winner every time; the same method helps one
class and harms another in the same run.

Not that augmentation is useless — 11 improvements are solid. But a single aggregate
number from a single classifier on a single benchmark can't support the general
recommendations routinely drawn from it.

---

# VII. Limitations

What the work does **not** establish. Stating these yourself is a strength — a
reviewer who finds an unacknowledged limit assumes you didn't look.

- **Five seeds.** Bounds the resolution of any rank test; fewer than some recent work.
- **Two classifier families.** Enough to show transfer fails; not enough to say which
  family is unusual.
- **The neural network is untuned by design.** Necessary for fair comparison, but its
  absolute scores are a floor, not a claim about neural networks.
- **One split on CICIDS2017.** Confidence intervals capture method and classifier
  variability, not split variability.
- **One consumer GPU.** A much larger generator might behave differently.
- **CTGAN isn't architecture-matched** — deliberate, and no mechanism claims drawn.

---

# VIII. Conclusion

Restates it in order of strength:

1. We introduced flow matching here for the first time, per attack sub-type, with
   inferred column types.
2. **It works, hardest where the problem is hardest** — 11 improvements, 6 surviving
   correction, 7 confirmed on both metrics, up to +85.6%.
3. **Sub-class generation is the variant that holds up most completely** — the only
   one improving both F1 and ranking quality everywhere it helps.
4. **Coverage tells you in advance** where it's worth the compute.
5. Three bounding findings: augmentation can harm; quality doesn't predict utility;
   rankings depend on the classifier.
6. The recommendation: **compute coverage first.** High → use per-type generation.
   Low, or class already well detected → classical resampling or nothing. A default
   resampler applied without checking can damage the classes it was meant to protect.

---

# The three things worth remembering

**1. Your method is the one that survives the strictest test.** Diffusion and plain
flow matching raise F1 by becoming trigger-happy. Per-type flow matching improves
actual discrimination. That's the sentence to lead with.

**2. Coverage is a real contribution.** Not "our method works" but "here is a number,
computable before you train anything, that tells you whether it will."

**3. The honest parts make the positive parts believable.** The 14 harms, the 4
metrics that fail, the 0-of-9 classifier disagreement — these are why a reviewer will
believe the 11 improvements. A paper reporting only wins invites the question of what
was left out. Yours answers it upfront.
