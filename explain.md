# What This Project Is — Explained From Scratch

No background assumed. If you know what a computer network is, you can read this.

---

## The 30-second version

Computers on a network constantly talk to each other. Most of that talk is normal. A tiny
fraction is an attack. We want software that spots the attacks.

The software works well on common attacks and **fails badly on rare ones** — the kind
that show up a few dozen times in a hundred thousand records. Those rare ones are often
the most dangerous.

Our idea: **use AI to manufacture realistic fake examples of the rare attacks**, add them
to the training material so the software sees enough of them to learn, and then check
honestly whether that actually helped.

---

## Part 1 — The setting

### What is an intrusion detection system?

Think of a security guard watching the entrance of a building. Every person who walks in
gets looked at. Most are employees going to work. Occasionally someone is trying to sneak
in.

A **Network Intrusion Detection System (NIDS)** is that guard, for a computer network.
Every connection — someone loading a webpage, an email arriving, a file transfer — gets
inspected. The system decides: *normal*, or *attack*?

We're building the "brain" of that guard using machine learning.

### What is machine learning, in one paragraph?

Instead of writing rules by hand ("block anything from this address"), you show a program
thousands of labelled examples — *this connection was normal, this one was an attack* —
and it works out the patterns by itself. Then you show it a connection it has never seen
and it guesses which kind it is.

The program is called a **model**. Teaching it is **training**. Checking it afterwards is
**testing**.

---

## Part 2 — What the data actually looks like

It is not video, or code, or scary hacker screens. It is a **spreadsheet**.

Each row is one network connection. Each column is a measured property of that
connection. Here are a few real columns from our dataset:

| column | meaning |
|---|---|
| `duration` | how many seconds the connection lasted |
| `protocol_type` | tcp, udp, or icmp |
| `service` | http, ftp, smtp… what it was used for |
| `src_bytes` | bytes sent from source to destination |
| `dst_bytes` | bytes sent back |
| `num_failed_logins` | how many login attempts failed |
| `label` | **the answer**: normal, or which attack type |

There are 41 such columns. We use the first 41 to predict the last one.

This is why the project runs on an ordinary laptop GPU: it's a spreadsheet with 125,973
rows, not high-resolution video.

### The dataset we're using first

**NSL-KDD.** A public, free, widely used benchmark. Everyone in this field has used it,
which is exactly why it's useful — our numbers can be compared against published ones.

Its attacks are grouped into four families:

- **DoS** (Denial of Service) — flood the target until it collapses. Like jamming a
  phone line by calling it non-stop.
- **Probe** — scan for weaknesses. Like walking around a building trying every door
  handle.
- **R2L** (Remote to Local) — break in from outside, e.g. by guessing a password.
- **U2R** (User to Root) — already inside as a normal user, escalate to administrator.
  Like a hotel guest getting hold of the master key.

Roughly: DoS and Probe are noisy and common. **R2L and U2R are rare and serious.**

---

## Part 3 — The actual problem

Here is the real distribution of our training data:

| class | training examples | share |
|---|---|---|
| normal | 67,343 | 53% |
| dos | 45,927 | 36% |
| probe | 11,656 | 9.3% |
| **r2l** | **995** | **0.8%** |
| **u2r** | **52** | **0.04%** |

Look at that last row. **52 examples out of 125,973.** For every U2R example, there are
about 1,295 normal ones.

This is called **class imbalance**, and it wrecks the model in a specific way.

### Why imbalance breaks things

Suppose you're graded on how many connections you classify correctly, and 99.96% of them
are not U2R. A lazy strategy scores 99.96%: **guess "not U2R" every single time, forever.**

You'd never catch a single U2R attack, and your score would look excellent.

That's roughly what the model learns to do. It isn't broken — it's doing exactly what we
asked. We asked the wrong question.

### What this looks like in our actual results

We trained a standard model on this data. Here's what came out:

| class | recall | meaning |
|---|---|---|
| normal | 97% | catches almost all normal traffic |
| dos | 84% | good |
| probe | 64% | okay |
| **r2l** | **12%** | **misses 88 out of every 100** |
| **u2r** | **21%** | **misses 79 out of every 100** |

Overall accuracy: **78%**. Sounds acceptable. It is hiding a near-total failure on the
two attack types you'd most want to catch.

### Two words you need

- **Recall** — of all the real R2L attacks out there, what fraction did we catch?
  *(Low recall = attacks slip through.)*
- **Precision** — of everything we flagged as R2L, what fraction really was?
  *(Low precision = false alarms.)*

Our model's R2L precision is **98.5%** but recall is **12%**. Translation: when it finally
speaks up, it's almost always right — it just almost never speaks up. A guard who is
never wrong because he barely ever stops anyone.

**F1** is a single number combining both, so you can't win by cheating on one. Our R2L F1
is 0.21 out of a possible 1.00.

---

## Part 4 — The obvious fix, and why it isn't enough

If the problem is "only 52 examples," the obvious answer is "make more examples."

### Approach 1: copy them

Duplicate the 52 U2R rows until you have thousands. This is **random oversampling**.

Problem: no new information. The model sees the same 52 patterns repeatedly and
memorises them exactly, then fails on any U2R attack that looks slightly different.
Like preparing for an exam by reading the same 52 flashcards a hundred times.

### Approach 2: average them (SMOTE)

**SMOTE** takes two similar real examples and invents a new one partway between them —
like averaging two faces to make a third.

Better, but it can only ever produce blends of what already exists. And on data where
the boundary between classes is complicated, the midpoint of two attacks can land in a
region where normal traffic lives — so you've just fed the model a mislabelled example.

SMOTE is from 2002 and is still the standard. It's the number to beat.

### Approach 3: teach a program to invent them

This is where modern AI comes in — **generative models**, the same family of ideas behind
image generators. Instead of copying or averaging, the program studies the 52 examples,
learns the underlying *pattern* of what makes something a U2R attack, and then produces
genuinely new examples that fit the pattern.

That's the promising direction. And it's where our project sits.

---

## Part 5 — What we're actually building

### The generative model idea, plainly

Imagine learning to draw a cat. One way: trace existing cat pictures (copying). Another:
study many cats until you understand cat-shape, then draw a new cat nobody has seen.

Generative models do the second thing. Two recent families:

**Diffusion models** — train by taking a real example and gradually burying it in random
static until it's pure noise, while the model learns to reverse each step. Afterwards,
hand it fresh noise and it reverses the process into something new and realistic. Like
learning to un-blur, then un-blurring random fog into a photograph.

**Flow matching** — same goal, more direct route. Instead of learning to undo many small
noise steps, it learns a *direction of travel*: from any point in "random noise" space,
which way is "realistic example"? Then it just follows that direction.

Fewer steps, simpler to train, less to tune. **That's what we're using.**

### Why flow matching and not diffusion?

Two reasons.

1. **Diffusion has already been done here.** Around ten papers have applied diffusion
   models to this exact problem. Doing it an eleventh time isn't research.
2. **Nobody has tried flow matching on this problem.** We searched. That's the gap.

A research paper needs to do something that hasn't been done. This is that something.

---

## Part 6 — How we'll know if it worked

This part matters more than the method. A result nobody trusts isn't a result.

### The rules we hold ourselves to

**Never let the model see the test data.** We split the data into training and test. The
model learns from training only. The test set is sealed until the very end.

This sounds obvious and is violated constantly. If you generate fake examples using
information from the test set, your scores will look wonderful and mean nothing. It's
studying the answer key before the exam and then bragging about your grade.

**Never report only overall accuracy.** We report precision, recall and F1 *for every
class separately*, always. The whole point is that the overall number lies.

**Never report a single run.** Machine learning has randomness in it. Run the same
experiment five times with different random starting points and report the average and
the spread. A single lucky run isn't evidence.

**Compare against everything, not just the weakest option.** We test our method against:
no fix at all, copying, SMOTE, and two other modern AI generators. Beating "no fix" is
easy and proves nothing. Beating SMOTE is the real bar.

### And a sanity check before any of that

Before trusting a single result, we check whether the fake examples are any good:

- Train a model **only on fake data**, test it on real data. If it works, the fakes
  captured something real.
- Train a separate program to **tell fake from real**. If it can do this easily, our
  fakes are obviously wrong and everything downstream is worthless.
- Check the fakes aren't just **near-copies** of the originals. If they are, we've
  reinvented copy-paste with extra steps.

---

## Part 7 — What we've found so far

We're one week in. Two things already surfaced.

### Finding 1: On this dataset, the rare-attack problem isn't what people think

We looked at *which specific attacks* make up the R2L class:

- **89% of R2L training data is one attack type called `warezclient`** — 890 of 995
  examples.
- **`warezclient` appears exactly zero times in the test set.**
- Meanwhile the biggest R2L attacks *in the test set* — `snmpguess`, `snmpgetattack`,
  `httptunnel` — appear **zero times in training**.

Think about what that means. You study for an exam using 890 questions about one topic.
The exam contains none of that topic and asks about three topics you never saw.

**No amount of generating fake examples fixes this.** A generator trained on R2L data
learns to produce `warezclient`. Producing more `warezclient` cannot help you recognise
`snmpguess`. The failure isn't "too few examples" — it's "the wrong examples entirely."

This is genuinely useful to know. The field cites this dataset's R2L failure constantly
as proof of an imbalance problem. It isn't one. That's a paragraph in our paper.

It also settled our plan: this dataset becomes a **teaching example** showing where
augmentation can't help, and our real claims move to two other datasets that don't have
this defect.

### Finding 2: The paper we're building on has an error

Standard practice: before improving someone's work, reproduce it. So we did.

Their results table reports **13,422 examples of U2R**. The entire dataset contains
**119**. The subset they used contains **11**.

Reading the full table, every class's number matches the *next* class's true count — the
labels are shifted by one row. Which means their widely-quoted finding "R2L detection
fails, F1 = 0.15" is actually about **U2R**, and their "perfect U2R detection" is actually
the **normal** class, which is easy — hence the perfect score.

Their underlying point survives: a rare class really does fail badly. Only the name
attached to it is wrong. We'll use our own measurements instead, and note the discrepancy
carefully.

---

## Part 8 — Where this goes

| stage | what happens | when |
|---|---|---|
| ✅ Setup | Load data, measure the imbalance, build honest baseline | done |
| Baselines | Add SMOTE and the other standard fixes | weeks 2–3 |
| **Build it** | Implement flow matching, check the fakes are real | weeks 4–6 |
| **The test** | Does it beat SMOTE? All datasets, five runs each | weeks 7–10 |
| Explain | Work out *why* it helps, using a tool called SHAP | weeks 11–12 |
| Write | Turn it into a paper and submit | weeks 13–16 |

Two honest checkpoints are built in. At week 6: *are the generated examples any good?*
At week 8: *does this actually beat SMOTE?* If either answer is no, we change direction
then — not after four months.

And if flow matching turns out **not** to beat SMOTE, that's still a publishable result.
"Everyone assumes the fancy method wins; here's careful evidence it doesn't" is a real
contribution. We're not required to win — we're required to find out honestly.

---

## Glossary

| term | plain meaning |
|---|---|
| **model** | the program that learns to classify |
| **training set** | examples the model learns from |
| **test set** | sealed examples used once, to grade it |
| **class** | a category (normal, dos, probe, r2l, u2r) |
| **class imbalance** | some categories have far fewer examples than others |
| **precision** | of what we flagged, how much was right |
| **recall** | of what was really there, how much we caught |
| **F1** | one number combining precision and recall |
| **oversampling** | adding more examples of a rare class |
| **SMOTE** | oversampling by averaging pairs of real examples |
| **generative model** | AI that invents new realistic examples |
| **diffusion model** | generative model that learns to reverse added noise |
| **flow matching** | newer, faster relative of diffusion — **our method** |
| **data leakage** | accidentally letting test information into training; makes results fake |
| **baseline** | the score you must beat for your work to mean anything |
| **ablation** | removing one piece at a time to see what actually mattered |
| **SHAP** | tool that shows which columns drove a decision |
