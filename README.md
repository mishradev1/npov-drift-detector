# Wikipedia Viewpoint-Drift Detector

A research tool that detects **when** a Wikipedia article's representation of
perspectives (and/or its semantic content) **began drifting over time**, and
which edits fall in that window — flagged for **human review**.

> **This is a drift detector, not an NPOV classifier.** It never outputs "this
> article is biased" or "this violates NPOV." Neutrality is a human judgment.
> The tool surfaces *measurable, directional, persistent* change as a
> **candidate for an editor to examine**, with the trajectory, the sections
> involved, and the responsible edits. A human decides what it means.

It works on **all article types**. Many articles (chemistry, geography, math)
have a single perspective; for those the viewpoint signal automatically goes
**inactive** and the tool falls back to semantic/structure drift. Contested
articles (politics, events, people, ideologies, medical controversies) are where
the viewpoint signal activates. The tool **self-detects** which case it is in,
per article, from the data — not from a hand-maintained topic list.

## Why this framing
Grounded in published research; see the design notes for full citations. The
load-bearing constraints:

- **NPOV = tone + due weight**, and *due weight* (proportional space to
  viewpoints by their prominence in reliable sources) is the hard part. We
  measure **trends, not moments**. (WMF, *Recommended Practices for NPOV
  Research*, arXiv:2510.21526.)
- **Never baseline on an article's first revisions** — a stub is
  unrepresentative. We baseline on a mature, stable window and on a
  **type-matched Featured/Good-Article reference distribution**.
- **Do not conflate** "editors contributing from a POV" with "an NPOV
  violation." An article can be contested yet neutral because diverse editors
  balance it. We only report directional drift, and only as a candidate.
- LLMs misapply NPOV (false balance, over-hedging). Any LLM use is confined to
  **narrow, checkable sub-tasks** (e.g. stance toward a named target), validated
  against humans — never neutrality verdicts. (arXiv:2407.04183.)

## Signals (planned)
| Signal | What it measures | Active when |
|---|---|---|
| **A. Stance balance** (primary) | share of sentences favorable / critical / neutral *toward the topic* | stance dispersion is non-trivial (self-gating) |
| **B. Perspective-cluster balance** (primary) | one latent perspective crowding out others (unsupervised) | >1 stable cluster exists |
| **C. Due-weight / section share** (primary) | proportional space per section over time; alt-view sections shrinking | all articles |
| **Section embedding drift** (secondary) | *directional* semantic movement per section (straightness, not churn) | all articles |

Each is tracked as a **time series**; onset is estimated with changepoint
detection and cross-checked across signals. If all viewpoint signals are
inactive, the tool reports *"no viewpoint drift measurable; semantic drift
only"* rather than forcing a verdict.

## Status — all 7 phases implemented
Build order (see design notes): **[1] ingestion ✅**, **[2] stance + self-gating
✅**, **[3] drift time series ✅**, **[4] FA/GA reference corpus ✅ (infra +
proof-of-mechanism run; thresholds not yet production-calibrated)**, **[5] mature
baseline + changepoint onset ✅**, **[6] POV-tag validation ✅ (self-contained
harness + small proof run)**, **[7] Streamlit dashboard ✅**.

The pipeline is end-to-end and honestly characterised. The biggest remaining
work is *not* more features but **rigour**: a larger type-balanced reference
corpus with fixed-interval snapshots, the **stance-signal validation** against
POV tags (Phase 6 showed structure/semantic signals alone don't discriminate),
and human inter-annotator labels for the stance model.

**Phase 1 provides:**
- A polite, cached MediaWiki client (descriptive UA, `maxlag`, exponential
  backoff honouring `Retry-After`, min request interval). Every response is
  cached to disk, so re-runs hit the network zero times.
- Full **dense revision metadata** (every revision) + **sampled content
  snapshots** (monthly by default, anchored on first/last revision).
- Wikitext → non-overlapping **sections** (heading, level, plain text, word
  counts) with generic **boilerplate** and **alt-view** flags.
- **Article-type classification** (ORES → Lift Wing `articletopic` → coarse
  bucket) for later baseline bucketing. The `contested_prior` it returns is a
  *prior for choosing a comparison baseline only* — **not** a bias judgment.

**Phase 2 provides** (`src/npov_drift/stance/`):
- A **stance-toward-topic classifier**: dependency-free sentence splitting, and
  a zero-shot NLI classifier (DeBERTa-v3-MNLI, CPU) that scores each sentence
  against three competing hypotheses — favorable / critical / *neutral factual
  description* of the topic. The explicit neutral hypothesis keeps a *neutral
  description of a negative fact* ("the drug can cause nausea") on NEUTRAL
  rather than CRITICAL (the WMF caution; stance is toward the TOPIC, not raw
  sentiment).
- The **self-gating inactivity test**: requires a minimum presence of BOTH
  favorable and critical stance (viewpoint *balance* needs two sides). This is
  robust to the stance model's main failure mode — on factual prose its false
  positives cluster on one side — so factual articles (chemistry, math) gate
  INACTIVE and the tool falls back to semantic/structure drift. Verified on real
  data: *Capital punishment* → ACTIVE (30% fav / 33% crit), *Sodium chloride* →
  INACTIVE (17% fav / 5% crit). Side-fraction floor is provisional, to be
  calibrated on the FA/GA noise floor in Phase 4.
- A **deterministic stub** classifier so the whole pipeline + CI run with no
  torch/GPU/network.
- A hand-labeled, multi-type **validation set** (`data/labeled/stance_gold.jsonl`,
  rubric in `docs/stance_labeling_rubric.md`) + validator.

Validation (70 curated probes, DeBERTa-v3-MNLI, CPU): overall accuracy **0.93**,
macro-F1 **0.93**; favorable/critical recall **1.00**; neutral recall **0.83**;
neutral-fact-vs-stance "hard" subset **0.88**. Honesty caveats: this is an
AI-authored, single-annotator probe set (**no human IAA yet**); accuracy is on
*curated probes*, not in-the-wild prose; residual errors are the NLI model
reading positive-outcome facts as favorable; decision thresholds are
deliberately **not** tuned to this set and will be calibrated on the FA/GA
corpus in Phase 4.

**Phase 3 provides** (`src/npov_drift/series/` + `embedding.py`) — the four drift
time-series, each unit-tested deterministically with a fake encoder / stance stub
(no torch/GPU/network in CI):
- **C. due-weight section-share drift** (pure): each section's share of the body
  over time + aggregate alt-view share. Works on all articles.
- **secondary. directional section embedding drift** (MiniLM): per-section
  *straightness* = net displacement / path length — directional movement, not
  churn. Boilerplate excluded.
- **B. perspective-cluster balance** (MiniLM + KMeans): sentences clustered into
  latent aspect groups (fit once on pooled sentences for stable identity);
  concentration tracked via HHI (rising = one perspective crowding out others).
- **A. stance-balance series**: the Phase 2 classifier applied per snapshot;
  tracks `balance` (favorable − critical) over time.

Real-data run honesty: the **directional section drift** signal produces clean,
interpretable output (e.g. *Capital punishment* → "Public opinion" straightness
0.76). But the **cluster HHI** and **section-share** signals are visibly
contaminated by **stub-era early snapshots** (a 12-word 2002 stub trivially has
HHI 1.0; section "movers" collapse to the lead) — a concrete demonstration of
why Phase 5's mature-baseline selection is required, and why exact-heading
section matching across decades needs to become embedding-based. alt-view-by-
heading is 0% for both demo articles (their perspective sections are named
"Abolition"/"Public opinion"); composition-based alt-view identification is the
planned refinement.

**Phase 4 provides** (`src/npov_drift/reference/`):
- **Corpus fetching + bucketing** (`corpus.py`): pulls `Category:Featured
  articles` (optionally Good articles) and buckets each by topic type using the
  *same* Phase-1 classifier, so target and reference are bucketed identically.
- **Per-type normal profile + noise floor** (`profile.py`, pure/tested): from
  each reference article's due-weight series over its **mature** window (stubs
  excluded), aggregates per bucket a normal profile (alt-view share, lead share,
  section count, share concentration) and a **noise floor** = how much stable
  articles of this type naturally move per step (total-variation of section
  shares). This is what turns the provisional thresholds into calibrated ones:
  drift = movement beyond the type's p90 noise floor.
- **Build orchestration** (`build.py`) + `scripts/phase4_build_reference.py`,
  persisting `data/out/reference_profile.json`.

Proof-of-mechanism run (7 FAs, due-weight signal): the **science** noise floor
(TV/step p90 ≈ 0.13) came out far below arts/biography/sports — confirming that
"normal movement" must be calibrated **per topic type**. But the numbers are
**not yet production thresholds**, for three honest reasons:
1. **Snapshot spacing confound** — with only ~6 snapshots over ~20 years,
   consecutive "mature" snapshots are years apart, so the noise floor conflates
   per-edit churn with restructuring accumulated over years. Fix: denser /
   time-normalized snapshots (dovetails with Phase 5's windowing).
2. **Tiny, skewed corpus** — a random sample of 12 FAs yielded no
   politics/history/geography buckets. A real run needs a larger, type-balanced
   corpus (and Good Articles, via `--include-good`).
3. **Cheap signal only** — the **stance** and **embedding** noise floors aren't
   populated yet; the machinery exists but the DeBERTa stance pass over a large
   corpus is the expensive piece.

**Phase 5 provides** (`src/npov_drift/onset/`):
- **Mature-baseline selection** (`baseline.py`): the first snapshot crossing a
  body-word maturity floor — never the stub era. Changepoint detection then runs
  only on the mature window, which directly defuses the Phase-4 spacing confound.
- **Changepoint detection** (`changepoint.py`): a deterministic mean-shift
  detector (numpy only) used as the reproducible per-signal onset, plus a
  **ruptures PELT (rbf)** detector for the fuller changepoint set.
- **Cross-signal onset reconciliation** (`detect.py`): builds drift series
  relative to baseline (due-weight departure, alt-view share, semantic departure
  via MiniLM, optional stance balance), finds each signal's onset, and clusters
  them into a **consensus onset + agreement count**. If the stance signal is
  inactive or not run, the report says "structure/semantic drift only" — it
  never forces a viewpoint verdict.

**Phase 6 provides** (`src/npov_drift/validation/`):
- **POV-tag ground truth** (`pov_tags.py`): detects NPOV-dispute templates
  (`{{POV}}`/`{{NPOV}}`/`{{POV-section}}`/`{{Neutrality}}`/`{{Unbalanced}}`) and
  binary-searches an article's history for the revision that first added the tag
  — self-contained, no external dataset needed.
- **Validation metrics** (`evaluate.py`, pure/tested): for POV-tagged articles,
  does a flagged onset PRECEDE the human tag, and by how much (lead time)? For
  matched, never-tagged Featured-Article controls, the false-positive rate.
- `scripts/phase6_validate.py` runs a small, bounded proof on real articles.

Scope honesty: this is a **self-contained proof of the validation pipeline, not
the full Wiki-Reliability evaluation**. Limitations: (1) small N; (2) it samples
*currently* tagged articles (a biased set) rather than the full historical
add/remove event log; (3) a rigorous detection/control-FP rate needs a
**calibrated significance threshold** (the type noise floor from Phase 4, still
preliminary); (4) it runs the cheap due-weight + MiniLM signals — the **stance**
signal (the one that most directly tests *viewpoint* drift) is off by default on
CPU.

**Proof run finding (honest, and important):** the harness works (it found 658
tagged articles via template transclusion and binary-searched real tag-add
dates), but the cheap structure/semantic signals **do not discriminate**
POV-tagged articles from neutral Featured-Article controls — control onset
effects (0.37–0.93) overlapped and exceeded the POV ones (≈0.47–0.50), so the
control false-positive rate was **100%** at threshold 0, and no effect threshold
cleanly separates them (FAs actually move *more*, being actively developed). The
"onset precedes tag" leads of 12–15 years are artifacts of unthresholded
largest-shift detection, not anticipation of the dispute. This empirically
confirms the project's core thesis: **structural/semantic drift alone cannot
flag viewpoint disputes — the viewpoint-specific stance signal is required.**
Validating that properly needs the (CPU-expensive) stance pass over pre-tag
history plus the calibrated per-type threshold.

**Phase 7 provides** (`src/npov_drift/dashboard/` + `streamlit_app.py`):
- A view-agnostic, **unit-tested** report builder (`dashboard/report.py`) that
  assembles every output section for an article, and a thin Streamlit UI.
- The spec's outputs: **active-signals summary** (which of due-weight / semantic
  / stance are live), **estimated onset** + per-signal agreement, **drift
  trajectories** (with the type noise-floor band when available), a **section
  directional-drift map**, **due-weight section-share changes**, and the **key
  edits in the drift window** (diff links + editors + comments) for human review.
- A **hedged plain-language statement** that is always a "candidate flagged for
  human review", never a determination of bias.

Real-data sanity check (*Capital punishment*, cheap signals): estimated onset
~2005, and the key-edits panel surfaced the actual responsible edits in that
window — including a −37 kB blanking by an IP and its +37 kB restoration — each
with a clickable diff link, exactly the human-review hand-off the tool is for.

## Install
Requires Python 3.11+ (3.12 recommended for the later ML stack). CPU-only.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"          # Phase 1 + tests
# Later phases:  pip install -e ".[dev,ml,viz]"
```

## Run the Phase 1 demo (real data)
```bash
python scripts/phase1_demo.py                       # contested + factual defaults
python scripts/phase1_demo.py "Climate change" "Pythagorean theorem"
```
Writes a JSON summary per article to `data/out/` and prints a structured report.

## Run the Phase 2 stance tools (needs the ML extras)
```bash
pip install -e ".[ml]"                  # torch + transformers (CPU)
python scripts/phase2_validate.py       # stance metrics on the labeled set
python scripts/phase2_demo.py           # stance + self-gating on real articles
python scripts/phase3_demo.py           # drift time series (pure + MiniLM)
python scripts/phase3_demo.py --with-stance   # also the (slow) stance series
python scripts/phase4_build_reference.py --per-bucket 2 --max-classify 12   # FA reference profile
python scripts/phase5_demo.py            # mature baseline + changepoint onset
python scripts/phase6_validate.py --n-pov 3 --n-control 3   # validate onset vs {{POV}} tags
streamlit run streamlit_app.py           # Phase 7 interactive dashboard (needs .[viz])
```
First run downloads DeBERTa-v3-MNLI (~440 MB) and/or MiniLM (~80 MB) once.

## Tests
```bash
pytest
```
All tests are offline and deterministic (injected fake HTTP session + synthetic
revisions); no network or GPU required. Every feature has unit tests with
hand-computed expected values.

## What each number does and does **not** show
- Section word-count shares show **proportional space**, a proxy for due
  weight — *not* correctness or fairness.
- Article type and `contested_prior` choose a **comparison baseline**; they do
  not say an article is contested in fact, let alone biased.
- Nothing in Phase 1 measures drift yet. It is plumbing: fetch, parse, classify,
  sample, cache.

## Hard constraints honoured
CPU-only; polite + fully cached API access; topic/subject-specific config kept
out of source (only generic heading lists + learned thresholds later); honest
about scope; deterministic offline unit tests for every feature.
