# Stance labeling rubric (toward the TOPIC)

We label each sentence by the stance it expresses **toward the article's topic**,
into one of three classes. This is *stance*, not *sentiment*.

## Labels
- **favorable** — the sentence expresses, endorses, or gives due-weight voice to
  a **positive evaluation** of the topic (praises it, defends it, argues for it,
  frames it as a success/virtue).
- **critical** — the sentence expresses, endorses, or gives due-weight voice to a
  **negative evaluation** of the topic (condemns it, argues against it, frames it
  as a failure/harm). Attributed evaluation counts ("Critics denounced X as
  cruel" is *critical*, because the article is voicing a critical stance).
- **neutral** — descriptive/factual statements, **including neutral descriptions
  of negative or positive facts**, procedural/historical statements, and balanced
  attributions.

## The crucial boundary (WMF caution)
A **neutral description of a negative fact is NEUTRAL, not critical.** Sentiment
or valence in the facts does not make a sentence take a stance.

| Sentence | Label | Why |
|---|---|---|
| "The drug can cause nausea in some patients." | neutral | states a side effect as fact; no evaluative stance |
| "Researchers warned the drug was dangerously overprescribed." | critical | evaluative condemnation of the topic |
| "The drug reduced symptoms in 70% of participants." | neutral | states a positive fact; not an endorsement |
| "Clinicians praised the drug as a breakthrough." | favorable | evaluative praise |
| "An estimated 16 million people died during the war." | neutral | factual casualty figure |
| "The war is condemned as a senseless catastrophe." | critical | evaluative judgment of the topic |

## Honesty note about this set
`data/labeled/stance_gold.jsonl` is an **AI-authored, rubric-based seed set with
a single annotator**. It is designed to *probe specific phenomena* (especially
the neutral-negative-fact boundary), not to be a random sample of Wikipedia
prose. Therefore:

- Accuracy on it measures behavior on **curated probes**, not ecological
  in-the-wild accuracy. Phase 6 validates on real POV-tagged articles.
- **No inter-annotator agreement (IAA) is reported** because there is only one
  (AI) annotator. Adding a second human annotation pass + reporting IAA is a
  pending, explicitly-acknowledged validation gap.

`hard: true` marks the neutral-negative/positive-fact probes; the validator
reports accuracy on that subset separately, since it is where the model is most
likely to fail by conflating sentiment with stance.
