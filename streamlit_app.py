"""Wikipedia Viewpoint-Drift Detector — Streamlit dashboard.

    pip install -e ".[ml,viz]"
    streamlit run streamlit_app.py

A candidate-flagger for human review, NOT an NPOV classifier. Every output is
hedged; the app never declares an article biased.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from npov_drift import config  # noqa: E402
from npov_drift.dashboard.report import build_drift_report  # noqa: E402
from npov_drift.embedding import MiniLMEncoder  # noqa: E402
from npov_drift.ingest.pipeline import ingest_article, make_client  # noqa: E402
from npov_drift.series.util import body_sentences  # noqa: E402
from npov_drift.stance.topic import topic_from_title  # noqa: E402


@st.cache_resource
def get_client():
    return make_client()


@st.cache_resource
def get_encoder():
    return MiniLMEncoder()


@st.cache_resource
def get_classifier():
    from npov_drift.stance.nli import NLIStanceClassifier

    return NLIStanceClassifier()


@st.cache_data(show_spinner=False)
def load_reference_profile():
    path = config.OUT_DIR / "reference_profile.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("profiles")
    return None


@st.cache_data(show_spinner="Ingesting + analysing (cached after first run)...")
def get_report(title: str, use_semantic: bool, use_stance: bool, min_words: int, max_snapshots: int):
    client = get_client()
    hist = ingest_article(client, title, max_snapshots=max_snapshots)
    encoder = get_encoder() if use_semantic else None
    classifier = get_classifier() if use_stance else None
    report = build_drift_report(
        hist,
        encoder=encoder,
        classifier=classifier,
        topic=topic_from_title(hist.title),
        sentence_fn=lambda s: body_sentences(s, min_words=5),
        min_words=min_words,
        reference_profile=load_reference_profile(),
    )
    return report


def trajectory_figure(onset):
    import plotly.graph_objects as go

    fig = go.Figure()
    for name, values in onset.series.items():
        fig.add_trace(go.Scatter(x=onset.timestamps, y=values, mode="lines+markers", name=name))
    if onset.consensus_timestamp:
        fig.add_vline(x=onset.consensus_timestamp, line_dash="dash", line_color="crimson",
                      annotation_text="onset")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h"), yaxis_title="signal value")
    return fig


def drift_map_figure(section_drifts):
    import plotly.graph_objects as go

    top = section_drifts[:12]
    fig = go.Figure(go.Bar(
        x=[d.straightness for d in top][::-1],
        y=[d.heading for d in top][::-1],
        orientation="h",
        marker_color=[d.net_displacement for d in top][::-1],
        text=[f"net={d.net_displacement:.2f}" for d in top][::-1],
    ))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="straightness (directionality)")
    return fig


def main():
    st.set_page_config(page_title="Wikipedia Viewpoint-Drift Detector", layout="wide")
    st.title("Wikipedia Viewpoint-Drift Detector")
    st.caption("A candidate-flagger for human review — it surfaces *measurable directional drift*, "
               "and never declares an article biased or NPOV-violating.")

    with st.sidebar:
        st.header("Article")
        title = st.text_input("Title", value="Capital punishment")
        min_words = st.slider("Maturity floor (body words)", 200, 3000, 800, 100)
        max_snapshots = st.slider("Content snapshots", 8, 40, 24, 2)
        st.header("Signals")
        use_semantic = st.checkbox("Semantic drift (MiniLM, ~slower)", value=True)
        use_stance = st.checkbox("Stance / viewpoint (DeBERTa, SLOW on CPU)", value=False)
        run = st.button("Analyse", type="primary")

    if not run:
        st.info("Enter an article and press **Analyse**. First run downloads models / fetches history; "
                "everything is cached afterwards.")
        return

    report = get_report(title, use_semantic, use_stance, min_words, max_snapshots)

    # --- header metrics ---------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Topic type", report.bucket)
    c2.metric("Revisions", f"{report.n_revisions:,}")
    c3.metric("Editors", f"{report.n_editors:,}")
    c4.metric("Span", f"{(report.date_span[0] or '?')[:4]}–{(report.date_span[1] or '?')[:4]}")

    # --- active signals ---------------------------------------------------
    st.subheader("Active signals")
    sc = st.columns(3)
    sc[0].success("Due-weight: ON") if report.active_signals["due_weight"] else sc[0].warning("Due-weight: off")
    sc[1].success("Semantic: ON") if report.active_signals["semantic"] else sc[1].warning("Semantic: off")
    if report.active_signals["stance"]:
        sc[2].success("Viewpoint (stance): ACTIVE")
    else:
        sc[2].warning("Viewpoint (stance): inactive / not run")

    # --- hedged statement + onset ----------------------------------------
    st.subheader("Estimated onset")
    st.info(report.hedged_statement)
    oc1, oc2 = st.columns(2)
    oc1.metric("Consensus onset", (report.onset.consensus_timestamp or "none")[:10])
    oc2.metric("Signal agreement", f"{report.onset.agreement}/{len(report.onset.signals)}")
    if report.onset.onsets:
        st.write("Per-signal onset:")
        st.table([
            {"signal": o.signal, "onset": o.timestamp[:10], "effect": round(o.effect, 3)}
            for o in sorted(report.onset.onsets, key=lambda o: o.timestamp)
        ])

    # --- trajectory -------------------------------------------------------
    st.subheader("Drift trajectories (relative to mature baseline)")
    if report.reference_noise_floor:
        st.caption(f"Type noise floor (share TV/step): {report.reference_noise_floor}")
    st.plotly_chart(trajectory_figure(report.onset), use_container_width=True)

    # --- section drift map -----------------------------------------------
    if report.section_drifts:
        st.subheader("Section directional-drift map")
        st.plotly_chart(drift_map_figure(report.section_drifts), use_container_width=True)

    # --- due-weight movers + key edits -----------------------------------
    left, right = st.columns(2)
    with left:
        st.subheader("Due-weight section-share change")
        if report.share_series:
            base, last = report.share_series[0], report.share_series[-1]
            common = set(base.shares) & set(last.shares)
            movers = sorted(((last.shares[h] - base.shares[h], h) for h in common), key=lambda x: x[0])
            rows = [{"section": h, "from": f"{base.shares[h]:.1%}", "to": f"{last.shares[h]:.1%}",
                     "Δ": f"{d:+.1%}"} for d, h in (movers[:5] + movers[-5:])]
            st.table(rows)
    with right:
        st.subheader("Key edits in the drift window")
        if report.key_edits:
            try:
                import pandas as pd

                df = pd.DataFrame(report.key_edits)[["timestamp", "user", "size_delta", "comment", "diff_url"]]
                st.dataframe(df, hide_index=True, use_container_width=True,
                             column_config={"diff_url": st.column_config.LinkColumn("diff")})
            except Exception:
                st.table(report.key_edits)
        else:
            st.write("No onset window / no edits to show.")

    st.caption("⚠️ Drift is a candidate signal for human review. Neutrality is a human judgment; "
               "diverse editors contributing from a viewpoint is not itself an NPOV violation.")


if __name__ == "__main__":
    main()
