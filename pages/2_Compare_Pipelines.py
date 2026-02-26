"""
pages/2_Compare_Pipelines.py
Run the same query through multiple RAG pipelines in parallel and compare side-by-side.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import concurrent.futures
import streamlit as st
import pandas as pd

from backend.pipelines import PIPELINE_MAP, PIPELINE_OPTIONS
from backend.pipelines.base import RAGResponse
from backend.config import get_generation_model, get_grading_model

st.set_page_config(
    page_title="Compare Pipelines | RAG Showcase",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0d1117; }
  .result-card {
    background: #161b22;
    border-radius: 10px;
    padding: 16px 18px;
    border-top: 3px solid;
    height: 100%;
    margin-bottom: 12px;
  }
  .result-header {
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 4px;
  }
  .result-meta {
    font-size: 0.78rem;
    color: #8b949e;
    margin-bottom: 12px;
  }
  .trace-mini {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 0.78rem;
    max-height: 180px;
    overflow-y: auto;
    color: #8b949e;
  }
</style>
""", unsafe_allow_html=True)

DEMO_QUERIES = [
    "Compare the risk factors cited by Apple and Microsoft in their 2023 annual reports",
    "How are rising interest rates connected to tech sector layoffs?",
    "What did the Federal Reserve signal about inflation in late 2023?",
    "What is Nvidia's current stock price?",
    "What is EBITDA?",
    "What happens to a company's balance sheet when it takes on debt for acquisitions?",
    "How do companies manage currency risk in international operations?",
    "How does EBITDA relate to debt covenants in a leveraged buyout?",
]

STEP_COLORS = {
    "RETRIEVE":    "#58a6ff",
    "GRADE":       "#f0883e",
    "ROUTE":       "#a371f7",
    "GENERATE":    "#3fb950",
    "EXPAND":      "#d2a8ff",
    "FUSE":        "#ffa657",
    "HYPOTHESIZE": "#79c0ff",
    "ERROR":       "#f85149",
}


@st.cache_resource(show_spinner="Initialising pipeline…")
def _get_pipeline(name: str):
    return PIPELINE_MAP[name]()


def _run_one(name: str, query: str) -> tuple[str, RAGResponse | None, str | None]:
    try:
        p = _get_pipeline(name)
        result = p._timed_run(query)
        return name, result, None
    except Exception as e:
        return name, None, str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────
pipeline_labels = {p["name"]: p["label"] for p in PIPELINE_OPTIONS}
pipeline_names  = [p["name"] for p in PIPELINE_OPTIONS]
pipeline_colors = {p["name"]: p["color"] for p in PIPELINE_OPTIONS}

with st.sidebar:
    st.markdown("## Select Pipelines")
    st.caption("Choose 2-4 pipelines to compare.")

    DEFAULT_SELECTED = {"agentic", "hyde", "fusion", "crag"}
    selected_names = []
    for p in PIPELINE_OPTIONS:
        checked = st.checkbox(
            p["label"],
            value=(p["name"] in DEFAULT_SELECTED),
            key=f"chk_{p['name']}",
            help=p["description"],
        )
        if checked:
            selected_names.append(p["name"])

    if len(selected_names) > 4:
        st.warning("Select at most 4 pipelines for readability.")
        selected_names = selected_names[:4]

    st.divider()

    run_parallel = st.toggle(
        "Run in parallel",
        value=True,
        help=(
            "Use ThreadPoolExecutor to run pipelines concurrently. "
            "Disable if you hit rate limits."
        ),
    )

    st.divider()
    st.caption(f"Generation: `{get_generation_model()}`")
    st.caption(f"Grading: `{get_grading_model()}`")

    st.divider()
    st.markdown("### Demo Queries")
    for q in DEMO_QUERIES:
        short = (q[:48] + "...") if len(q) > 48 else q
        if st.button(short, key=f"dq_{q[:20]}", use_container_width=True):
            st.session_state["cmp_query"] = q
            st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Compare Pipelines")
st.caption(
    "Run the same query through multiple architectures simultaneously and compare "
    "answers, latency, sources, and reasoning traces head-to-head."
)

query = st.text_area(
    "Your question",
    value=st.session_state.get("cmp_query", ""),
    placeholder="Ask a question to run through all selected pipelines...",
    height=88,
    key="cmp_query_input",
)

col_run, col_clear, _ = st.columns([1, 1, 6])
with col_run:
    run_btn = st.button(
        "Run Comparison",
        type="primary",
        use_container_width=True,
        disabled=(len(selected_names) < 1),
    )
with col_clear:
    if st.button("Clear", use_container_width=True):
        for k in ["cmp_query", "cmp_results", "cmp_query_used"]:
            st.session_state.pop(k, None)
        st.rerun()

if not selected_names:
    st.info("Select at least one pipeline in the sidebar.")

# ── Run pipelines ─────────────────────────────────────────────────────────────
if run_btn and query.strip() and selected_names:
    cache_key = (query.strip(), frozenset(selected_names))
    if st.session_state.get("_cmp_cache_key") != cache_key:
        prog = st.progress(0, text="Starting...")

        if run_parallel:
            results_map: dict[str, tuple[RAGResponse | None, str | None]] = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(selected_names)
            ) as ex:
                futures = {
                    ex.submit(_run_one, name, query.strip()): name
                    for name in selected_names
                }
                done = 0
                for fut in concurrent.futures.as_completed(futures):
                    name, result, err = fut.result()
                    results_map[name] = (result, err)
                    done += 1
                    prog.progress(
                        done / len(selected_names),
                        text=f"Completed {pipeline_labels[name]} ({done}/{len(selected_names)})",
                    )
        else:
            results_map = {}
            for i, name in enumerate(selected_names):
                prog.progress(i / len(selected_names), text=f"Running {pipeline_labels[name]}...")
                _, result, err = _run_one(name, query.strip())
                results_map[name] = (result, err)
                prog.progress(
                    (i + 1) / len(selected_names),
                    text=f"Done {pipeline_labels[name]}",
                )

        prog.empty()

        st.session_state["cmp_results"] = {
            name: results_map[name]
            for name in selected_names
            if name in results_map
        }
        st.session_state["cmp_query_used"] = query.strip()
        st.session_state["_cmp_cache_key"] = cache_key

elif run_btn and not query.strip():
    st.warning("Please enter a question first.")

# ── Display results ───────────────────────────────────────────────────────────
if "cmp_results" in st.session_state:
    cmp    = st.session_state["cmp_results"]
    q_used = st.session_state.get("cmp_query_used", "")

    st.divider()
    st.markdown(f"**Query:** `{q_used}`")

    # ── Summary cards ─────────────────────────────────────────────────────
    st.markdown("### Summary")

    n = len(cmp)
    sum_cols = st.columns(n)
    max_latency = max(
        (r.latency_ms for r, _ in cmp.values() if r),
        default=1,
    ) or 1

    for col, (name, (result, err)) in zip(sum_cols, cmp.items()):
        color = pipeline_colors.get(name, "#58a6ff")
        label = pipeline_labels[name]
        with col:
            if err or result is None:
                st.markdown(
                    f'<div class="result-card" style="border-top-color:{color};">'
                    f'<div class="result-header" style="color:{color};">{label}</div>'
                    f'<div style="color:#f85149; font-size:0.82rem;">Error: {str(err)[:80]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                continue

            bar_pct = int((result.latency_ms / max_latency) * 100)
            st.markdown(
                f'<div class="result-card" style="border-top-color:{color};">'
                f'<div class="result-header" style="color:{color};">{label}</div>'
                f'<div class="result-meta">'
                f'{result.latency_ms} ms &nbsp;|&nbsp; '
                f'{len(result.sources)} sources &nbsp;|&nbsp; '
                f'{len(result.trace)} steps'
                f'</div>'
                f'<div style="height:5px; background:{color}; '
                f'width:{bar_pct}%; border-radius:3px; margin-bottom:8px;"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Answers side by side ──────────────────────────────────────────────
    st.markdown("### Answers")
    ans_cols = st.columns(n)
    for col, (name, (result, err)) in zip(ans_cols, cmp.items()):
        color = pipeline_colors.get(name, "#58a6ff")
        label = pipeline_labels[name]
        with col:
            st.markdown(
                f'<div style="color:{color}; font-weight:700; margin-bottom:8px; font-size:0.9rem;">'
                f'{label}</div>',
                unsafe_allow_html=True,
            )
            if err or result is None:
                st.error(f"Error: {err}")
            else:
                st.markdown(result.answer)

    st.divider()

    # ── Reasoning traces ──────────────────────────────────────────────────
    st.markdown("### Reasoning Traces")
    trace_cols = st.columns(n)
    for col, (name, (result, err)) in zip(trace_cols, cmp.items()):
        color = pipeline_colors.get(name, "#58a6ff")
        label = pipeline_labels[name]
        with col:
            st.markdown(
                f'<div style="color:{color}; font-weight:700; margin-bottom:8px; font-size:0.9rem;">'
                f'{label}</div>',
                unsafe_allow_html=True,
            )
            if result:
                with st.expander("Show full trace", expanded=False):
                    for step in result.trace:
                        lc = STEP_COLORS.get(step.step, "#8b949e")
                        st.markdown(
                            f'<div style="margin:4px 0; font-size:0.82rem;">'
                            f'<span style="color:{lc}; font-weight:700;">[{step.step}]</span> '
                            f'{step.description}</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.caption("No trace (error)")

    st.divider()

    # ── Head-to-head analysis ─────────────────────────────────────────────
    successful = {
        name: result
        for name, (result, err) in cmp.items()
        if result is not None
    }

    if len(successful) >= 2:
        st.markdown("### Head-to-Head Analysis")

        fastest      = min(successful, key=lambda n: successful[n].latency_ms)
        most_sources = max(successful, key=lambda n: len(successful[n].sources))
        most_steps   = max(successful, key=lambda n: len(successful[n].trace))

        a1, a2, a3 = st.columns(3)
        for col, title, winner_name, value in [
            (a1, "Fastest",          fastest,      f"{successful[fastest].latency_ms} ms"),
            (a2, "Most sources",     most_sources, f"{len(successful[most_sources].sources)} sources"),
            (a3, "Most trace steps", most_steps,   f"{len(successful[most_steps].trace)} steps"),
        ]:
            wc = pipeline_colors.get(winner_name, "#3fb950")
            wl = pipeline_labels.get(winner_name, winner_name)
            with col:
                st.markdown(
                    f'<div style="background:#161b22; border-radius:8px; padding:16px; '
                    f'border-top:3px solid {wc};">'
                    f'<div style="font-size:0.8rem; color:#8b949e;">{title}</div>'
                    f'<div style="font-size:1.05rem; font-weight:700; color:{wc}; margin:4px 0;">'
                    f'{wl}</div>'
                    f'<div style="color:#8b949e; font-size:0.85rem;">{value}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("#### Response Time (ms)")
        df = pd.DataFrame({
            "Pipeline":     [pipeline_labels[n] for n in successful],
            "Latency (ms)": [successful[n].latency_ms for n in successful],
        }).set_index("Pipeline")
        st.bar_chart(df)
