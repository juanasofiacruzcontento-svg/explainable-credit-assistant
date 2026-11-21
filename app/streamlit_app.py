"""Interfaccia demo. Avvio: streamlit run app/streamlit_app.py"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from credit_xai import config, rag, service
from credit_xai.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

st.set_page_config(page_title="Explainable Credit Assistant", layout="wide")
st.title("Explainable Credit Assistant")
st.caption(
    "Demo di portfolio su dati pubblici (German Credit, UCI). "
    "Non è un sistema reale di concessione del credito."
)

try:
    model = service.get_model()
except FileNotFoundError:
    st.error("Modello non trovato. Esegui prima `python -m credit_xai.pipeline`.")
    st.stop()

NUMERIC_LIMITS = {
    "duration_months": (4, 72),
    "credit_amount": (250, 20000),
    "installment_rate": (1, 4),
    "residence_since": (1, 4),
    "age": (18, 80),
    "existing_credits": (1, 4),
    "people_liable": (1, 2),
}

tab_decide, tab_ask, tab_model = st.tabs(
    ["Valuta una richiesta", "Domande sull'AI Act", "Il modello"]
)

# ------------------------------------------------------------------ richiesta
with tab_decide:
    c1, c2 = st.columns([1, 1])
    if c1.button("Carica un esempio rifiutato"):
        st.session_state["form"] = service.example(rejected=True)
    if c2.button("Carica un esempio approvato"):
        st.session_state["form"] = service.example(rejected=False)
    form = st.session_state.get("form") or service.example(seed=0)
    options = service.options()

    with st.form("application"):
        cols = st.columns(3)
        values = {}
        for i, feat in enumerate(NUMERIC_FEATURES + CATEGORICAL_FEATURES):
            label = config.FEATURE_LABELS.get(feat, feat).capitalize()
            col = cols[i % 3]
            if feat in NUMERIC_FEATURES:
                lo, hi = NUMERIC_LIMITS[feat]
                values[feat] = col.number_input(
                    label, min_value=lo, max_value=hi, value=min(max(int(form[feat]), lo), hi)
                )
            else:
                opts = options[feat]
                values[feat] = col.selectbox(label, opts, index=opts.index(form[feat]))
        submitted = st.form_submit_button("Valuta", type="primary")

    if submitted:
        with st.spinner("Calcolo decisione, spiegazione e controfattuali..."):
            res = service.explain(values)

        if res["approved"]:
            st.success(f"Richiesta APPROVATA · probabilità di insolvenza {res['prob_default']:.0%}")
        else:
            st.error(
                f"Richiesta NON APPROVATA · probabilità di insolvenza {res['prob_default']:.0%}"
            )
        st.caption(
            f"Soglia di rifiuto {res['threshold']:.0%} (scelta per minimizzare il costo atteso). "
            f"Id decisione: {res['decision_id']}"
        )

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Spiegazione per il cliente")
            st.write(res["narration"]["text"])
            st.caption(f"Generata da: {res['narration']['provider']}")

            from credit_xai.evaluation.faithfulness import check

            verdict = check(
                {"approved": res["approved"], "contributions": res["contributions"]},
                res["narration"],
            )
            if verdict["faithful"]:
                st.info("Controllo di fedeltà superato: la spiegazione è coerente con il modello.")
            else:
                st.warning(
                    f"Controllo di fedeltà NON superato: {json.dumps(verdict, ensure_ascii=False)}"
                )

            if res["counterfactuals"]:
                st.subheader("Cosa potrebbe cambiare l'esito")
                for i, cf in enumerate(res["counterfactuals"], 1):
                    changes = "; ".join(
                        f"**{c['label']}**: {c['from']} → {c['to']}" for c in cf["changes"]
                    )
                    st.markdown(f"{i}. {changes} (rischio stimato {cf['prob_default']:.0%})")

        with right:
            st.subheader("Contributo dei fattori (SHAP)")
            top = pd.DataFrame(res["contributions"][:10])[::-1]
            fig, ax = plt.subplots(figsize=(6, 4.5))
            colors = ["#c0392b" if v > 0 else "#2471a3" for v in top["shap"]]
            ax.barh(top["label"] + " = " + top["value"], top["shap"], color=colors)
            ax.axvline(0, color="grey", lw=0.8)
            ax.set_xlabel("← riduce il rischio | aumenta il rischio →")
            fig.tight_layout()
            st.pyplot(fig)

# ------------------------------------------------------------------ RAG
with tab_ask:
    st.write(
        "Domande sugli obblighi normativi dei sistemi di credit scoring "
        "(AI Act, GDPR, direttiva sul credito ai consumatori)."
    )
    q = st.text_input("Domanda", "Ho diritto a una spiegazione se mi rifiutano il credito?")
    if st.button("Chiedi") and q:
        res = rag.answer(q)
        st.write(res["answer"])
        with st.expander("Fonti recuperate"):
            for s in res["sources"]:
                st.markdown(f"**{s['title']}** (score {s['score']})")
        st.caption("Sintesi non ufficiali, non è consulenza legale.")

# ------------------------------------------------------------------ modello
with tab_model:
    st.subheader("Metriche sul test set")
    if config.METRICS_FILE.exists():
        metrics = json.loads(config.METRICS_FILE.read_text())
        table = pd.DataFrame(
            {k: {m: v for m, v in d.items() if not isinstance(v, dict)} for k, d in metrics.items()}
        )
        st.dataframe(table, width="stretch")
    faith = config.REPORTS_DIR / "faithfulness.json"
    if faith.exists():
        st.subheader("Fedeltà delle spiegazioni")
        st.json(json.loads(faith.read_text()))
    report = config.PROJECT_ROOT / "docs" / "fairness_report.md"
    if report.exists():
        with st.expander("Fairness report"):
            st.markdown(report.read_text(encoding="utf-8"))
    card = config.PROJECT_ROOT / "docs" / "model_card.md"
    if card.exists():
        with st.expander("Model card"):
            st.markdown(card.read_text(encoding="utf-8"))
