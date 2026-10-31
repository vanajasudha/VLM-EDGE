"""Evaluation View: Manifest validation, ground-truth metrics, and qualitative review."""
from html import escape
from pathlib import Path
import pandas as pd
import streamlit as st
from src import ui
from src.benchmark_metrics import binary_metrics
from src.evaluation_manifest import load_manifest, ManifestError

ROOT = Path(__file__).resolve().parents[2]


def render_evaluation():
    ui.header("Evaluation & Verification", "Model performance validation against annotated reference data.", "Quality Assurance")

    t1, t2, t3 = st.tabs(["Ground Truth Accuracy", "Manifest Validation", "Description Review Protocol"])

    with t1:
        ui.section_title("Quantitative Verification (Binary Questions)", "Ground Truth Metrics")
        ui.callout(
            "Evaluation Status: The physical warehouse dataset has not completed final human labelling. "
            "To evaluate model predictions, upload a verified ground-truth CSV with columns: "
            "<code>frame_id</code>, <code>question</code>, <code>ground_truth</code>, <code>prediction</code> (yes/no).",
            "info",
        )

        up = st.file_uploader("Upload Annotated Evaluation CSV", type="csv", key="eval_csv_uploader")
        if up:
            try:
                df = pd.read_csv(up)
                if not {"ground_truth", "prediction"} <= set(df.columns):
                    raise ValueError("Uploaded file must contain 'ground_truth' and 'prediction' columns.")

                m = binary_metrics(df["ground_truth"], df["prediction"], invalid_policy="count")

                c1, c2, c3, c4 = st.columns(4)
                ui.kpi(c1, "Accuracy", f"{m['accuracy']:.2f}", f"n = {m['n']}")
                ui.kpi(c2, "Precision", f"{m['precision']:.2f}", f"TP: {m['tp']} | FP: {m['fp']}")
                ui.kpi(c3, "Recall", f"{m['recall']:.2f}", f"TP: {m['tp']} | FN: {m['fn']}")
                ui.kpi(c4, "F1 Score", f"{m['f1']:.2f}", f"Invalid: {m['n_invalid']}/{m['n']}")

                st.caption(
                    f"Ambiguous or unparseable responses: {m['n_invalid']}/{m['n']} (scored as incorrect under strict policy). "
                    f"Metrics reflect uploaded predictions only; no model weights were re-evaluated."
                )

                st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
                ui.section_title("Binary Confusion Matrix")
                conf_matrix = pd.DataFrame(
                    [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
                    index=["Reference: No", "Reference: Yes"],
                    columns=["Predicted: No", "Predicted: Yes"],
                )
                st.dataframe(conf_matrix, width="stretch")

            except Exception as exc:
                ui.callout(f"Could not calculate evaluation metrics: {exc}", "bad")
        else:
            ui.empty_state("No evaluation results file loaded. Upload a verified CSV above to compute precision, recall, and accuracy.")

    with t2:
        ui.section_title("Dataset Manifest Validator", "Schema & Hash Integrity")
        st.markdown(
            f'<div style="font-size:0.84rem;color:{ui.C["muted"]};margin-bottom:0.8rem;">'
            f'  Validates CSV dataset manifests against the VLM-Edge evaluation schema version 1. Checks image byte hashes, '
            f'  verifies split exclusivity (train/validation/test), and confirms required annotation fields.'
            f'</div>',
            unsafe_allow_html=True,
        )

        template_path = ROOT / "evaluation/manifest_template.csv"
        manifest_path = st.text_input("Manifest File Path", value=str(template_path), key="manifest_path_input")
        split_opt = st.selectbox("Dataset Split", ["test", "validation", "train"], key="manifest_split_opt")

        if st.button("Validate Manifest", type="primary", key="btn_validate_manifest"):
            try:
                p = Path(manifest_path)
                if not p.is_file():
                    st.error(f"File not found: {p}")
                else:
                    dataset = load_manifest(p)
                    selected = dataset.select(split_opt)
                    st.success(f"Manifest is VALID. Total samples: {len(dataset.samples)} | Selected split '{split_opt}': {len(selected)}")
                    
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.markdown(f"**Manifest SHA256:** `{dataset.manifest_sha256[:16]}...`")
                    with info_col2:
                        st.markdown(f"**Dataset SHA256:** `{dataset.dataset_sha256[:16]}...`")
            except ManifestError as exc:
                ui.callout(f"Manifest Validation Failed: {exc}", "bad")
            except Exception as exc:
                ui.callout(f"Manifest could not be checked: {exc}", "bad")

    with t3:
        ui.section_title("Qualitative Description Review Protocol", "Human Adjudication")
        st.markdown(
            f"""
            <div class="card">
                <div style="font-weight:700;font-size:0.92rem;color:{ui.C['text']};margin-bottom:0.4rem;">
                    Four-Pillar Qualitative Review Criteria
                </div>
                <div style="color:{ui.C['muted']};font-size:0.84rem;line-height:1.55;">
                    Open-ended visual descriptions cannot be scored by pure binary metrics. The VLM-Edge protocol requires 
                    double-blind human review using the template at <code>evaluation/description_review_template.csv</code>:
                </div>
                <ul style="color:{ui.C['muted']};font-size:0.83rem;line-height:1.6;margin-top:0.5rem;padding-left:1.2rem;">
                    <li><strong>Factual Correctness (1-5):</strong> Are the objects, positions, and activities described physically present?</li>
                    <li><strong>Scene Coverage (1-5):</strong> Did the model capture the salient warehouse foreground or omit key machinery/personnel?</li>
                    <li><strong>Hallucination Severity (None / Minor / Severe):</strong> Did the model fabricate non-existent obstacles, hazards, or personnel?</li>
                    <li><strong>Uncertainty Calibration:</strong> Did the model express appropriate epistemic caution for distant or occluded regions?</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )