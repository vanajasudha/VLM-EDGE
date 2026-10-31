"""AI Assistant Workspace: Camera-centric visual question-answering with Moondream2."""
from html import escape
from pathlib import Path
import streamlit as st
from src import model_inference, ui


def render_ai_assistant(frames):
    ui.header("AI Assistant", "Visual intelligence inspection workstation for industrial scenes.", "Moondream2 VLM")

    st.session_state.setdefault("qa_result", None)
    st.session_state.setdefault("qa_error", None)
    st.session_state.setdefault("qa_inputs", None)

    # Top info banner
    ui.callout("Answers are generated locally on the RTX 3050 (4 GB) GPU via selective 4-bit NF4 quantization. Initial request loads model weights into VRAM.", "info")

    left, right = st.columns([1, 1.25])
    selected = None
    image_id = None
    image_error = None

    with left:
        ui.section_title("Visual Evidence Source")
        ids = [f["frame_id"] for f in frames]
        fid = st.selectbox("Image", ids, key="qa_image") if ids else None
        selected = next((f for f in frames if f["frame_id"] == fid), None)

        if selected:
            try:
                image_id = model_inference.image_identity(selected["path"])
                source_label = f"Saved Image — {selected['filename']}"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">'
                    f'  <span style="font-size:0.75rem;font-weight:600;color:{ui.C["muted"]};">{escape(source_label)}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.image(selected["path"], width="stretch")
                with st.expander("Diagnostics", expanded=False):
                    st.caption(f"Frame ID: {selected['frame_id']} | Source: test_frames/{selected['filename']} | SHA256: {image_id[:12]}...")
            except Exception as exc:
                image_error = str(exc)
                st.error(f"Cannot use selected image: {exc}")
        else:
            st.warning("No saved images are available in the gallery.")

    with right:
        ui.section_title("AI Inspection Query")

        question = st.text_input(
            "Your question",
            placeholder="e.g. Is anything blocking the walkway?",
            key="qa_question",
        )

        inputs = (selected["path"] if selected else None, image_id, question)
        if st.session_state.qa_inputs != inputs:
            st.session_state.qa_result = None
            st.session_state.qa_error = None
            st.session_state.qa_inputs = inputs

        valid = selected is not None and image_id is not None and bool(question.strip())
        
        # Primary Ask button MUST be at.button[0] for automated testing compatibility
        submitted = st.button("Ask", disabled=not valid, type="primary")

        # Suggested Questions Selectbox for quick template insertion without adding competing button widgets
        suggestions = [
            "— Select example question —",
            "What is visible in this scene?",
            "Is a forklift visible?",
            "How many people are visible?",
            "Describe the activity in this area.",
            "Is the walkway visibly obstructed?",
        ]
        chosen_sug = st.selectbox("Template suggestions", suggestions, key="qa_template_sug", label_visibility="collapsed")
        if chosen_sug != "— Select example question —":
            if st.session_state.get("qa_question") != chosen_sug:
                st.session_state["qa_question"] = chosen_sug
                st.rerun()

        answer_area = st.empty()

        if submitted:
            # Clear before loading/inference: a failed retry must never display an old answer.
            st.session_state.qa_result = None
            st.session_state.qa_error = None
            answer_area.empty()
            try:
                if not valid:
                    raise ValueError(image_error or "Select a valid image and enter a question.")
                with st.spinner("Generating answer locally..."):
                    result = model_inference.infer(
                        model_inference.load_moondream(),
                        selected["path"],
                        question,
                    )
                st.session_state.qa_result = {
                    **result,
                    "image": fid,
                    "question": question,
                    "inputs": inputs,
                }
            except Exception as exc:
                st.session_state.qa_error = str(exc) or "The model request failed. Please try again."

        with answer_area.container():
            result = st.session_state.qa_result
            if st.session_state.qa_error:
                st.error(st.session_state.qa_error)
            elif result and result["inputs"] == inputs:
                st.markdown(
                    f'<div class="answer-panel">'
                    f'  <div class="answer-header">'
                    f'    <span class="answer-label">Inspection Assessment</span>'
                    f'    <span class="answer-meta">{result["duration_ms"]:.1f} ms</span>'
                    f'  </div>'
                    f'  <div class="answer-text">{escape(result["answer"])}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Image: {result['image']} | Question: {result['question']}")
                st.caption(f"Inference latency: {result['duration_ms']:.1f} ms (excludes model loading and queue wait)")
            else:
                ui.empty_state("Select an image, enter a question, and click Ask to begin AI inspection.")

        st.caption("Model answers may be incorrect or incomplete. One image cannot prove an area is safe.")