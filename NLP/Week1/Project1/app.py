"""Neurova NLP — language detection then Arabic or English sentiment."""

import streamlit as st

from src.pipeline import NeurovaNLPPipeline
from src.validation import EmptyTextError

st.set_page_config(page_title="Neurova NLP", page_icon="💬", layout="centered")

DEMOS = {
    "English · Positive": "I absolutely loved this movie.",
    "English · Negative": "This was one of the worst movies I've ever watched.",
    "Arabic · Positive": "الخدمة ممتازة والتجربة كانت رائعة",
    "Arabic · Negative": "الخدمة سيئة جدا ولن أكرر التجربة",
}


@st.cache_resource
def load_pipeline() -> NeurovaNLPPipeline:
    return NeurovaNLPPipeline.load()


st.markdown(
    "<h1 style='text-align:center;margin-bottom:0'>Neurova NLP</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:#888'>Arabic / English sentiment</p>",
    unsafe_allow_html=True,
)

try:
    nlp = load_pipeline()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if "draft" not in st.session_state:
    st.session_state.draft = ""

st.text_area("Enter your text", key="draft", height=140)

left, mid, right = st.columns([1, 1, 1])
with mid:
    clicked = st.button("Analyze", type="primary", use_container_width=True)

st.caption("Demo examples")
demo_cols = st.columns(4)
for col, (label, sample) in zip(demo_cols, DEMOS.items()):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state.draft = sample
            st.rerun()

if clicked:
    try:
        result = nlp.analyze(st.session_state.draft)
    except (EmptyTextError, TypeError) as exc:
        st.warning(str(exc))
    else:
        st.divider()
        st.markdown(f"**User Text:** {result['User Text']}")
        st.markdown(f"**Language:** {result['Language']}")
        st.markdown(f"**Sentiment Classification:** {result['Sentiment Classification']}")
