import streamlit as st
import uuid
from datetime import datetime

# --- Simple In-Memory Storage (for demo only) ---
EVALS = {}

st.set_page_config(page_title="Evalsify MVP", page_icon="✅", layout="centered")
st.title("Evalsify MVP — Run, Score, Share")

st.write("### 1. Enter Your Prompt")
prompt = st.text_area("Prompt", placeholder="Type your prompt here...")

st.write("### 2. Expected Behavior / Criteria")
expected = st.text_area("Expected output or description")

st.write("### 3. Model Output (Paste for now)")
output = st.text_area("Paste model output here")

st.write("### 4. Score the Output")
col1, col2, col3 = st.columns(3)
with col1:
    score_correctness = st.slider("Correctness", 1, 5, 3)
with col2:
    score_tone = st.slider("Tone", 1, 5, 3)
with col3:
    score_clarity = st.slider("Clarity", 1, 5, 3)

notes = st.text_area("Reviewer Notes")

if st.button("Save & Generate Share Link"):
    eval_id = str(uuid.uuid4())[:8]
    EVALS[eval_id] = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "expected": expected,
        "output": output,
        "scores": {
            "correctness": score_correctness,
            "tone": score_tone,
            "clarity": score_clarity
        },
        "notes": notes
    }
    share_url = f"?view={eval_id}"
    st.success("Evaluation saved!")
    st.markdown(f"**Share this link:** {st.experimental_get_query_params().get('base_url', [''])[0]}{share_url}")

# --- View Mode ---
params = st.experimental_get_query_params()
if "view" in params:
    eval_id = params["view"][0]
    if eval_id in EVALS:
        e = EVALS[eval_id]
        st.header("Shared Evaluation")
        st.write(f"**Prompt:** {e['prompt']}")
        st.write(f"**Expected:** {e['expected']}")
        st.write(f"**Output:** {e['output']}")
        st.write("### Scores")
        st.write(e['scores'])
        st.write("### Notes")
        st.write(e['notes'])
    else:
        st.error("Evaluation not found.")
