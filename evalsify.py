# Requirements (put in requirements.txt):
# streamlit>=1.35
# pandas>=2.0
# sqlalchemy>=2.0
# aiosqlite
# openai>=1.30
# altair>=5.0
# python-dateutil
#
# Optional: If not using OpenAI, you can stub judge() and model_infer() to return dummy data.

import os
import uuid
import json
import time
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import altair as alt
import streamlit as st
from dateutil import tz
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# -----------------------------
# Config / Secrets
# -----------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
DEFAULT_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
]
LOCAL_TZ = tz.gettz("America/New_York")

# -----------------------------
# DB helpers (SQLite)
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    # Streamlit Cloud persists file storage across reruns while the app lives
    db_path = os.path.join(os.getcwd(), "evalsify.db")
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS project (
              id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS rubric (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL,
              name TEXT NOT NULL,
              json TEXT NOT NULL,
              version INTEGER NOT NULL,
              created_at TEXT NOT NULL
            );
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS dataset (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL,
              name TEXT NOT NULL,
              meta_json TEXT,
              created_at TEXT NOT NULL
            );
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS run (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL,
              dataset_id TEXT NOT NULL,
              rubric_id TEXT NOT NULL,
              model TEXT NOT NULL,
              params_json TEXT,
              status TEXT NOT NULL,
              started_at TEXT NOT NULL,
              finished_at TEXT,
              cost_usd REAL DEFAULT 0,
              avg_score REAL DEFAULT 0
            );
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS run_item (
              id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              row_index INTEGER,
              input TEXT,
              expected TEXT,
              output TEXT,
              judge_scores_json TEXT,
              judge_rationale TEXT,
              total_score REAL,
              usage_json TEXT
            );
            """
        )
    return engine


def now_iso() -> str:
    return datetime.utcnow().isoformat()


# -----------------------------
# Model & Judge
# -----------------------------
if OPENAI_API_KEY:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None


def model_infer(model: str, prompt: str) -> Dict[str, Any]:
    """Call the tested model. Returns {text, usage}.
    If no API key, return a stubbed response for demo.
    """
    if not client:
        time.sleep(0.1)
        return {"text": f"[STUB OUTPUT for: {prompt[:60]}…]", "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0}}

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text_out = resp.choices[0].message.content if resp.choices else ""
        usage = getattr(resp, "usage", None)
        usage_obj = {
            "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
            "cost_usd": 0,  # You can post-compute with your own pricing table per model
        }
        return {"text": text_out, "usage": usage_obj}
    except Exception as e:
        return {"text": f"[ERROR: {e}]", "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0}}


JUDGE_SYSTEM_PROMPT = (
    "You are an evaluation judge. Score the MODEL_OUTPUT against the EXPECTED_BEHAVIOR using the CRITERIA.\n"
    "Return STRICT JSON with fields: {scores: {<criterion>: number 1-5}, rationale: string}. No prose."
)


def judge_output(model: str, output: str, expected: str, criteria: List[Dict[str, Any]]):
    """LLM-as-judge. Returns (scores_dict, rationale_str, usage)."""
    criteria_list = [
        {"name": c.get("name"), "desc": c.get("desc", ""), "weight": c.get("weight", 1)}
        for c in criteria
    ]
    judge_prompt = (
        "CRITERIA: " + json.dumps(criteria_list) + "\n" +
        "EXPECTED_BEHAVIOR: " + (expected or "") + "\n" +
        "MODEL_OUTPUT: " + (output or "") + "\n" +
        "Return JSON only."
    )

    if not client:
        # Stub judge: give neutral scores
        scores = {c["name"]: 3 for c in criteria_list}
        return scores, "[STUB RATIONALE]", {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0}

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0,
        )
        text_out = resp.choices[0].message.content if resp.choices else "{}"
        usage = getattr(resp, "usage", None)
        usage_obj = {
            "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
            "cost_usd": 0,
        }
        try:
            payload = json.loads(text_out)
            scores = payload.get("scores", {})
            rationale = payload.get("rationale", "")
        except Exception:
            scores = {}
            rationale = text_out
        return scores, rationale, usage_obj
    except Exception as e:
        return {}, f"[JUDGE ERROR: {e}]", {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0}


# -----------------------------
# UI Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def list_projects_df() -> pd.DataFrame:
    eng = get_engine()
    with eng.begin() as conn:
        df = pd.read_sql("SELECT * FROM project ORDER BY created_at DESC", conn)
    return df


def create_project(engine: Engine, name: str) -> str:
    pid = str(uuid.uuid4())[:8]
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO project (id,name,created_at) VALUES (:i,:n,:t)"),
                     {"i": pid, "n": name, "t": now_iso()})
    list_projects.clear()
    return pid


def save_rubric(engine: Engine, project_id: str, name: str, criteria: List[Dict[str, Any]], version: int = 1) -> str:
    rid = str(uuid.uuid4())[:8]
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO rubric (id, project_id, name, json, version, created_at)
            VALUES (:i,:p,:n,:j,:v,:t)
        """), {"i": rid, "p": project_id, "n": name, "j": json.dumps(criteria), "v": version, "t": now_iso()})
    return rid


def list_rubrics(engine: Engine, project_id: str) -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT * FROM rubric WHERE project_id=:p ORDER BY created_at DESC"), conn, params={"p": project_id})
    return df


def upload_dataset(engine: Engine, project_id: str, name: str, df: pd.DataFrame) -> str:
    did = str(uuid.uuid4())[:8]
    # Persist CSV to disk for simplicity
    path = os.path.join(os.getcwd(), f"dataset_{did}.csv")
    df.to_csv(path, index=False)
    meta = {"path": path, "rows": len(df), "columns": list(df.columns)}
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO dataset (id, project_id, name, meta_json, created_at)
            VALUES (:i,:p,:n,:m,:t)
        """), {"i": did, "p": project_id, "n": name, "m": json.dumps(meta), "t": now_iso()})
    return did


def list_datasets(engine: Engine, project_id: str) -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT * FROM dataset WHERE project_id=:p ORDER BY created_at DESC"), conn, params={"p": project_id})
    return df


def read_dataset_meta(engine: Engine, dataset_id: str) -> Dict[str, Any]:
    with engine.begin() as conn:
        row = conn.execute(text("SELECT meta_json FROM dataset WHERE id=:i"), {"i": dataset_id}).fetchone()
    return json.loads(row[0]) if row else {}


def create_run(engine: Engine, project_id: str, dataset_id: str, rubric_id: str, model: str, params: Dict[str, Any]) -> str:
    rid = str(uuid.uuid4())[:8]
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO run (id, project_id, dataset_id, rubric_id, model, params_json, status, started_at)
            VALUES (:i,:p,:d,:r,:m,:j,'running',:t)
        """), {"i": rid, "p": project_id, "d": dataset_id, "r": rubric_id, "m": model, "j": json.dumps(params), "t": now_iso()})
    return rid


def complete_run(engine: Engine, run_id: str, avg_score: float, cost_usd: float):
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE run SET status='finished', finished_at=:f, avg_score=:s, cost_usd=:c WHERE id=:i
        """), {"f": now_iso(), "s": float(avg_score or 0), "c": float(cost_usd or 0), "i": run_id})


def save_run_item(engine: Engine, run_id: str, row_index: int, input_text: str, expected: str, output_text: str,
                  judge_scores: Dict[str, Any], judge_rationale: str, total_score: float, usage: Dict[str, Any]):
    iid = str(uuid.uuid4())[:8]
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO run_item (id, run_id, row_index, input, expected, output, judge_scores_json, judge_rationale, total_score, usage_json)
            VALUES (:i,:r,:ri,:in,:ex,:out,:js,:jr,:ts,:u)
        """), {
            "i": iid, "r": run_id, "ri": row_index,
            "in": input_text, "ex": expected, "out": output_text,
            "js": json.dumps(judge_scores or {}), "jr": judge_rationale or "",
            "ts": float(total_score or 0), "u": json.dumps(usage or {})
        })


def list_runs(engine: Engine, project_id: str) -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT * FROM run WHERE project_id=:p ORDER BY started_at DESC"), conn, params={"p": project_id})
    return df


def get_run_items(engine: Engine, run_id: str) -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT * FROM run_item WHERE run_id=:r ORDER BY row_index ASC"), conn, params={"r": run_id})
    return df


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Evalsify — Week1 MVP", page_icon="✅", layout="wide")
engine = get_engine()

st.title("Evalsify — Week1 MVP (Batch • Judge • Compare)")

# Shareable report mode (?report=<run_id>)
params = st.query_params
if "report" in params:
    report_run_id = params["report"][0]
    st.subheader("Shared Report (Read-only)")
    with engine.begin() as conn:
        run_row = conn.execute(text("SELECT * FROM run WHERE id=:i"), {"i": report_run_id}).fetchone()
    if not run_row:
        st.error("Run not found.")
        st.stop()
    run_df = pd.DataFrame([dict(run_row._mapping)])
    items_df = get_run_items(engine, report_run_id)
    st.write("**Run Info**")
    st.write(run_df[["id","model","status","started_at","finished_at","avg_score","cost_usd"]])
    st.write("**Results**")
    st.dataframe(items_df[["row_index","total_score","input","expected","output","judge_rationale"]], use_container_width=True)
    st.stop()

# Sidebar — Project picker / creator
with st.sidebar:
    st.header("Project")
    projects = list_projects_df()
    if projects.empty:
        pname = st.text_input("Create project name", value="My First Project")
        if st.button("Create Project"):
            pid = create_project(engine, pname)
            st.rerun()
        st.stop()
    else:
        selected_name = st.selectbox("Select project", projects["name"].tolist())
        project_id = projects.loc[projects["name"] == selected_name, "id"].iloc[0]
        st.caption(f"Project ID: {project_id}")

st.markdown("---")

# Tabs: Run | Review | Reports | Compare
run_tab, review_tab, reports_tab, compare_tab = st.tabs(["Run Batch", "Review Items", "Reports", "Compare Runs"])

with run_tab:
    st.subheader("Upload Dataset & Define Rubric")

    # Upload CSV
    up = st.file_uploader("Upload CSV with columns: input[, expected]", type=["csv"])
    dataset_id = None
    df = None
    if up is not None:
        df = pd.read_csv(up)
        if "input" not in df.columns:
            st.error("CSV must include an 'input' column.")
        else:
            st.write(df.head())
            dataset_name = st.text_input("Dataset name", value=f"dataset-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            if st.button("Save Dataset"):
                dataset_id = upload_dataset(engine, project_id, dataset_name, df)
                st.success(f"Saved dataset {dataset_id}")

    # Existing datasets
    dsets = list_datasets(engine, project_id)
    dsel = None
    if not dsets.empty:
        dsel_name = st.selectbox("Or pick existing dataset", dsets["name"].tolist())
        dsel = dsets.loc[dsets["name"] == dsel_name, "id"].iloc[0]

    st.markdown("### Rubric")
    default_criteria = [
        {"name": "correctness", "desc": "Is the answer factually correct and addresses the prompt?", "weight": 1},
        {"name": "clarity", "desc": "Is the answer clear and well-structured?", "weight": 1},
        {"name": "tone", "desc": "Is the tone appropriate to the task?", "weight": 1},
    ]
    crit_json = st.text_area("Criteria JSON (list of {name, desc, weight})", value=json.dumps(default_criteria, indent=2), height=180)
    rubric_name = st.text_input("Rubric name", value="Default Rubric v1")
    rubric_id = None
    if st.button("Save Rubric"):
        try:
            criteria = json.loads(crit_json)
            rubric_id = save_rubric(engine, project_id, rubric_name, criteria, version=1)
            st.success(f"Saved rubric {rubric_id}")
        except Exception as e:
            st.error(f"Rubric JSON error: {e}")

    # Pick saved rubric if desired
    rubs = list_rubrics(engine, project_id)
    rsel = None
    if not rubs.empty:
        rsel_name = st.selectbox("Or pick existing rubric", rubs["name"].tolist())
        rsel = rubs.loc[rubs["name"] == rsel_name, "id"].iloc[0]

    st.markdown("### Model & Run")
    model = st.selectbox("Model under test", DEFAULT_MODELS)
    max_rows = st.number_input("Max rows to run (for quick demos)", min_value=1, value=20)

    if st.button("Run Batch"):
        # Resolve dataset & rubric ids
        dataset_to_use = dataset_id or dsel
        rubric_to_use = rubric_id or rsel
        if not dataset_to_use or not rubric_to_use:
            st.error("Please provide a dataset and a rubric (save new or select existing).")
        else:
            # Load dataset
            meta = read_dataset_meta(engine, dataset_to_use)
            data_df = pd.read_csv(meta["path"]) if meta and os.path.exists(meta["path"]) else pd.DataFrame()
            if data_df.empty:
                st.error("Dataset file missing or empty.")
                st.stop()
            if "input" not in data_df.columns:
                st.error("Dataset must include 'input' column.")
                st.stop()

            # Load rubric JSON
            with engine.begin() as conn:
                row = conn.execute(text("SELECT json FROM rubric WHERE id=:i"), {"i": rubric_to_use}).fetchone()
            criteria = json.loads(row[0]) if row else default_criteria

            run_id = create_run(engine, project_id, dataset_to_use, rubric_to_use, model, params={})
            st.info(f"Run started: {run_id}")
            prog = st.progress(0, text="Running…")
            total = min(len(data_df), int(max_rows))

            total_scores = []
            total_cost = 0.0
            for idx, row in data_df.head(total).iterrows():
                prompt = str(row.get("input", ""))
                expected = str(row.get("expected", "")) if "expected" in data_df.columns else ""

                infer = model_infer(model, prompt)
                output_text = infer["text"]
                usage = infer.get("usage", {})
                # Judge using same model or a fixed judge model
                judge_scores, judge_rationale, judge_usage = judge_output(model, output_text, expected, criteria)

                # Weighted total score (1-5 per criterion)
                total_score = 0.0
                weight_sum = 0.0
                for c in criteria:
                    name = c.get("name")
                    w = float(c.get("weight", 1))
                    s = float(judge_scores.get(name, 0))
                    total_score += w * s
                    weight_sum += w
                total_score = total_score / weight_sum if weight_sum else 0
                total_scores.append(total_score)

                # Accumulate costs if you add a pricing table
                total_cost += float(usage.get("cost_usd", 0) or 0) + float(judge_usage.get("cost_usd", 0) or 0)

                save_run_item(engine, run_id, int(idx), prompt, expected, output_text, judge_scores, judge_rationale, total_score, {"infer": usage, "judge": judge_usage})
                prog.progress(int(((len(total_scores))/total)*100), text=f"{len(total_scores)}/{total} items")

            avg_score = sum(total_scores)/len(total_scores) if total_scores else 0
            complete_run(engine, run_id, avg_score, total_cost)
            st.success(f"Run complete. Avg score: {avg_score:.2f}")
            st.write(f"**Share report link:** append `?report={run_id}` to your app URL")

with review_tab:
    st.subheader("Runs & Items")
    runs_df = list_runs(engine, project_id)
    if runs_df.empty:
        st.info("No runs yet. Create one in the Run tab.")
    else:
        run_name_opts = [f"{r['id']} — {r['model']} — {r['started_at']}" for _, r in runs_df.iterrows()]
        pick = st.selectbox("Pick a run", run_name_opts)
        chosen_id = runs_df.iloc[run_name_opts.index(pick)]["id"]
        items = get_run_items(engine, chosen_id)
        st.write("**Run Summary**")
        st.dataframe(runs_df[runs_df["id"]==chosen_id][["id","model","status","avg_score","cost_usd","started_at","finished_at"]], use_container_width=True)
        st.write("**Items**")
        st.dataframe(items[["row_index","total_score","input","expected","output","judge_rationale"]], use_container_width=True)

with reports_tab:
    st.subheader("Reports & Charts")
    runs_df = list_runs(engine, project_id)
    if runs_df.empty:
        st.info("No runs yet.")
    else:
        chart = alt.Chart(runs_df).mark_line(point=True).encode(
            x=alt.X("started_at:T", title="Run Time"),
            y=alt.Y("avg_score:Q", title="Average Score"),
            color=alt.Color("model:N", title="Model"),
            tooltip=["id","model","avg_score","cost_usd","started_at"]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        st.caption("Tip: Share any single run with '?report=<run_id>' appended to the URL.")

with compare_tab:
    st.subheader("Compare Two Runs")
    runs_df = list_runs(engine, project_id)
    if len(runs_df) < 2:
        st.info("Need at least two runs to compare.")
    else:
        left = st.selectbox("Run A", runs_df["id"])
        right = st.selectbox("Run B", runs_df["id"], index=1)
        a_items = get_run_items(engine, left)
        b_items = get_run_items(engine, right)

        # Simple comparison by total_score distribution
        a_items["which"] = "A"
        b_items["which"] = "B"
        both = pd.concat([a_items[["total_score","which"]], b_items[["total_score","which"]]])
        chart = alt.Chart(both).mark_boxplot().encode(
            x=alt.X("which:N", title="Run"),
            y=alt.Y("total_score:Q", title="Total Score (1-5)"),
        )
        st.altair_chart(chart, use_container_width=True)

        # Side-by-side averages
        a_avg = a_items["total_score"].mean() if not a_items.empty else 0
        b_avg = b_items["total_score"].mean() if not b_items.empty else 0
        col1, col2 = st.columns(2)
        col1.metric("Run A Avg", f"{a_avg:.2f}")
        col2.metric("Run B Avg", f"{b_avg:.2f}")

        st.caption("Future: align rows by a stable ID to compare per-item across runs.")
