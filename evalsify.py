# Requirements (put in requirements.txt):
# streamlit>=1.35,<2
# pandas>=2.0
# sqlalchemy>=2.0
# aiosqlite
# openai>=1.30
# altair>=5.0
# python-dateutil
# stripe>=6
#
# Optional: If not using OpenAI, you can stub judge() and model_infer() to return dummy data.

import os
import uuid
import json
import time
from datetime import datetime, UTC
from typing import Dict, Any, List, Optional

import pandas as pd
import altair as alt
import streamlit as st
from dateutil import tz
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# -----------------------------
# Config / Secrets
# -----------------------------
OPENAI_API_KEY_DEFAULT = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
STRIPE_SECRET_KEY = st.secrets.get("STRIPE_SECRET_KEY") or os.getenv("STRIPE_SECRET_KEY")
STRIPE_PRICE_LOOKUP = st.secrets.get("STRIPE_PRICE_LOOKUP")  # optional mapping template_id->price_id (JSON string)
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
              usage_json TEXT,
              human_scores_json TEXT,
              human_total_score REAL,
              human_notes TEXT,
              agreed_bool INTEGER
            );
            """
        )
            # Best-effort migrations for new Week2 columns (SQLite allows ADD COLUMN)
        try:
            conn.exec_driver_sql("ALTER TABLE run_item ADD COLUMN human_scores_json TEXT;")
        except Exception:
            pass
        try:
            conn.exec_driver_sql("ALTER TABLE run_item ADD COLUMN human_total_score REAL;")
        except Exception:
            pass
        try:
            conn.exec_driver_sql("ALTER TABLE run_item ADD COLUMN human_notes TEXT;")
        except Exception:
            pass
        try:
            conn.exec_driver_sql("ALTER TABLE run_item ADD COLUMN agreed_bool INTEGER;")
        except Exception:
            pass
        # Ensure entitlement table exists (idempotent)
        try:
            conn.exec_driver_sql("""
                CREATE TABLE IF NOT EXISTS entitlement (
                  id TEXT PRIMARY KEY,
                  email TEXT,
                  template_id TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
            """)
        except Exception:
            pass
    return engine


def now_iso() -> str:
    # timezone-aware UTC timestamp (avoid deprecation)
    return datetime.now(UTC).isoformat()


# -----------------------------
# Model & Judge
# -----------------------------
from openai import OpenAI

def get_openai_client() -> Optional[OpenAI]:
    # Per-user key via sidebar input; fallback to app secret; else None (stub mode)
    key = st.session_state.get("user_openai_key") or OPENAI_API_KEY_DEFAULT
    if key:
        try:
            return OpenAI(api_key=key)
        except Exception:
            return None
    return None

client = get_openai_client()


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
# Utility helpers (Week2)
# -----------------------------

def compute_weighted_total(criteria: List[Dict[str, Any]], scores: Dict[str, Any]) -> float:
    total, wsum = 0.0, 0.0
    for c in criteria:
        name = c.get("name"); w = float(c.get("weight", 1) or 1)
        s = float(scores.get(name, 0) or 0)
        total += w * s; wsum += w
    return total / wsum if wsum else 0.0


def get_rubric_for_run(engine: Engine, run_id: str) -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        row = conn.execute(text("SELECT rubric.json FROM run JOIN rubric ON run.rubric_id=rubric.id WHERE run.id=:i"), {"i": run_id}).fetchone()
    return json.loads(row[0]) if row else []


def agreement_bool(judge_total: float, human_total: float, tolerance: float = 0.5) -> int:
    try:
        return 1 if abs(float(judge_total) - float(human_total)) <= tolerance else 0
    except Exception:
        return 0

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
    # invalidate cached project list
    try:
        list_projects_df.clear()
    except Exception:
        pass
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
st.set_page_config(page_title="Evalsify — Week3 MVP", page_icon="✅", layout="wide")
engine = get_engine()

st.title("Evalsify — Week3 MVP (Batch • Judge • Human Review • Templates)")

# Shareable report mode (?report=<run_id>)
params = st.query_params
if "report" in params:
    _report_val = params.get("report")
    report_run_id = _report_val[0] if isinstance(_report_val, list) else _report_val
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

# Sidebar — API key + Project picker / creator
with st.sidebar:
    with st.expander("🔑 API Keys", expanded=False):
        st.caption("Use your own OpenAI key for inference/judging. Not stored server-side — session only.")
        user_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.get("user_openai_key", ""))
        if st.button("Use this key"):
            st.session_state["user_openai_key"] = user_key.strip() or None
            st.success("Key set for this session.")
            # Recreate client with new key
            client = get_openai_client()
            st.rerun()
    st.header("Project")
    projects = list_projects_df()
    if projects.empty:
        pname = st.text_input("Create project name", value="My First Project")
        if st.button("Create Project"):
            pid = create_project(engine, pname)
            st.session_state["current_project_id"] = pid
            st.rerun()
        st.stop()
    else:
        id_list = projects["id"].tolist()
        name_list = projects["name"].tolist()
        current_id = st.session_state.get("current_project_id", (id_list[0] if id_list else None))
        try:
            default_index = id_list.index(current_id) if current_id in id_list else 0
        except Exception:
            default_index = 0
        selected_name = st.selectbox("Select project", name_list, index=default_index)
        project_id = projects.loc[projects["name"] == selected_name, "id"].iloc[0]
        st.session_state["current_project_id"] = project_id
        st.caption(f"Project ID: {project_id}")

st.markdown("---")

# Tabs: Run | Review | Reports | Compare | Templates
_tab_names = ["Run Batch", "Review Items", "Reports", "Compare Runs", "Templates"]
run_tab, review_tab, reports_tab, compare_tab, templates_tab = st.tabs(_tab_names)

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
            colx, coly = st.columns([1,2])
            with colx:
                if st.button("Open report here"):
                    # Navigate by updating query params
                    st.query_params["report"] = run_id
                    st.rerun()
            with coly:
                st.text_input("Share querystring", value=f"?report={run_id}", help="Append this to your app URL", label_visibility="visible")

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

        # Metrics: reviewed count, agreement, calibrated average
        reviewed = items[items["human_total_score"].notna()]
        reviewed_count = len(reviewed)
        agreement = reviewed["agreed_bool"].mean() if reviewed_count else float("nan")
        # Calibrated average: human where available, else judge
        calibrated_series = items.apply(lambda r: r["human_total_score"] if pd.notna(r["human_total_score"]) else r["total_score"], axis=1)
        calibrated_avg = float(calibrated_series.mean()) if not items.empty else 0.0

        colA, colB, colC = st.columns(3)
        colA.metric("Reviewed items", f"{reviewed_count}")
        colB.metric("Agreement (±0.5)", f"{(agreement*100):.0f}%" if reviewed_count else "—")
        colC.metric("Calibrated Avg", f"{calibrated_avg:.2f}")

        st.write("**Run Summary**")
        st.dataframe(runs_df[runs_df["id"]==chosen_id][["id","model","status","avg_score","cost_usd","started_at","finished_at"]], use_container_width=True)

        st.markdown("### Reviewer Queue")
        # Next item without human score
        pending = items[items["human_total_score"].isna()].head(1)
        if pending.empty:
            st.success("All items reviewed for this run!")
        else:
            row = pending.iloc[0]
            st.write(f"**Item #{int(row['row_index'])}**")
            st.write("**Input**")
            st.code(row["input"])
            if isinstance(row["expected"], str) and row["expected"]:
                st.write("**Expected**")
                st.code(row["expected"])
            st.write("**Model Output**")
            st.code(row["output"])
            st.write("**Judge rationale**")
            st.text(row.get("judge_rationale", ""))

            criteria = get_rubric_for_run(engine, chosen_id)
            st.write("**Human Scores (1–5)**")
            human_scores = {}
            for c in criteria:
                nm = c.get("name")
                human_scores[nm] = st.slider(nm, 1, 5, 3)
            human_notes = st.text_area("Reviewer notes (optional)")

            if st.button("Save Human Review"):
                h_total = compute_weighted_total(criteria, human_scores)
                agree = agreement_bool(row["total_score"], h_total, tolerance=0.5)
                # persist
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE run_item SET human_scores_json=:hs, human_total_score=:ht, human_notes=:hn, agreed_bool=:ab
                        WHERE id=:id
                    """), {
                        "hs": json.dumps(human_scores), "ht": float(h_total), "hn": human_notes, "ab": int(agree), "id": row["id"]
                    })
                st.success(f"Saved. Human total: {h_total:.2f} (agree: {'yes' if agree else 'no'})")
                st.rerun()

        st.markdown("### All Items")
        display_cols = ["row_index","total_score","human_total_score","agreed_bool","input","expected","output","judge_rationale"]
        show_cols = [c for c in display_cols if c in items.columns]
        st.dataframe(items[show_cols], use_container_width=True)

with reports_tab:
    st.subheader("Reports & Charts")
    runs_df = list_runs(engine, project_id)
    if runs_df.empty:
        st.info("No runs yet.")
    else:
        # Judge average over time
        chart1 = alt.Chart(runs_df).mark_line(point=True).encode(
            x=alt.X("started_at:T", title="Run Time"),
            y=alt.Y("avg_score:Q", title="Judge Avg Score"),
            color=alt.Color("model:N", title="Model"),
            tooltip=["id","model","avg_score","cost_usd","started_at"]
        ).properties(height=280, title="Judge Average over Time")
        st.altair_chart(chart1, use_container_width=True)

        # Calibrated averages (computed on the fly per run)
        cal_rows = []
        for _, r in runs_df.iterrows():
            items = get_run_items(engine, r["id"])
            if items.empty:
                cal = None; agree = None; reviewed = 0
            else:
                calibrated = items.apply(lambda x: x["human_total_score"] if pd.notna(x["human_total_score"]) else x["total_score"], axis=1)
                cal = float(calibrated.mean()) if len(calibrated) else None
                reviewed = int(items[items["human_total_score"].notna()].shape[0])
                agree = float(items["agreed_bool"].mean()) if reviewed else None
            cal_rows.append({"run_id": r["id"], "model": r["model"], "started_at": r["started_at"], "calibrated_avg": cal, "reviewed": reviewed, "agreement": agree})
        cal_df = pd.DataFrame(cal_rows)
        if not cal_df.empty:
            chart2 = alt.Chart(cal_df).mark_line(point=True).encode(
                x=alt.X("started_at:T", title="Run Time"),
                y=alt.Y("calibrated_avg:Q", title="Calibrated Avg Score"),
                color=alt.Color("model:N"),
                tooltip=["run_id","model","calibrated_avg","reviewed","agreement"]
            ).properties(height=280, title="Calibrated Average over Time")
            st.altair_chart(chart2, use_container_width=True)
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

with templates_tab:
    st.subheader("Template Gallery")
    st.caption("Import a starter pack: creates a dataset + rubric (or whole project) in your current workspace. Paid templates can be purchased via Stripe (if configured) or unlocked with a demo code.")

    # Buyer identity (optional for demos, used to persist entitlements)
    st.markdown("#### Buyer identity (optional)")
    buyer_email = st.text_input("Email for purchases/unlocks (to look up past entitlements)", value=st.session_state.get("buyer_email", ""))
    if buyer_email != st.session_state.get("buyer_email"):
        st.session_state["buyer_email"] = buyer_email
        st.write(
            "You are now logged in with this email in your current workspace. Paid templates can be purchased via Stripe (if configured) or unlocked with a demo code."
        )


    # -----------------------------
    # Templates loading (folder, index, upload, fallback)
    # -----------------------------
    def load_templates_from_path(path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "templates" in data:
                return data["templates"]
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def load_templates_folder(folder: str):
        items = []
        index_path = os.path.join(folder, "index.json")
        if os.path.exists(index_path):
            # If index exists, it can reference files or embed templates
            idx = load_templates_from_path(index_path)
            if isinstance(idx, list):
                for ent in idx:
                    if isinstance(ent, str) and ent.endswith('.json'):
                        items.extend(load_templates_from_path(os.path.join(folder, ent)) or [])
                    elif isinstance(ent, dict):
                        items.append(ent)
        # Also load any loose *.json files in folder
        try:
            for fn in sorted(os.listdir(folder)):
                if fn.endswith('.json') and fn != 'index.json':
                    items.extend(load_templates_from_path(os.path.join(folder, fn)) or [])
        except Exception:
            pass
        return items

    # Built-in fallback
    SAMPLE_TEMPLATES = [
        {
            "id": "summ_faith_v1",
            "name": "Summarization Faithfulness (News)",
            "description": "Evaluate if summaries preserve key facts from news articles.",
            "is_paid": False,
            "price_usd": 0,
            "rubric": [
                {"name": "faithfulness", "desc": "No invented facts; aligns with source.", "weight": 2},
                {"name": "coverage", "desc": "Captures key points.", "weight": 1},
                {"name": "clarity", "desc": "Readable and concise.", "weight": 1}
            ],
            "dataset_rows": [
                {"input": "Article: OpenAI announced new features for its platform.", "expected": "Summary mentions the features and release timeline."},
                {"input": "Article: The city council approved a $5M budget for parks.", "expected": "Summary includes vote outcome and main allocations."}
            ]
        },
        {
            "id": "rag_acc_v1",
            "name": "RAG Answer Accuracy (Q&A)",
            "description": "Score groundedness of answers versus provided context.",
            "is_paid": True,
            "price_usd": 29,
            "rubric": [
                {"name": "groundedness", "desc": "Supported by the provided context.", "weight": 2},
                {"name": "completeness", "desc": "Answers all parts of the question.", "weight": 1},
                {"name": "precision", "desc": "Avoids hallucinated details.", "weight": 1}
            ],
            "dataset_rows": [
                {"input": "Context: Warranty period is 2 years. Question: What is the warranty period?", "expected": "Answer states 2 years."},
                {"input": "Context: The event venue is Hall B, Building 3. Question: Where is the venue?", "expected": "Answer: Hall B, Building 3."}
            ],
            "stripe_price_id": "price_XXXX_optional"
        }
    ]

    # Load order: uploaded -> templates/ folder -> repo templates.json -> fallback
    repo_templates_root = load_templates_from_path(os.path.join(os.getcwd(), "templates.json"))
    folder_templates = load_templates_folder(os.path.join(os.getcwd(), "templates"))
    up_json = st.file_uploader("Upload templates.json (list or {\"templates\": [...]})", type=["json"], key="tpl_uploader")
    uploaded_templates = []
    if up_json is not None:
        try:
            uploaded_templates = json.load(up_json)
            if isinstance(uploaded_templates, dict) and "templates" in uploaded_templates:
                uploaded_templates = uploaded_templates["templates"]
            if not isinstance(uploaded_templates, list):
                uploaded_templates = []
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            uploaded_templates = []

    TEMPLATES = uploaded_templates or folder_templates or repo_templates_root or SAMPLE_TEMPLATES
    st.caption(f"Loaded {len(TEMPLATES)} template(s) — source: " + ("uploaded" if uploaded_templates else ("templates/" if folder_templates else ("repo templates.json" if repo_templates_root else "built-in samples"))))

    # -----------------------------
    # Stripe purchase flow (optional)
    # -----------------------------
    try:
        import stripe  # type: ignore
        stripe.api_key = STRIPE_SECRET_KEY if STRIPE_SECRET_KEY else None
    except Exception:
        stripe = None

    def start_checkout(price_id: str, template_id: str):
        if not (stripe and STRIPE_SECRET_KEY):
            st.error("Stripe not configured. Set STRIPE_SECRET_KEY in Secrets.")
            return None
        # Build a success URL pointing back to this app
        try:
            current_url = st.experimental_get_url()  # Streamlit 1.35
        except Exception:
            current_url = ""
        success_url = (current_url or "").split('#')[0]
        if "?" in success_url:
            success_url += f"&entitle={template_id}"
        else:
            success_url += f"?entitle={template_id}"
        cancel_url = success_url
        try:
            sess = stripe.checkout.Session.create(
                mode="payment",
                line_items=[{"price": price_id, "quantity": 1}],
                success_url=success_url + "&session_id={CHECKOUT_SESSION_ID}",
                cancel_url=cancel_url,
            )
            return sess.url
        except Exception as e:
            st.error(f"Stripe error: {e}")
            return None

    # Entitlements stored in session for MVP (plus persisted in SQLite when email present)
    if "entitlements" not in st.session_state:
        st.session_state["entitlements"] = set()

    # Helpers for entitlement persistence
    def save_entitlement(engine: Engine, template_id: str, email: Optional[str]):
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO entitlement (id, email, template_id, created_at)
                VALUES (:i,:e,:t,:c)
            """), {"i": str(uuid.uuid4())[:8], "e": (email or None), "t": template_id, "c": now_iso()})

    def has_entitlement_db(engine: Engine, template_id: str, email: Optional[str]) -> bool:
        if not email:
            return False
        with engine.begin() as conn:
            row = conn.execute(text("SELECT 1 FROM entitlement WHERE template_id=:t AND email=:e LIMIT 1"), {"t": template_id, "e": email}).fetchone()
        return bool(row)

    # If redirected back with entitlement
    q = st.query_params
    entitle = q.get("entitle")
    sess_id = q.get("session_id")
    if entitle and sess_id and stripe and STRIPE_SECRET_KEY:
        try:
            sess = stripe.checkout.Session.retrieve(sess_id)
            if sess and sess.get("payment_status") == "paid":
                st.session_state["entitlements"].add(entitle)
                # derive email from Stripe session if available
                email_from_stripe = None
                try:
                    email_from_stripe = (sess.get("customer_details") or {}).get("email")
                except Exception:
                    email_from_stripe = None
                email_final = email_from_stripe or st.session_state.get("buyer_email")
                if email_final:
                    save_entitlement(engine, entitle, email_final)
                st.success(f"Purchase confirmed. Unlocked template '{entitle}'.")
        except Exception as e:
            st.warning(f"Could not verify Stripe session: {e}")

    def has_entitlement(tpl: dict) -> bool:
        if not tpl.get("is_paid"):
            return True
        # Session unlock or DB entitlement by email, or unlock code if Stripe missing
        session_ok = tpl.get("id") in st.session_state.get("entitlements", set())
        email_ok = has_entitlement_db(engine, tpl.get("id"), st.session_state.get("buyer_email"))
        if session_ok or email_ok:
            return True
        code_ok = False
        if STRIPE_SECRET_KEY is None:
            code_key = f"code_{tpl.get('id','x')}"
            code_val = st.text_input("Enter unlock code (demo)", type="password", key=code_key)
            code_ok = st.secrets.get("TEMPLATE_UNLOCK_CODE") and code_val == st.secrets.get("TEMPLATE_UNLOCK_CODE")
        return bool(code_ok)

    def import_template(engine: Engine, project_id: str, tpl: dict):
        # Create rubric
        rid = save_rubric(engine, project_id, tpl.get("name", "Template") + " — Rubric", tpl.get("rubric", []), version=1)
        # Create dataset
        rows = tpl.get("dataset_rows") or tpl.get("dataset") or []
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["input","expected"]) 
        did = upload_dataset(engine, project_id, tpl.get("name", "Template") + " — Dataset", df)
        return rid, did

    def create_project_from_template(engine: Engine, tpl: dict) -> str:
        pid = create_project(engine, tpl.get("name", "New Project"))
        rid, did = import_template(engine, pid, tpl)
        return pid

    # Render cards
    for tpl in TEMPLATES:
        with st.container(border=True):
            st.markdown(f"### {tpl.get('name','Untitled Template')}")
            cols = st.columns([3,1])
            with cols[0]:
                st.write(tpl.get("description", ""))
                st.write("Rubric:")
                st.json(tpl.get("rubric", []), expanded=False)
            with cols[1]:
                price = tpl.get("price_usd", 0)
                tag = "Paid" if tpl.get("is_paid") else "Free"
                st.metric(tag, f"${price}")

            # Actions
            colA, colB, colC = st.columns([1,1,2])
            entitled = has_entitlement(tpl)
            proj_btn = colA.button(f"Create project from template", key=f"proj_{tpl.get('id', str(uuid.uuid4())[:8])}", disabled=tpl.get("is_paid") and not entitled)
            imp_btn = colB.button(f"Import into current project", key=f"imp_{tpl.get('id', str(uuid.uuid4())[:8])}", disabled=tpl.get("is_paid") and not entitled)
            if proj_btn:
                new_pid = create_project_from_template(engine, tpl)
                st.success(f"Project created: {new_pid}. Open it from the sidebar.")
            if imp_btn:
                rid, did = import_template(engine, project_id, tpl)
                st.success(f"Imported rubric {rid} and dataset {did} into project {project_id}.")

            # Purchase if needed
            if tpl.get("is_paid") and not entitled:
                price_id = tpl.get("stripe_price_id")
                if price_id and stripe and STRIPE_SECRET_KEY:
                    if colC.button("Purchase via Stripe", key=f"buy_{tpl.get('id','x')}"):
                        url = start_checkout(price_id, tpl.get("id","tpl"))
                        if url:
                            st.markdown(f"[Open Checkout]({url})")
                else:
                    st.caption("Unlock code demo is active (no Stripe price configured). Set TEMPLATE_UNLOCK_CODE in Secrets.")
