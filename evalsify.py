# evalsify.py — Evalsify demo
# Features:
# - Projects, datasets, rubrics, runs (SQLite)
# - Batch run: CSV (input, expected) -> model outputs
# - LLM-as-judge with JSON rubric (criteria+weight); rationale
# - Human review queue; agreement; calibrated averages
# - Reports: judge avg over time; calibrated avg; export CSV/JSON
# - Compare runs (boxplot)
# - Templates gallery: load from templates/ folder or templates.json; import; create project
# - Paid templates (Stripe checkout MVP) or demo unlock code
# - Sidebar API key entry (per session) + fallback to app secret
# - Analytics: per-project summary, trends, reviewer progress
# - Profile: set display name, email; used for entitlements lookup
#
# Requirements (requirements.txt):
# streamlit>=1.35,<2
# pandas>=2.2
# sqlalchemy>=2.0
# aiosqlite>=0.20
# openai>=1.30
# altair>=5.3
# python-dateutil>=2.9
# stripe>=6
#
# runtime.txt:
# python-3.11

import os, json, uuid, time, io
from datetime import datetime, UTC
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
import altair as alt
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# -----------------------------
# Config / Secrets
# -----------------------------
OPENAI_API_KEY_DEFAULT = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
STRIPE_SECRET_KEY = st.secrets.get("STRIPE_SECRET_KEY") or os.getenv("STRIPE_SECRET_KEY")
TEMPLATE_UNLOCK_CODE = st.secrets.get("TEMPLATE_UNLOCK_CODE")
DEFAULT_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]

st.set_page_config(page_title="Evalsify — Week4 MVP", page_icon="✅", layout="wide")

# -----------------------------
# DB
# -----------------------------
SCHEMA_VERSION = 3

@st.cache_resource(show_spinner=False)
def get_engine(schema_version: int = SCHEMA_VERSION) -> Engine:
    db_path = os.getenv("EVALSIFY_DB_PATH", "evalsify_dev.db")
    #os.environ["EVALSIFY_DB_PATH"] = "evalsify_dev.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True)

    ddl = [
        "PRAGMA journal_mode=WAL",
        """
        CREATE TABLE IF NOT EXISTS project (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS dataset (
          id TEXT PRIMARY KEY,
          project_id TEXT NOT NULL,
          name TEXT NOT NULL,
          csv_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS rubric (
          id TEXT PRIMARY KEY,
          project_id TEXT NOT NULL,
          name TEXT NOT NULL,
          json TEXT NOT NULL,
          version INTEGER NOT NULL,
          created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS run (
          id TEXT PRIMARY KEY,
          project_id TEXT NOT NULL,
          dataset_id TEXT NOT NULL,
          rubric_id TEXT NOT NULL,
          model TEXT NOT NULL,
          status TEXT NOT NULL,
          avg_score REAL,
          cost_usd REAL,
          started_at TEXT NOT NULL,
          finished_at TEXT
        )
        """,
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
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS entitlement (
          id TEXT PRIMARY KEY,
          email TEXT,
          template_id TEXT NOT NULL,
          created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS profile (
          id INTEGER PRIMARY KEY CHECK (id=1),
          display_name TEXT,
          email TEXT
        )
        """,
    ]

    with engine.begin() as conn:
        for stmt in ddl:
            try:
                conn.exec_driver_sql(stmt)
            except Exception:
                pass
        row = conn.execute(text("SELECT 1 FROM profile WHERE id=1")).fetchone()
        if not row:
            conn.execute(text(
                "INSERT INTO profile (id, display_name, email) VALUES (1, 'Anonymous', '')"
            ))
    return engine


def now_iso() -> str:
    return datetime.now(UTC).isoformat()

# -----------------------------
# OpenAI client (optional)
# -----------------------------
from openai import OpenAI

def get_openai_client() -> Optional[OpenAI]:
    key = st.session_state.get("user_openai_key") or OPENAI_API_KEY_DEFAULT
    if key:
        try:
            return OpenAI(api_key=key)
        except Exception:
            return None
    return None

client = get_openai_client()

# -----------------------------
# Cached readers
# -----------------------------
@st.cache_data(show_spinner=False)
def list_projects_df() -> pd.DataFrame:
    eng = get_engine(SCHEMA_VERSION)

    with eng.begin() as conn:
        df = pd.read_sql("SELECT * FROM project ORDER BY created_at DESC", conn)
    return df

def create_project(engine: Engine, name: str) -> str:
    pid = str(uuid.uuid4())[:8]
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO project (id,name,created_at) VALUES (:i,:n,:t)"),
                     {"i": pid, "n": name, "t": now_iso()})
    try:
        list_projects_df.clear()
    except Exception:
        pass
    return pid

def upload_dataset(engine: Engine, project_id: str, name: str, df: pd.DataFrame) -> str:
    if not project_id:
        raise ValueError("No project selected. Please pick or create a project first.")

    # Guard: ensure table exists (defensive for Streamlit cache / first-run race)
    with engine.begin() as conn:
        conn.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS dataset (
          id TEXT PRIMARY KEY,
          project_id TEXT NOT NULL,
          name TEXT NOT NULL,
          csv_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        )
        """)    
    did = str(uuid.uuid4())[:8]
    payload = df.to_json(orient="records")
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO dataset (id, project_id, name, csv_json, created_at)
            VALUES (:i,:p,:n,:j,:t)
        """), {"i": did, "p": project_id, "n": name, "j": payload, "t": now_iso()})
    return did

def save_rubric(engine: Engine, project_id: str, name: str, criteria: List[Dict[str, Any]], version: int=1) -> str:
    rid = str(uuid.uuid4())[:8]
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO rubric (id, project_id, name, json, version, created_at)
            VALUES (:i,:p,:n,:j,:v,:t)
        """), {"i": rid, "p": project_id, "n": name, "j": json.dumps(criteria), "v": version, "t": now_iso()})
    return rid

def list_runs(engine: Engine, project_id: str) -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT * FROM run WHERE project_id=:p ORDER BY started_at DESC"),
                         conn, params={"p": project_id})
    return df

def get_dataset_df(engine: Engine, dataset_id: str) -> pd.DataFrame:
    with engine.begin() as conn:
        row = conn.execute(text("SELECT csv_json FROM dataset WHERE id=:i"), {"i": dataset_id}).fetchone()
    if not row:
        return pd.DataFrame(columns=["input","expected"])
    data = json.loads(row[0])
    return pd.DataFrame(data)

def get_rubric_for_run(engine: Engine, run_id: str) -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        row = conn.execute(text("SELECT rubric.json FROM run JOIN rubric ON run.rubric_id=rubric.id WHERE run.id=:i"),
                           {"i": run_id}).fetchone()
    return json.loads(row[0]) if row else []

def get_run_items(engine: Engine, run_id: str) -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT * FROM run_item WHERE run_id=:i ORDER BY row_index ASC"), conn, params={"i": run_id})
    return df

# -----------------------------
# Scoring helpers
# -----------------------------
def compute_weighted_total(criteria: List[Dict[str, Any]], scores: Dict[str, Any]) -> float:
    total, wsum = 0.0, 0.0
    for c in criteria:
        nm = c.get("name"); w = float(c.get("weight", 1) or 1)
        s = float(scores.get(nm, 0) or 0)
        total += w*s; wsum += w
    return total/wsum if wsum else 0.0

def agreement_bool(judge_total: float, human_total: float, tolerance: float=0.5) -> int:
    try:
        return 1 if abs(float(judge_total)-float(human_total)) <= tolerance else 0
    except Exception:
        return 0

# -----------------------------
# Model stubs
# -----------------------------
def model_infer(prompt: str, model: str) -> str:
    # Replace with real inference if you want to demo generation too.
    return f"[{model}] Response to: {prompt[:80]}..."

def judge_output(input_text: str, expected: str, output: str, rubric: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    """
    If OpenAI key is present, call a judge; else stub.
    Return: {scores: {crit:1-5}, rationale:str, total:float, usage:{}}
    """
    if client is None:
        scores = {c["name"]: 3 for c in rubric}
        total = compute_weighted_total(rubric, scores)
        return {"scores": scores, "rationale": "(stub) looks okay", "total": total, "usage": {}}

    # Simple judging prompt
    crit_block = "\n".join([f"- {c['name']}: {c.get('desc','')}" for c in rubric])
    sys = "You are a strict evaluation judge. Score each criterion from 1-5 and explain briefly."
    user = f"""INPUT:
{input_text}

EXPECTED (optional):
{expected or ""}

OUTPUT:
{output}

CRITERIA:
{crit_block}

Return JSON: {{"scores": {{"<criterion>": 1-5}}, "rationale": "<brief>", "total": "<float 1-5>"}}"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0
        )
        content = resp.choices[0].message.content
        # Try to parse JSON in response
        parsed = None
        try:
            start = content.find("{")
            end = content.rfind("}")
            parsed = json.loads(content[start:end+1]) if start!=-1 and end!=-1 else None
        except Exception:
            parsed = None
        if not parsed:
            scores = {c["name"]: 3 for c in rubric}
            total = compute_weighted_total(rubric, scores)
            return {"scores": scores, "rationale": content[:300], "total": total, "usage": getattr(resp, "usage", {}) or {}}
        scores = parsed.get("scores", {})
        total = float(parsed.get("total") or compute_weighted_total(rubric, scores))
        return {"scores": scores, "rationale": parsed.get("rationale",""), "total": total, "usage": getattr(resp, "usage", {}) or {}}
    except Exception as e:
        scores = {c["name"]: 3 for c in rubric}
        total = compute_weighted_total(rubric, scores)
        return {"scores": scores, "rationale": f"(error fallback) {e}", "total": total, "usage": {}}

# -----------------------------
# UI: Sidebar keys + project selection
# -----------------------------
with st.sidebar:
    with st.expander("🔑 API Keys", expanded=False):
        st.caption("Use your own OpenAI key (session only). If empty, judge uses a stub.")
        user_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.get("user_openai_key",""))
        if st.button("Use this key"):
            st.session_state["user_openai_key"] = user_key.strip() or None
            st.success("Key set for this session.")
            client = get_openai_client()
            st.rerun()

    st.header("Project")
    engine = get_engine(SCHEMA_VERSION)
    projects = list_projects_df()
    if projects.empty:
        pname = st.text_input("Create project name", value="My First Project")
        if st.button("Create Project"):
            pid = create_project(engine, pname)
            st.session_state["current_project_id"] = pid
            st.rerun()
        st.stop()
    else:
        id_list = projects["id"].tolist(); name_list = projects["name"].tolist()
        current_id = st.session_state.get("current_project_id", (id_list[0] if id_list else None))
        try:
            default_index = id_list.index(current_id) if current_id in id_list else 0
        except Exception:
            default_index = 0
        sel_name = st.selectbox("Select project", name_list, index=default_index)
        project_id = projects.loc[projects["name"]==sel_name,"id"].iloc[0]
        st.session_state["current_project_id"] = project_id
        st.caption(f"Project ID: {project_id}")

# -----------------------------
# Tabs
# -----------------------------
tab_names = ["Run Batch","Review Items","Reports","Compare Runs","Templates","Analytics","Profile"]
run_tab, review_tab, reports_tab, compare_tab, templates_tab, analytics_tab, profile_tab = st.tabs(tab_names)

# -----------------------------
# Run Batch
# -----------------------------
with run_tab:
    st.subheader("Create & Run")
    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload CSV with columns: input, expected (optional)", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            if "input" not in df.columns: st.error("CSV must contain an 'input' column."); st.stop()
            if "expected" not in df.columns: df["expected"] = ""
            dname = st.text_input("Dataset name", value=f"Dataset {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            if st.button("Save Dataset"):
                did = upload_dataset(engine, project_id, dname, df)
                st.success(f"Saved dataset {did} with {len(df)} rows.")
    with col2:
        st.write("Rubric (JSON list of {name, desc, weight})")
        default_rubric = json.dumps([
            {"name":"accuracy","desc":"Factually correct","weight":2},
            {"name":"clarity","desc":"Clear and readable","weight":1}
        ], indent=2)
        rubric_json = st.text_area("Rubric JSON", value=default_rubric, height=160)
        try:
            criteria = json.loads(rubric_json)
            assert isinstance(criteria, list)
        except Exception:
            criteria = []
        rname = st.text_input("Rubric name", value="Default Rubric v1")
        if st.button("Save Rubric"):
            rid = save_rubric(engine, project_id, rname, criteria, version=1)
            st.success(f"Saved rubric {rid}")

    st.markdown("---")
    # Launch a run
    with engine.begin() as conn:
        dsets = pd.read_sql(text("SELECT id,name,created_at FROM dataset WHERE project_id=:p ORDER BY created_at DESC"), conn, params={"p": project_id})
        rubs  = pd.read_sql(text("SELECT id,name,version FROM rubric WHERE project_id=:p ORDER BY created_at DESC"), conn, params={"p": project_id})
    if dsets.empty or rubs.empty:
        st.info("Create a dataset and a rubric first to run a batch.")
    else:
        dsel = st.selectbox("Dataset", [f"{r.id} — {r.name}" for r in dsets.itertuples()], key="run_d")
        rsel = st.selectbox("Rubric", [f"{r.id} — {r.name} v{r.version}" for r in rubs.itertuples()], key="run_r")
        model = st.selectbox("Judge Model", DEFAULT_MODELS, index=0)
        if st.button("Run Batch"):
            run_id = str(uuid.uuid4())[:8]
            did = dsel.split(" — ")[0]; rid = rsel.split(" — ")[0]
            # create run
            with engine.begin() as conn:
                conn.execute(text("""
                INSERT INTO run (id, project_id, dataset_id, rubric_id, model, status, started_at)
                VALUES (:i,:p,:d,:r,:m,'running',:t)
                """), {"i": run_id, "p": project_id, "d": did, "r": rid, "m": model, "t": now_iso()})
            st.info(f"Running… ({run_id})")
            data_df = get_dataset_df(engine, did)
            rubric_list = get_rubric_for_run(engine, run_id) or json.loads(rubric_json)
            items = []
            for idx, row in data_df.iterrows():
                input_text = str(row.get("input",""))
                expected = str(row.get("expected",""))
                output = model_infer(input_text, model)
                judged = judge_output(input_text, expected, output, rubric_list, model)
                row_id = str(uuid.uuid4())[:8]
                items.append({"id": row_id, "run_id": run_id, "row_index": int(idx), "input": input_text,
                              "expected": expected, "output": output,
                              "judge_scores_json": json.dumps(judged["scores"]),
                              "judge_rationale": judged["rationale"],
                              "total_score": float(judged["total"]), "usage_json": json.dumps(judged.get("usage",{}))})
                if idx % 10 == 0:
                    st.write(f"…processed {idx+1}/{len(data_df)}")
            # bulk insert
            with engine.begin() as conn:
                for it in items:
                    conn.execute(text("""
                        INSERT INTO run_item (id, run_id, row_index, input, expected, output, judge_scores_json, judge_rationale, total_score, usage_json)
                        VALUES (:id,:run_id,:row_index,:input,:expected,:output,:judge_scores_json,:judge_rationale,:total_score,:usage_json)
                    """), it)
                avg = float(pd.DataFrame(items)["total_score"].mean()) if items else None
                conn.execute(text("UPDATE run SET status='complete', avg_score=:a, finished_at=:f WHERE id=:i"),
                             {"a": avg, "f": now_iso(), "i": run_id})
            st.success(f"Run complete. Avg score: {avg:.2f}")
            colx, coly = st.columns([1,2])
            with colx:
                if st.button("Open report here"):
                    st.query_params["report"] = run_id
                    st.rerun()
            with coly:
                st.text_input("Share querystring", value=f"?report={run_id}", help="Append this to your app URL")

# -----------------------------
# Review Items
# -----------------------------
with review_tab:
    st.subheader("Runs & Items")
    runs_df = list_runs(engine, project_id)
    if runs_df.empty:
        st.info("No runs yet.")
    else:
        opts = [f"{r['id']} — {r['model']} — {r['started_at']}" for _, r in runs_df.iterrows()]
        pick = st.selectbox("Pick a run", opts)
        chosen_id = runs_df.iloc[opts.index(pick)]["id"]
        items = get_run_items(engine, chosen_id)
        reviewed = items[items["human_total_score"].notna()]
        reviewed_count = len(reviewed)
        agreement = reviewed["agreed_bool"].mean() if reviewed_count else float("nan")
        calibrated_series = items.apply(lambda r: r["human_total_score"] if pd.notna(r["human_total_score"]) else r["total_score"], axis=1)
        calibrated_avg = float(calibrated_series.mean()) if not items.empty else 0.0
        colA, colB, colC = st.columns(3)
        colA.metric("Reviewed items", f"{reviewed_count}")
        colB.metric("Agreement (±0.5)", f"{(agreement*100):.0f}%" if reviewed_count else "—")
        colC.metric("Calibrated Avg", f"{calibrated_avg:.2f}")

        st.markdown("### Reviewer Queue")
        pending = items[items["human_total_score"].isna()].head(1)
        if pending.empty:
            st.success("All items reviewed for this run!")
        else:
            row = pending.iloc[0]
            st.write(f"**Item #{int(row['row_index'])}**")
            st.write("**Input**"); st.code(row["input"])
            if isinstance(row["expected"], str) and row["expected"]:
                st.write("**Expected**"); st.code(row["expected"])
            st.write("**Model Output**"); st.code(row["output"])
            st.write("**Judge rationale**"); st.text(row.get("judge_rationale",""))
            criteria = get_rubric_for_run(engine, chosen_id)
            st.write("**Human Scores (1–5)**")
            human_scores = {}
            for c in criteria:
                nm = c.get("name"); human_scores[nm] = st.slider(nm, 1, 5, 3)
            human_notes = st.text_area("Reviewer notes (optional)")
            if st.button("Save Human Review"):
                h_total = compute_weighted_total(criteria, human_scores)
                agree = agreement_bool(row["total_score"], h_total, tolerance=0.5)
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE run_item SET human_scores_json=:hs, human_total_score=:ht, human_notes=:hn, agreed_bool=:ab
                        WHERE id=:id
                    """), {"hs": json.dumps(human_scores), "ht": float(h_total), "hn": human_notes, "ab": int(agree), "id": row["id"]})
                st.success(f"Saved. Human total: {h_total:.2f} (agree: {'yes' if agree else 'no'})")
                st.rerun()

        st.markdown("### All Items")
        display_cols = ["row_index","total_score","human_total_score","agreed_bool","input","expected","output","judge_rationale"]
        show_cols = [c for c in display_cols if c in items.columns]
        st.dataframe(items[show_cols], use_container_width=True)

# -----------------------------
# Reports
# -----------------------------
with reports_tab:
    st.subheader("Reports & Charts")
    runs_df = list_runs(engine, project_id)
    if runs_df.empty:
        st.info("No runs yet.")
    else:
        chart1 = alt.Chart(runs_df).mark_line(point=True).encode(
            x=alt.X("started_at:T", title="Run Time"),
            y=alt.Y("avg_score:Q", title="Judge Avg Score"),
            color=alt.Color("model:N", title="Model"),
            tooltip=["id","model","avg_score","cost_usd","started_at"]
        ).properties(height=280, title="Judge Average over Time")
        st.altair_chart(chart1, use_container_width=True)

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

        st.caption("Share any single run with '?report=<run_id>' appended to the URL.")

        st.markdown("### Export")
        # Concatenate all run items for export
        all_items = []
        for _, r in runs_df.iterrows():
            df_items = get_run_items(engine, r["id"])
            if not df_items.empty:
                df_items = df_items.assign(run_id=r["id"], model=r["model"], started_at=r["started_at"])
                all_items.append(df_items)
        if all_items:
            exp_df = pd.concat(all_items, ignore_index=True)
            csv_bytes = exp_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download all items (CSV)", csv_bytes, file_name="evalsify_items.csv")
            json_bytes = exp_df.to_json(orient="records").encode("utf-8")
            st.download_button("Download all items (JSON)", json_bytes, file_name="evalsify_items.json")
        else:
            st.info("No items to export yet.")

# -----------------------------
# Compare Runs
# -----------------------------
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
        a_items["which"] = "A"; b_items["which"] = "B"
        both = pd.concat([a_items[["total_score","which"]], b_items[["total_score","which"]]])
        chart = alt.Chart(both).mark_boxplot().encode(
            x=alt.X("which:N", title="Run"),
            y=alt.Y("total_score:Q", title="Total Score (1-5)")
        )
        st.altair_chart(chart, use_container_width=True)
        a_avg = a_items["total_score"].mean() if not a_items.empty else 0
        b_avg = b_items["total_score"].mean() if not b_items.empty else 0
        col1, col2 = st.columns(2)
        col1.metric("Run A Avg", f"{a_avg:.2f}"); col2.metric("Run B Avg", f"{b_avg:.2f}")
        st.caption("Future: align rows by a stable ID to compare per-item across runs.")

# -----------------------------
# Templates
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
    idx_path = os.path.join(folder, "index.json")
    if os.path.exists(idx_path):
        idx = load_templates_from_path(idx_path)
        if isinstance(idx, list):
            for ent in idx:
                if isinstance(ent, str) and ent.endswith(".json"):
                    items.extend(load_templates_from_path(os.path.join(folder, ent)) or [])
                elif isinstance(ent, dict):
                    items.append(ent)
    try:
        for fn in sorted(os.listdir(folder)):
            if fn.endswith(".json") and fn != "index.json":
                items.extend(load_templates_from_path(os.path.join(folder, fn)) or [])
    except Exception:
        pass
    return items

def import_template(engine: Engine, project_id: str, tpl: dict):
    rid = save_rubric(engine, project_id, tpl.get("name","Template")+" — Rubric", tpl.get("rubric", []), version=1)
    rows = tpl.get("dataset_rows") or tpl.get("dataset") or []
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["input","expected"])
    did = upload_dataset(engine, project_id, tpl.get("name","Template")+" — Dataset", df)
    return rid, did

def create_project_from_template(engine: Engine, tpl: dict) -> str:
    pid = create_project(engine, tpl.get("name","New Project"))
    import_template(engine, pid, tpl)
    return pid

with templates_tab:
    st.subheader("Template Gallery")
    st.caption("Import a starter pack: creates a dataset + rubric (or whole project). Paid templates via Stripe or demo unlock code.")
    # Buyer identity for entitlement persistence
    st.markdown("#### Buyer identity (optional)")
    buyer_email = st.text_input("Email for purchases/unlocks (to look up past entitlements)",
                                value=st.session_state.get("buyer_email",""))
    if buyer_email != st.session_state.get("buyer_email"):
        st.session_state["buyer_email"] = buyer_email

    repo_templates_root = load_templates_from_path(os.path.join(os.getcwd(),"templates.json"))
    folder_templates = load_templates_folder(os.path.join(os.getcwd(),"templates"))
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
            st.error(f"Invalid JSON: {e}"); uploaded_templates = []
    # Fallback samples
    SAMPLE_TEMPLATES = [{
        "id":"summ_faith_v1",
        "name":"Summarization Faithfulness (News)",
        "description":"Evaluate if summaries preserve key facts from news articles.",
        "is_paid":False, "price_usd":0,
        "rubric":[
            {"name":"faithfulness","desc":"No invented facts; aligns with source.","weight":2},
            {"name":"coverage","desc":"Captures key points.","weight":1},
            {"name":"clarity","desc":"Readable and concise.","weight":1}
        ],
        "dataset_rows":[
            {"input":"Article: OpenAI announced new features for its platform.","expected":"Summary mentions the features and release timeline."},
            {"input":"Article: The city council approved a $5M budget for parks.","expected":"Summary includes vote outcome and main allocations."}
        ]
    },{
        "id":"rag_acc_v1",
        "name":"RAG Answer Accuracy (Q&A)",
        "description":"Score groundedness of answers versus provided context.",
        "is_paid":True, "price_usd":29, "stripe_price_id":"price_XXXX_optional",
        "rubric":[
            {"name":"groundedness","desc":"Supported by the provided context.","weight":2},
            {"name":"completeness","desc":"Answers all parts of the question.","weight":1},
            {"name":"precision","desc":"Avoids hallucinated details.","weight":1}
        ],
        "dataset_rows":[
            {"input":"Context: Warranty period is 2 years. Question: What is the warranty period?","expected":"Answer states 2 years."},
            {"input":"Context: The event venue is Hall B, Building 3. Question: Where is the venue?","expected":"Answer: Hall B, Building 3."}
        ]
    }]
    TEMPLATES = uploaded_templates or folder_templates or repo_templates_root or SAMPLE_TEMPLATES
    st.caption(f"Loaded {len(TEMPLATES)} template(s) — source: " + ("uploaded" if uploaded_templates else ("templates/" if folder_templates else ("repo templates.json" if repo_templates_root else "built-in samples"))))

    # Stripe (optional)
    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY if STRIPE_SECRET_KEY else None
    except Exception:
        stripe = None

    # session entitlements
    if "entitlements" not in st.session_state:
        st.session_state["entitlements"] = set()

    def save_entitlement(engine: Engine, template_id: str, email: Optional[str]):
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO entitlement (id, email, template_id, created_at)
                VALUES (:i,:e,:t,:c)
            """), {"i": str(uuid.uuid4())[:8], "e": (email or None), "t": template_id, "c": now_iso()})

    def has_entitlement_db(engine: Engine, template_id: str, email: Optional[str]) -> bool:
        if not email: return False
        with engine.begin() as conn:
            row = conn.execute(text("SELECT 1 FROM entitlement WHERE template_id=:t AND email=:e LIMIT 1"),
                               {"t": template_id, "e": email}).fetchone()
        return bool(row)

    # handle checkout return
    q = st.query_params
    entitle = q.get("entitle"); sess_id = q.get("session_id")
    if entitle and sess_id and stripe and STRIPE_SECRET_KEY:
        try:
            sess = stripe.checkout.Session.retrieve(sess_id)
            if sess and sess.get("payment_status") == "paid":
                st.session_state["entitlements"].add(entitle)
                email_from_stripe = None
                try:
                    email_from_stripe = (sess.get("customer_details") or {}).get("email")
                except Exception:
                    pass
                email_final = email_from_stripe or st.session_state.get("buyer_email")
                if email_final:
                    save_entitlement(engine, entitle, email_final)
                st.success(f"Purchase confirmed. Unlocked template '{entitle}'.")
        except Exception as e:
            st.warning(f"Could not verify Stripe session: {e}")

    def has_entitlement(tpl: dict) -> bool:
        if not tpl.get("is_paid"): return True
        if tpl.get("id") in st.session_state.get("entitlements", set()):
            return True
        if has_entitlement_db(engine, tpl.get("id"), st.session_state.get("buyer_email")):
            return True
        # demo unlock when Stripe not configured
        if STRIPE_SECRET_KEY is None and TEMPLATE_UNLOCK_CODE:
            code_key = f"code_{tpl.get('id','x')}"
            val = st.text_input("Enter unlock code (demo)", type="password", key=code_key)
            if val == TEMPLATE_UNLOCK_CODE:
                return True
        return False

    def start_checkout(price_id: str, template_id: str):
        if not (stripe and STRIPE_SECRET_KEY):
            st.error("Stripe not configured. Set STRIPE_SECRET_KEY in Secrets.")
            return None
        try:
            current_url = st.experimental_get_url()
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

    # Search/filter
    squery = st.text_input("Search templates")
    for tpl in [t for t in TEMPLATES if (not squery or squery.lower() in t.get("name","").lower())]:
        with st.container(border=True):
            st.markdown(f"### {tpl.get('name','Untitled Template')}")
            cols = st.columns([3,1])
            with cols[0]:
                st.write(tpl.get("description",""))
                st.write("Rubric:"); st.json(tpl.get("rubric", []), expanded=False)
            with cols[1]:
                price = tpl.get("price_usd", 0); tag = "Paid" if tpl.get("is_paid") else "Free"
                st.metric(tag, f"${price}")
            entitled = has_entitlement(tpl)
            colA, colB, colC = st.columns([1,1,2])
            if colA.button("Create project from template", key=f"proj_{tpl.get('id',str(uuid.uuid4())[:8])}", disabled=tpl.get("is_paid") and not entitled):
                new_pid = create_project_from_template(engine, tpl)
                st.session_state["current_project_id"] = new_pid
                st.success(f"Project created: {new_pid}. Selecting it now…")
                st.rerun()
            if colB.button("Import into current project", key=f"imp_{tpl.get('id',str(uuid.uuid4())[:8])}", disabled=tpl.get("is_paid") and not entitled):
                rid, did = import_template(engine, project_id, tpl)
                st.success(f"Imported rubric {rid} and dataset {did} into project {project_id}.")
            if tpl.get("is_paid") and not entitled:
                price_id = tpl.get("stripe_price_id")
                if price_id and stripe and STRIPE_SECRET_KEY:
                    if colC.button("Purchase via Stripe", key=f"buy_{tpl.get('id','x')}"):
                        url = start_checkout(price_id, tpl.get("id","tpl"))
                        if url: st.markdown(f"[Open Checkout]({url})")
                else:
                    st.caption("Unlock code demo active (no Stripe price configured).")

# -----------------------------
# Analytics
# -----------------------------
with analytics_tab:
    st.subheader("Project Analytics")
    runs_df = list_runs(engine, project_id)
    if runs_df.empty:
        st.info("No runs yet.")
    else:
        # Summary metrics
        total_runs = len(runs_df)
        last_run = runs_df["started_at"].max()
        avg_judge = runs_df["avg_score"].mean()
        col1,col2,col3 = st.columns(3)
        col1.metric("Total runs", f"{total_runs}")
        col2.metric("Last run", f"{last_run}")
        col3.metric("Judge avg (all runs)", f"{avg_judge:.2f}" if pd.notna(avg_judge) else "—")

        # Reviewer progress
        reviewed_counts = []
        for _, r in runs_df.iterrows():
            it = get_run_items(engine, r["id"])
            n = len(it); rev = int(it["human_total_score"].notna().sum()) if not it.empty else 0
            reviewed_counts.append({"run_id": r["id"], "started_at": r["started_at"], "reviewed": rev, "total": n})
        prog_df = pd.DataFrame(reviewed_counts)
        if not prog_df.empty:
            chart = alt.Chart(prog_df).mark_bar().encode(
                x=alt.X("started_at:T", title="Run"),
                y=alt.Y("reviewed:Q", title="Reviewed items"),
                tooltip=["run_id","reviewed","total"]
            ).properties(height=250, title="Reviewer Progress")
            st.altair_chart(chart, use_container_width=True)

        # Export convenience for current project
        st.markdown("### Export (this project)")
        # Collect all items for project
        all_items = []
        for _, r in runs_df.iterrows():
            df_items = get_run_items(engine, r["id"])
            if not df_items.empty:
                df_items = df_items.assign(run_id=r["id"], model=r["model"], started_at=r["started_at"])
                all_items.append(df_items)
        if all_items:
            exp_df = pd.concat(all_items, ignore_index=True)
            csv_bytes = exp_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download project items (CSV)", csv_bytes, file_name=f"project_{project_id}_items.csv")
            json_bytes = exp_df.to_json(orient="records").encode("utf-8")
            st.download_button("Download project items (JSON)", json_bytes, file_name=f"project_{project_id}_items.json")
        else:
            st.info("No items to export yet.")

# -----------------------------
# Profile
# -----------------------------
with profile_tab:
    st.subheader("My Profile")
    with engine.begin() as conn:
        prof = conn.execute(text("SELECT display_name, email FROM profile WHERE id=1")).fetchone()
    disp = st.text_input("Display name", value=prof[0] if prof else "Anonymous")
    mail = st.text_input("Email", value=prof[1] if prof else "", help="Used to rehydrate template entitlements")
    if st.button("Save Profile"):
        with engine.begin() as conn:
            conn.execute(text("UPDATE profile SET display_name=:n, email=:e WHERE id=1"), {"n": disp, "e": mail})
        st.success("Profile updated!")
        st.session_state["buyer_email"] = mail

st.markdown("---")
st.caption("Evalsify Demo")
