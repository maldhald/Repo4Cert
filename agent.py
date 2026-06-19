"""
agent.py -- Task Status Update Agent

Workflow:
  1. Reads the Excel task file from documents/
  2. Finds all tasks active today (Start <= today <= End)
  3. Prompts user for status + notes per task
  4. Updates Actual Start Date and Actual End Date based on status
  5. Saves to a new Excel file with today's date in the filename
  6. Emails a summary report via Gmail SMTP

Run with:
    streamlit run agent.py

Gmail setup:
  - Enable 2-Step Verification on your Google account
  - Create an App Password: myaccount.google.com/apppasswords
  - Use that App Password (not your regular password) in the app
"""

import os
import glob
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import streamlit as st
import openpyxl
from openpyxl.styles import PatternFill, Font
from copy import copy

# ── Constants ─────────────────────────────────────────────────────────────────
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "documents")
TODAY         = datetime.date.today()
DATE_STR      = TODAY.strftime("%Y-%m-%d")

STATUS_OPTIONS = ["Not Started", "In Progress", "Completed", "Done", "Blocked", "Delayed"]

# Status colour coding for Excel output
STATUS_COLORS = {
    "Not Started": "FFFFFF",   # white
    "In Progress": "DDEEFF",   # light blue
    "Completed":   "CCFFCC",   # light green
    "Done":        "CCFFCC",   # light green
    "Blocked":     "FFCCCC",   # light red
    "Delayed":     "FFF0CC",   # light orange
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Task Status Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0d1117; color: #e6edf3; }
    [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
    [data-testid="stHeader"] { background: transparent; }
    h1, h2, h3 { color: #e6edf3; font-family: 'Segoe UI', system-ui, sans-serif; }
    p, li, label { color: #8b949e; font-family: 'Segoe UI', system-ui, sans-serif; }

    .task-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #1f6feb;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
    }
    .task-title {
        font-size: 16px;
        font-weight: 700;
        color: #e6edf3;
        margin-bottom: 6px;
    }
    .task-meta {
        font-size: 13px;
        color: #8b949e;
        margin-bottom: 4px;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-high     { background: #ff6b6b22; color: #ff6b6b; border: 1px solid #ff6b6b44; }
    .badge-critical { background: #ff000022; color: #ff4444; border: 1px solid #ff444444; }
    .badge-medium   { background: #ffa50022; color: #ffa500; border: 1px solid #ffa50044; }
    .badge-low      { background: #00ff0022; color: #3fb950; border: 1px solid #3fb95044; }
    .summary-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
    }
    .summary-num { font-size: 32px; font-weight: 700; color: #58a6ff; }
    .summary-label { font-size: 13px; color: #8b949e; }
    .stButton > button {
        background: #1f6feb; color: #fff; border: none;
        border-radius: 8px; padding: 8px 20px;
        font-weight: 600; transition: background 0.2s;
    }
    .stButton > button:hover { background: #388bfd; }
    .status-ok  { color: #3fb950; font-size: 13px; }
    .status-err { color: #f85149; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_excel_file():
    """Find the first non-temp xlsx file in documents/."""
    files = glob.glob(os.path.join(DOCUMENTS_DIR, "*.xlsx"))
    files = [f for f in files if not os.path.basename(f).startswith("~$")]
    return files[0] if files else None


def parse_date(val):
    """Parse various date formats into a date object."""
    if val is None:
        return None
    if isinstance(val, datetime.datetime):
        return val.date()
    if isinstance(val, datetime.date):
        return val
    if isinstance(val, str):
        val = val.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                return datetime.datetime.strptime(val, fmt).date()
            except ValueError:
                continue
    return None


def load_tasks(filepath):
    """
    Load tasks from Excel. Returns (headers, tasks, col_index_map).
    Each task is a dict of {header: value}.
    """
    wb = openpyxl.load_workbook(filepath, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if not rows:
        return [], [], {}

    # Find header row
    headers = [str(c).strip() if c else "" for c in rows[0]]
    col_map = {h: i for i, h in enumerate(headers)}

    tasks = []
    for row in rows[1:]:
        if not any(c for c in row if c is not None):
            continue
        task = {headers[i]: (row[i] if i < len(row) else None) for i in range(len(headers))}
        tasks.append(task)

    return headers, tasks, col_map


def find_active_tasks(tasks, col_map, headers):
    """Return tasks active today or overdue (not yet completed)."""
    # Match column names precisely with fallback
    start_col = next((h for h in headers if h.strip() == "Start Target Date"), None)
    end_col   = next((h for h in headers if h.strip() == "End Target Date"), None)
    id_col    = next((h for h in headers if h.strip() == "Task ID"), None)

    # Fallback: use Task name as identifier if no Task ID column
    task_col  = next((h for h in headers if h.strip() == "Task"), None)
    if not id_col:
        id_col = task_col

    # Fallback flexible matching for date cols
    if not start_col:
        start_col = next((h for h in headers if "start" in h.lower() and "actual" not in h.lower()), None)
    if not end_col:
        end_col = next((h for h in headers if "end" in h.lower() and "actual" not in h.lower()), None)

    # Fallback flexible matching for date cols
    if not start_col:
        start_col = next((h for h in headers if "start" in h.lower() and "actual" not in h.lower()), None)
    if not end_col:
        end_col = next((h for h in headers if "end" in h.lower() and "actual" not in h.lower()), None)

    print(f"[Debug] start_col={start_col}, end_col={end_col}, id_col={id_col}")

    # Completed statuses - tasks with these are excluded from overdue
    DONE_STATUSES = ("Completed", "Done")

    active = []
    for task in tasks:
        start  = parse_date(task.get(start_col))
        end    = parse_date(task.get(end_col))
        status = str(task.get("Status", "")).strip()

        if start and end:
            is_active  = start <= TODAY <= end
            is_overdue = end < TODAY and status not in DONE_STATUSES
            if is_active or is_overdue:
                active.append(task)

    return active, start_col, end_col, id_col


def save_updated_excel(filepath, headers, tasks, updates):
    """
    Save updated tasks to a new Excel file with today's date in filename.
    updates = {task_id: {"Status": ..., "Notes": ..., "Actual Start Date": ..., "Actual End Date": ...}}
    """
    # Ensure Actual columns exist
    extra_cols = ["Actual Start Date", "Actual End Date", "Status", "Notes"]
    all_headers = list(headers)
    for col in extra_cols:
        if col not in all_headers:
            all_headers.append(col)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Tasks"

    # Write header row
    for ci, h in enumerate(all_headers, 1):
        cell = ws.cell(row=1, column=ci, value=h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1f6feb")

    # Write data rows
    id_col = next((h for h in all_headers if "task id" in h.lower() or h.lower() == "id"), None)

    for ri, task in enumerate(tasks, 2):
        task_id = str(task.get(id_col, "")).strip() if id_col else ""
        update  = updates.get(task_id, {})

        # Merge update into task
        merged = dict(task)
        if update:
            merged["Status"]            = update.get("Status", task.get("Status", ""))
            merged["Notes"]             = update.get("Notes", task.get("Notes", ""))
            merged["Actual Start Date"] = update.get("Actual Start Date", task.get("Actual Start Date", ""))
            merged["Actual End Date"]   = update.get("Actual End Date", task.get("Actual End Date", ""))

        status = str(merged.get("Status", "")).strip()
        fill_color = STATUS_COLORS.get(status, "FFFFFF")
        row_fill   = PatternFill("solid", fgColor=fill_color)

        for ci, h in enumerate(all_headers, 1):
            val  = merged.get(h, "")
            cell = ws.cell(row=ri, column=ci, value=val)
            if fill_color != "FFFFFF":
                cell.fill = row_fill

    # Auto-width columns
    for col in ws.columns:
        max_len = max((len(str(cell.value)) if cell.value else 0) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

    # Build output filename
    base    = os.path.splitext(os.path.basename(filepath))[0]
    outname = f"{base}_updated_{DATE_STR}.xlsx"
    outpath = os.path.join(OUTPUT_DIR, outname)
    wb.save(outpath)
    return outpath


# ── Session state ─────────────────────────────────────────────────────────────
if "updates"       not in st.session_state:
    st.session_state.updates = {}
if "current_idx"   not in st.session_state:
    st.session_state.current_idx = 0
if "done"          not in st.session_state:
    st.session_state.done = False
if "saved_path"    not in st.session_state:
    st.session_state.saved_path = None
if "active_tasks"  not in st.session_state:
    st.session_state.active_tasks = None
if "meta"          not in st.session_state:
    st.session_state.meta = {}
if "email_sent"    not in st.session_state:
    st.session_state.email_sent = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Task Status Agent")
    st.markdown(f"*Today: **{DATE_STR}***")
    st.divider()

    excel_file = find_excel_file()
    if excel_file:
        st.markdown(f'<p class="status-ok">✅ {os.path.basename(excel_file)}</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-err">❌ No Excel file found in documents/</p>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("""
    1. Finds tasks active **today**
    2. Asks you for status + notes
    3. Sets actual start/end dates
    4. Saves updated Excel file
    """)

    if st.session_state.active_tasks:
        total    = len(st.session_state.active_tasks)
        reviewed = len(st.session_state.updates)
        st.divider()
        st.markdown(f"**Progress: {reviewed}/{total} tasks reviewed**")
        st.progress(reviewed / total if total else 0)

    st.divider()
    if st.button("🔄 Restart Agent"):
        for key in ["updates", "current_idx", "done", "saved_path", "active_tasks", "meta"]:
            del st.session_state[key]
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("## 🤖 Task Status Update Agent")
st.markdown(f"Reviewing tasks active on **{DATE_STR}**")

if not excel_file:
    st.error("No Excel file found in the `documents/` folder. Please add your task sheet and refresh.")
    st.stop()

# Load tasks once
if st.session_state.active_tasks is None:
    headers, all_tasks, col_map = load_tasks(excel_file)
    active, start_col, end_col, id_col = find_active_tasks(all_tasks, col_map, headers)
    st.session_state.active_tasks = active
    st.session_state.meta = {
        "headers":   headers,
        "all_tasks": all_tasks,
        "col_map":   col_map,
        "start_col": start_col,
        "end_col":   end_col,
        "id_col":    id_col,
        "filepath":  excel_file,
    }

active_tasks = st.session_state.active_tasks
meta         = st.session_state.meta

if not active_tasks:
    st.info(f"No tasks are scheduled for today ({DATE_STR}). Check your Start Target and End Target columns.")
    st.stop()

# ── Summary row ───────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="summary-card"><div class="summary-num">{len(active_tasks)}</div><div class="summary-label">Tasks Active Today</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="summary-card"><div class="summary-num">{len(st.session_state.updates)}</div><div class="summary-label">Reviewed</div></div>', unsafe_allow_html=True)
with col3:
    remaining = len(active_tasks) - len(st.session_state.updates)
    st.markdown(f'<div class="summary-card"><div class="summary-num">{remaining}</div><div class="summary-label">Remaining</div></div>', unsafe_allow_html=True)

st.divider()

# ── Done state ────────────────────────────────────────────────────────────────
if st.session_state.done and st.session_state.saved_path:
    st.success(f"✅ All tasks reviewed! Saved to: `{os.path.basename(st.session_state.saved_path)}`")

    id_col        = meta["id_col"]
    task_name_col = next((h for h in meta["headers"] if h.strip() == "Task"), "Task")
    ISSUE_STATUSES = ("Blocked", "Delayed", "Not Started")
    DONE_STATUSES  = ("Done", "Completed")

    # Build summary data
    all_rows    = []
    issue_rows  = []
    done_count  = 0
    issue_count = 0

    for task in active_tasks:
        tid    = str(task.get(id_col, "")).strip()
        update = st.session_state.updates.get(tid, {})
        status = update.get("Status", str(task.get("Status", "—")).strip())
        notes  = update.get("Notes", "")
        act_s  = update.get("Actual Start Date", "—")
        act_e  = update.get("Actual End Date", "—")
        tname  = str(task.get(task_name_col, "")).strip()
        priority = str(task.get("Priority", "")).strip()

        row = {"tid": tid, "task": tname, "status": status,
               "notes": notes, "act_s": act_s, "act_e": act_e, "priority": priority}
        all_rows.append(row)

        if status in DONE_STATUSES:
            done_count += 1
        if status in ISSUE_STATUSES:
            issue_rows.append(row)
            issue_count += 1

    # ── Summary cards ──────────────────────────────────────────────────────────
    st.markdown("### 📊 Session Summary")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown(f'<div class="summary-card"><div class="summary-num">{len(all_rows)}</div><div class="summary-label">Tasks Reviewed</div></div>', unsafe_allow_html=True)
    with sc2:
        st.markdown(f'<div class="summary-card"><div class="summary-num" style="color:#3fb950">{done_count}</div><div class="summary-label">Completed / Done</div></div>', unsafe_allow_html=True)
    with sc3:
        st.markdown(f'<div class="summary-card"><div class="summary-num" style="color:#f85149">{issue_count}</div><div class="summary-label">Issues (Blocked / Delayed)</div></div>', unsafe_allow_html=True)

    # ── Issue highlight ────────────────────────────────────────────────────────
    if issue_rows:
        st.markdown("### 🚨 Tasks Needing Attention")
        for r in issue_rows:
            badge_class = f"badge-{r['priority'].lower()}" if r['priority'].lower() in ["high","critical","medium","low"] else "badge-low"
            st.markdown(f"""
            <div class="task-card" style="border-left-color:#f85149">
                <div class="task-title">{r['tid']} — {r['task']}</div>
                <div class="task-meta">
                    <span class="badge {badge_class}">{r['priority']}</span>
                    Status: <strong style="color:#f85149">{r['status']}</strong>
                </div>
                <div class="task-meta">Notes: {r['notes'] or '—'}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Full task list ─────────────────────────────────────────────────────────
    st.markdown("### 📋 All Reviewed Tasks")
    for r in all_rows:
        priority    = r["priority"]
        badge_class = f"badge-{priority.lower()}" if priority.lower() in ["high","critical","medium","low"] else "badge-low"
        st.markdown(f"""
        <div class="task-card">
            <div class="task-title">{r['tid']} — {r['task']}</div>
            <div class="task-meta">
                <span class="badge {badge_class}">{priority}</span>
                Status: <strong>{r['status']}</strong> &nbsp;|&nbsp;
                Actual Start: <strong>{r['act_s']}</strong> &nbsp;|&nbsp;
                Actual End: <strong>{r['act_e']}</strong>
            </div>
            <div class="task-meta">Notes: {r['notes'] or '—'}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Email section ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📧 Send Summary Email")

    if st.session_state.email_sent:
        st.success("✅ Email sent successfully!")
    else:
        with st.form("email_form"):
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                sender_email = st.text_input("Your Gmail address", placeholder="you@gmail.com")
                app_password = st.text_input("Gmail App Password", type="password",
                    help="Not your regular password. Create one at myaccount.google.com/apppasswords")
            with col_e2:
                to_email    = st.text_input("Send to (recipient)", placeholder="manager@company.com")
                cc_email    = st.text_input("CC (optional)", placeholder="team@company.com")

            attach_excel = st.checkbox("Attach updated Excel file", value=True)
            send_btn     = st.form_submit_button("📤 Send Email", use_container_width=True)

        if send_btn:
            if not sender_email or not app_password or not to_email:
                st.error("Please fill in your Gmail, App Password, and recipient email.")
            else:
                # Build HTML email body
                issue_html = ""
                if issue_rows:
                    issue_html = "<h2 style='color:#cc0000'>🚨 Tasks Needing Attention</h2><table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;width:100%'>"
                    issue_html += "<tr style='background:#cc0000;color:white'><th>Task ID</th><th>Task</th><th>Status</th><th>Priority</th><th>Notes</th></tr>"
                    for r in issue_rows:
                        issue_html += f"<tr><td>{r['tid']}</td><td>{r['task']}</td><td><b>{r['status']}</b></td><td>{r['priority']}</td><td>{r['notes'] or '—'}</td></tr>"
                    issue_html += "</table><br>"

                all_html = "<h2>📋 Full Task Summary</h2><table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;width:100%'>"
                all_html += "<tr style='background:#0d9488;color:white'><th>Task ID</th><th>Task</th><th>Status</th><th>Priority</th><th>Actual Start</th><th>Actual End</th><th>Notes</th></tr>"
                for r in all_rows:
                    bg = "#fff0f0" if r["status"] in ISSUE_STATUSES else "#f0fff0" if r["status"] in DONE_STATUSES else "#ffffff"
                    all_html += f"<tr style='background:{bg}'><td>{r['tid']}</td><td>{r['task']}</td><td><b>{r['status']}</b></td><td>{r['priority']}</td><td>{r['act_s']}</td><td>{r['act_e']}</td><td>{r['notes'] or '—'}</td></tr>"
                all_html += "</table>"

                html_body = f"""
                <html><body style='font-family:Arial,sans-serif;color:#1e293b'>
                <h1 style='color:#0d9488'>📊 Task Status Update — {DATE_STR}</h1>
                <p><b>Reviewed:</b> {len(all_rows)} tasks &nbsp;|&nbsp;
                   <b style='color:green'>Done:</b> {done_count} &nbsp;|&nbsp;
                   <b style='color:red'>Issues:</b> {issue_count}</p>
                <hr>
                {issue_html}
                {all_html}
                <br><hr>
                <p style='color:#64748b;font-size:12px'>Generated by Task Status Agent · {DATE_STR}</p>
                </body></html>
                """

                try:
                    msg = MIMEMultipart("alternative")
                    msg["Subject"] = f"Task Status Update — {DATE_STR} ({issue_count} issues)"
                    msg["From"]    = sender_email
                    msg["To"]      = to_email
                    if cc_email.strip():
                        msg["Cc"] = cc_email.strip()

                    msg.attach(MIMEText(html_body, "html"))

                    # Attach Excel if requested
                    if attach_excel and st.session_state.saved_path and os.path.exists(st.session_state.saved_path):
                        with open(st.session_state.saved_path, "rb") as f:
                            part = MIMEBase("application", "octet-stream")
                            part.set_payload(f.read())
                        encoders.encode_base64(part)
                        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(st.session_state.saved_path)}")
                        msg.attach(part)

                    recipients = [to_email] + ([cc_email.strip()] if cc_email.strip() else [])
                    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                        server.login(sender_email, app_password)
                        server.sendmail(sender_email, recipients, msg.as_string())

                    st.session_state.email_sent = True
                    st.rerun()

                except smtplib.SMTPAuthenticationError:
                    st.error("Authentication failed. Make sure you are using a Gmail App Password, not your regular password.")
                except smtplib.SMTPException as e:
                    st.error(f"Email error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    st.stop()

# ── Task-by-task prompting ────────────────────────────────────────────────────
idx = st.session_state.current_idx

if idx >= len(active_tasks):
    # All tasks reviewed — save file
    if not st.session_state.done:
        saved = save_updated_excel(
            meta["filepath"],
            meta["headers"],
            meta["all_tasks"],
            st.session_state.updates,
        )
        st.session_state.saved_path = saved
        st.session_state.done       = True
        st.rerun()
else:
    task    = active_tasks[idx]
    id_col  = meta["id_col"]
    task_id = str(task.get(id_col, f"task_{idx}")).strip()

    # Detect column names
    task_name_col = next((h for h in meta["headers"] if h.strip() == "Task"), "Task")
    priority_col  = next((h for h in meta["headers"] if "priority" in h.lower()), "Priority")
    phase_col     = next((h for h in meta["headers"] if "phase" in h.lower()), "Phase")
    owner_col     = next((h for h in meta["headers"] if "owner" in h.lower()), "Owner")
    depends_col   = next((h for h in meta["headers"] if "depend" in h.lower()), None)
    est_col       = next((h for h in meta["headers"] if "estimate" in h.lower()), None)

    priority    = str(task.get(priority_col, "")).strip()
    badge_class = f"badge-{priority.lower()}" if priority.lower() in ["high","critical","medium","low"] else "badge-low"

    start_val = parse_date(task.get(meta["start_col"]))
    end_val   = parse_date(task.get(meta["end_col"]))

    st.markdown(f"### Task {idx + 1} of {len(active_tasks)}")
    st.markdown(f"""
    <div class="task-card">
        <div class="task-title">{task_id} — {task.get(task_name_col, "")}</div>
        <div class="task-meta">
            <span class="badge {badge_class}">{priority}</span>
            Phase: {task.get(phase_col, "—")} &nbsp;|&nbsp; Owner: {task.get(owner_col, "—")}
        </div>
        <div class="task-meta">
            📅 Planned: <strong>{start_val}</strong> → <strong>{end_val}</strong>
            &nbsp;|&nbsp; Est: {task.get(est_col, "—")} days
        </div>
        {f'<div class="task-meta">Depends on: {task.get(depends_col, "")}</div>' if depends_col else ""}
    </div>
    """, unsafe_allow_html=True)

    # Status + Notes form
    with st.form(key=f"form_{task_id}_{idx}"):
        status = st.selectbox(
            "Status",
            STATUS_OPTIONS,
            index=STATUS_OPTIONS.index(str(task.get("Status", "Not Started")).strip())
                  if str(task.get("Status", "Not Started")).strip() in STATUS_OPTIONS else 0,
        )
        notes = st.text_area(
            "Notes",
            value=str(task.get("Notes", "") or ""),
            placeholder="Add any notes, blockers, or comments...",
            height=100,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            submitted = st.form_submit_button("✅ Save & Next", use_container_width=True)
        with col_b:
            skipped = st.form_submit_button("⏭️ Skip", use_container_width=True)

    if submitted or skipped:
        if submitted:
            # Set actual dates based on status
            if status in ("Completed", "Done"):
                actual_start = str(start_val) if start_val else str(TODAY)
                actual_end   = str(TODAY)
            elif status in ("In Progress", "Delayed", "Blocked"):
                actual_start = str(start_val) if start_val else str(TODAY)
                actual_end   = ""
            elif status == "Not Started":
                actual_start = ""
                actual_end   = ""
            else:
                actual_start = ""
                actual_end   = ""

            st.session_state.updates[task_id] = {
                "Status":            status,
                "Notes":             notes,
                "Actual Start Date": actual_start,
                "Actual End Date":   actual_end,
            }

        st.session_state.current_idx += 1
        st.rerun()

    # Progress bar
    st.markdown("---")
    progress = idx / len(active_tasks)
    st.progress(progress, text=f"Progress: {idx}/{len(active_tasks)} tasks reviewed")
