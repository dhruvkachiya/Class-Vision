import pandas as pd
import os
import csv
from glob import glob

ATTENDANCE_PATH     = "Attendance"
STUDENT_DETAIL_PATH = "StudentDetails/studentdetails.csv"

TEXT        = "#1E293B"
ACCENT      = "#2563EB"
ACCENT2     = "#059669"
DANGER      = "#DC2626"
WARN        = "#D97706"
def get_all_dates():
    """Find all unique YYYY-MM-DD dates from all subjects in Attendance/."""
    all_dates = set()
    try:
        files = glob(os.path.join(ATTENDANCE_PATH, "*", "*.csv"))
        for f in files:
            name = os.path.basename(f)
            # Find YYYY-MM-DD pattern (10 chars after first underscore)
            # subject_YYYY-MM-DD_HH-MM-SS.csv
            parts = name.split("_")
            if len(parts) >= 2:
                all_dates.add(parts[-2] if "-" in parts[-2] else parts[1])
    except: pass
    valid_dates = set()
    for d in all_dates:
        # Check if it matches YYYY-MM-DD
        parts = d.split("-")
        if len(parts) == 3 and len(parts[0]) == 4 and len(parts[1]) == 2 and len(parts[2]) == 2:
            valid_dates.add(d)
    return sorted(list(valid_dates))


def get_sessions(subject):
    """List all session files for a subject. Returns list of strings for UI."""
    path_dir = os.path.join(ATTENDANCE_PATH, subject)
    if not os.path.exists(path_dir):
        return []
    pattern   = os.path.join(path_dir, f"{subject}_*.csv")
    filenames = glob(pattern)
    sessions = []
    for f in sorted(filenames, reverse=True): 
        base = os.path.basename(f).replace(f"{subject}_", "").replace(".csv", "")
        sessions.append(base)
    return sessions


def show_session_data(subject, session_label, tree_widget, notif_label):
    """Show raw attendance for ONE specific file."""
    path_dir = os.path.join(ATTENDANCE_PATH, subject)
    file_path = os.path.join(path_dir, f"{subject}_{session_label}.csv")
    if not os.path.exists(file_path): return
    try:
        df = pd.read_csv(file_path, dtype=str).fillna("")
        cols = list(df.columns)
        tree_widget["columns"] = cols
        for c in cols:
            tree_widget.heading(c, text=c)
            tree_widget.column(c, width=120, anchor="center")
        tree_widget.delete(*tree_widget.get_children())
        for _, r in df.iterrows():
            tree_widget.insert("", "end", values=list(r.values))
        _safe_lbl(notif_label, f"✅ Log: {session_label}", ACCENT2)
    except: pass


def generate_and_show(subject, start_date, end_date, tree_widget, notif_label, stream=None, semester=None):
    """
    Merge CSVs for subject WITHIN the date range [start_date, end_date].
    Dates format: YYYY-MM-DD
    """
    # --- CLEAR TABLE ON START ---
    tree_widget.delete(*tree_widget.get_children())

    # 1. Sabse pehle check karein ki students registered hain ya nahi
    try:
        master = pd.read_csv(STUDENT_DETAIL_PATH, dtype=str).fillna("")
        filtered = master.copy()
        if stream and stream.strip() != "":
            filtered = filtered[filtered["Stream"].str.strip() == stream.strip()]
        if semester and semester != "All":
            filtered = filtered[filtered["Semester"].str.strip() == str(semester).strip()]

        if filtered.empty:
            msg = f"❌ No registered students for '{stream}'"
            if semester != "All": msg += f" Sem {semester}"
            _safe_lbl(notif_label, msg + ". Register students first.", DANGER)
            return None
    except Exception:
        _safe_lbl(notif_label, "❌ Error reading student details.", DANGER)
        return None

    # 2. Agar students hain, tab subject folder dhoondhein
    path_dir  = os.path.join(ATTENDANCE_PATH, subject)
    if not os.path.exists(path_dir):
        _safe_lbl(notif_label, f"❌ No records found for '{subject}'.", WARN)
        return None

    pattern   = os.path.join(path_dir, f"{subject}_*.csv")
    all_files = glob(pattern)
    
    # Filter by date range
    valid_files = []
    for f in all_files:
        try:
            f_name = os.path.basename(f)
            parts = f_name.replace(f"{subject}_", "").split("_")
            file_date = parts[0]
            if start_date <= file_date <= end_date:
                valid_files.append(f)
        except: continue

    if not valid_files:
        _safe_lbl(notif_label, f"❌ No sessions found in this date range.", WARN)
        return None

    try:
        # Read session files and filter by stream/semester if data available
        session_frames = []
        for f in sorted(valid_files):
            d = pd.read_csv(f, dtype=str).fillna("")
            if "Stream" in d.columns and "Semester" in d.columns:
                if not d.empty:
                    f_stream = str(d["Stream"].iloc[0]).strip()
                    f_sem    = str(d["Semester"].iloc[0]).strip()
                    if f_stream == str(stream).strip():
                        if semester == "All" or f_sem == str(semester).strip():
                            session_frames.append(d)
            else:
                session_frames.append(d)

        total_sessions = len(session_frames)

        # Count attendance only for currently registered students
        all_enrollments = filtered["Enrollment"].str.strip().tolist()
        present_count = {e: 0 for e in all_enrollments}

        for d in session_frames:
            if "Enrollment" in d.columns:
                p_set = set(d["Enrollment"].str.strip().tolist())
                for e in all_enrollments:
                    if e in p_set:
                        present_count[e] += 1

        report_rows = []
        for _, row in filtered.iterrows():
            e = str(row["Enrollment"]).strip()
            cnt = present_count.get(e, 0)
            pct = round((cnt / total_sessions) * 100) if total_sessions > 0 else 0
            status = "✅ Good" if pct >= 75 else ("⚠ Low" if pct >= 50 else "❌ Poor")
            report_rows.append({
                "Enrollment": e,
                "Name": row.get("Name", ""),
                "Stream": row.get("Stream", ""),
                "Semester": row.get("Semester", ""),
                "Total Sessions": str(total_sessions),
                "Attended": str(cnt),
                "Percentage %": f"{pct}%",
                "Status": status
            })

        report_df = pd.DataFrame(report_rows)

        # UI Populate
        cols = list(report_df.columns)
        tree_widget["columns"] = cols
        for c in cols:
            tree_widget.heading(c, text=c)
            tree_widget.column(c, width=120, anchor="center")

        tree_widget.delete(*tree_widget.get_children())
        for _, r in report_df.iterrows():
            tag = "good" if "Good" in r["Status"] else ("warn" if "Low" in r["Status"] else "poor")
            tree_widget.insert("", "end", values=list(r.values), tags=(tag,))

        tree_widget.tag_configure("good", foreground="#059669")
        tree_widget.tag_configure("warn", foreground="#D97706")
        tree_widget.tag_configure("poor", foreground="#DC2626")

        _safe_lbl(notif_label, f"✅ Range Report: {total_sessions} sessions analyzed.", ACCENT2)

        # --- AUTO-SAVE FOR EXPORT (CSV & Excel) ---
        report_dir = os.path.join(ATTENDANCE_PATH, subject)
        os.makedirs(report_dir, exist_ok=True)
        report_df.to_csv(os.path.join(report_dir, "attendance_report.csv"), index=False)
        
        # Save Excel with auto-width
        try:
            excel_report = os.path.join(report_dir, "attendance_report.xlsx")
            with pd.ExcelWriter(excel_report, engine='openpyxl') as writer:
                report_df.to_excel(writer, index=False, sheet_name='Report')
                worksheet = writer.sheets['Report']
                for idx, col in enumerate(report_df.columns):
                    max_len = max(report_df[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.column_dimensions[chr(65+idx)].width = max_len
        except Exception as e:
            print(f"[ReportExcel] Error: {e}")

        # Return stats for dashboard
        if report_df.empty:
            return None
        
        avg_p = 0
        try:
            avg_p = round(report_df["Percentage %"].str.replace("%","").astype(float).mean())
        except: pass

        return {
            "total": len(report_df),
            "avg": f"{avg_p}%",
            "good": len(report_df[report_df["Status"].str.contains("Good")]),
            "poor": len(report_df[report_df["Status"].str.contains("Poor")])
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        _safe_lbl(notif_label, f"Error: {e}", DANGER)


def export_report(subject, start_date, end_date, stream=None, semester=None):
    """
    NEW: Export Date-wise raw logs (5 columns) for a specific range.
    This is now called by the 'Export Report' button.
    """
    path_dir = os.path.join(ATTENDANCE_PATH, subject)
    pattern = os.path.join(path_dir, f"{subject}_*.csv")
    all_files = glob(pattern)
    
    valid_files = []
    for f in all_files:
        try:
            f_name = os.path.basename(f)
            parts = f_name.replace(f"{subject}_", "").split("_")
            file_date = parts[0]
            if start_date <= file_date <= end_date:
                valid_files.append(f)
        except: continue

    if not valid_files:
        raise FileNotFoundError("No session records found in this date range.")

    all_frames = []
    for f in sorted(valid_files):
        try:
            d = pd.read_csv(f, dtype=str).fillna("")
            if stream and "Stream" in d.columns and not d.empty:
                if str(d["Stream"].iloc[0]).strip() != str(stream).strip(): continue
            if semester and semester != "All" and "Semester" in d.columns and not d.empty:
                if str(d["Semester"].iloc[0]).strip() != str(semester).strip(): continue
            all_frames.append(d)
        except: continue

    if not all_frames:
        raise FileNotFoundError("No matching records for selected filters.")

    combined = pd.concat(all_frames, ignore_index=True)
    cols = ["Enrollment", "Name", "Date", "Time", "Status"]
    existing = [c for c in cols if c in combined.columns]
    df_final = combined[existing]

    out_path = os.path.join(path_dir, "range_logs_export.xlsx")
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        df_final.to_excel(writer, index=False, sheet_name='Attendance Logs')
        ws = writer.sheets['Attendance Logs']
        for idx, col in enumerate(df_final.columns):
            max_len = max(df_final[col].astype(str).map(len).max(), len(col)) + 3
            ws.column_dimensions[chr(65+idx)].width = max_len
    
    return out_path


def export_full_history(subject):
    """
    NEW: Returns the Summary report (8 columns).
    This is now called by the 'Download Full History' button.
    """
    path_dir = os.path.join(ATTENDANCE_PATH, subject)
    report_xlsx = os.path.join(path_dir, "attendance_report.xlsx")
    if os.path.exists(report_xlsx):
        return report_xlsx
    
    report_csv = os.path.join(path_dir, "attendance_report.csv")
    if os.path.exists(report_csv):
        return report_csv
        
    raise FileNotFoundError("No summary report found. Please click 'Generate Range Report' first.")


# ── Legacy support (subjectchoose still available if called elsewhere) ────────
def subjectchoose(text_to_speech):
    pass   # handled in attendance.py now


def _safe_lbl(label, text, color):
    try:
        if label and label.winfo_exists():
            label.config(text=text, fg=color)
    except Exception:
        pass
