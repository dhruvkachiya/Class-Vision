import tkinter as tk
from tkinter import *
from tkinter import ttk, messagebox, filedialog
import os
import cv2
import numpy as np
from PIL import ImageTk, Image
import pandas as pd
import datetime
import time
import threading
import json

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False

# Project modules
import show_attendance
import takeImage
import trainImage
import automaticAttedance

# ─── PATHS ────────────────────────────────────────────────────────────────────
HAARCASCADE_PATH        = "haarcascade_frontalface_default.xml"
TRAIN_IMAGE_LABEL_PATH  = "./TrainingImageLabel/Trainner.yml"
TRAIN_IMAGE_PATH        = "TrainingImage"
STUDENT_DETAIL_PATH     = "./StudentDetails/studentdetails.csv"
ATTENDANCE_PATH         = "Attendance"
SEMESTER_SUBJECTS_PATH  = "./StudentDetails/semester_subjects.json"
CAMERA_CONFIG_PATH      = "./camera_config.json"

for d in [TRAIN_IMAGE_PATH, "StudentDetails", ATTENDANCE_PATH, "TrainingImageLabel"]:
    os.makedirs(d, exist_ok=True)

# Ensure CSV has correct columns
try:
    df_check = pd.read_csv(STUDENT_DETAIL_PATH)
    for col in ["Stream", "Semester"]:
        if col not in df_check.columns:
            df_check[col] = ""
    df_check.to_csv(STUDENT_DETAIL_PATH, index=False)
except Exception:
    pd.DataFrame(columns=["Enrollment","Name","Stream","Semester"]).to_csv(STUDENT_DETAIL_PATH, index=False)

# ─── SEMESTER SUBJECTS HELPERS (Stream → Semester → Subjects) ────────────────────
def load_sem_subjects():
    """Returns dict: {stream: {sem_str: [subject, ...]}, ...}"""
    try:
        with open(SEMESTER_SUBJECTS_PATH, "r") as f:
            data = json.load(f)
        return data
    except Exception:
        return {}

def save_sem_subjects(data):
    with open(SEMESTER_SUBJECTS_PATH, "w") as f:
        json.dump(data, f, indent=2)

def get_streams():
    return sorted(load_sem_subjects().keys())

def get_sems_for_stream(stream):
    data = load_sem_subjects()
    return sorted(data.get(stream, {}).keys(), key=lambda x: int(x))

def get_subjects_for_stream_sem(stream, sem):
    data = load_sem_subjects()
    return data.get(stream, {}).get(str(sem), [])

# ─── CAMERA CONFIG HELPERS ────────────────────────────────────────────────────
def load_camera_config():
    try:
        with open(CAMERA_CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception:
        # Default fallback
        return {"droidcam_ip": "192.168.1.7", "droidcam_port": "4747"}

def save_camera_config(ip, port):
    try:
        with open(CAMERA_CONFIG_PATH, "w") as f:
            json.dump({"droidcam_ip": ip, "droidcam_port": port}, f, indent=2)
        return True
    except Exception:
        return False

def text_to_speech(user_text):
    if not HAS_PYTTSX3:
        print(f"[TTS Mock] {user_text}")
        return
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.say(user_text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_speak, daemon=True).start()

# ─── LIGHT THEME DESIGN TOKENS ───────────────────────────────────────────────
BG          = "#F0F4F8"
WHITE       = "#FFFFFF"
CARD        = "#FFFFFF"
ACCENT      = "#2563EB"        # royal blue
ACCENT_HOV  = "#1D4ED8"
ACCENT2     = "#059669"        # green
DANGER      = "#DC2626"        # red
WARN        = "#D97706"        # amber
BORDER      = "#E2E8F0"
TEXT        = "#1E293B"
TEXT_MUTED  = "#64748B"
SIDEBAR_BG  = "#1E293B"        # dark sidebar
SIDEBAR_FG  = "#CBD5E1"
SIDEBAR_ACT = "#2563EB"
HEADER_BG   = "#1E293B"

FONT_H1    = ("Segoe UI", 20, "bold")
FONT_H2    = ("Segoe UI", 15, "bold")
FONT_H3    = ("Segoe UI", 12, "bold")
FONT_BODY  = ("Segoe UI", 11)
FONT_SMALL = ("Segoe UI", 10)
FONT_BTN   = ("Segoe UI", 11, "bold")
FONT_ENTRY = ("Segoe UI", 12)
FONT_LABEL = ("Segoe UI", 11, "bold")

# Global references to stream comboboxes for syncing
stream_combos = []

def sync_all_streams():
    """Refreshes all stream comboboxes across the application with data from JSON."""
    streams = get_streams()
    for cb in stream_combos:
        try:
            if cb.winfo_exists():
                cb["values"] = streams
        except Exception:
            pass
    # Trigger sub-refreshes for specific tabs
    try: _refresh_attendance_tab()
    except: pass
    try: _refresh_reports_tab()
    except: pass

# ─── UTILITIES ────────────────────────────────────────────────────────────────
def add_hover(btn, normal=ACCENT, hover=ACCENT_HOV):
    btn.bind("<Enter>", lambda e: btn.config(bg=hover))
    btn.bind("<Leave>", lambda e: btn.config(bg=normal))

def make_styled_btn(parent, text, command, bg=ACCENT, fg="white",
                    width=None, height=None, font=FONT_BTN, padx=16, pady=8):
    kw = dict(text=text, command=command, bg=bg, fg=fg, font=font,
              relief=FLAT, bd=0, cursor="hand2",
              activebackground=bg, activeforeground=fg, padx=padx, pady=pady)
    if width:  kw["width"] = width
    if height: kw["height"] = height
    btn = tk.Button(parent, **kw)
    return btn

def make_card(parent, **grid_kw):
    f = tk.Frame(parent, bg=CARD, relief=FLAT, bd=0,
                 highlightbackground=BORDER, highlightthickness=1)
    return f

def separator(parent, **pack_kw):
    tk.Frame(parent, bg=BORDER, height=1).pack(**pack_kw)

# ─── ERR POPUP ────────────────────────────────────────────────────────────────
def err_screen():
    popup = Toplevel()
    popup.title("Warning")
    popup.geometry("360x120")
    popup.configure(bg=WHITE)
    popup.resizable(False, False)
    popup.grab_set()
    tk.Label(popup, text="⚠  Enrollment & Name are required!",
             fg=DANGER, bg=WHITE, font=FONT_H3).pack(pady=22)
    make_styled_btn(popup, "OK", popup.destroy, width=10).pack()

# ─── VALIDATE ─────────────────────────────────────────────────────────────────
def testVal(inStr, acttyp):
    if acttyp == "1":
        if not inStr.isdigit():
            return False
    return True

# ─── RETRAIN HELPER ──────────────────────────────────────────────────────────
def retrain_model(notif_label=None):
    def _run():
        try:
            if notif_label:
                notif_label.config(text="⚙  Retraining model…", fg=WARN)
            trainImage.TrainImage(HAARCASCADE_PATH, TRAIN_IMAGE_PATH,
                                  TRAIN_IMAGE_LABEL_PATH, notif_label or tk.Label(), text_to_speech)
        except Exception as e:
            print(f"[Retrain] {e}")
    threading.Thread(target=_run, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
#  ADMIN TAB  ── Register + Edit Student + Quick Access
# ─────────────────────────────────────────────────────────────────────────────
def build_admin_tab(parent):
    # scrollable canvas
    canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
    sb = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=sb.set)
    inner = tk.Frame(canvas, bg=BG)
    canvas_win = canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.pack(side=LEFT, fill=BOTH, expand=True)
    sb.pack(side=RIGHT, fill=Y)

    # Make inner frame take Canvas width
    def configure_canvas(event):
        canvas.itemconfig(canvas_win, width=event.width)
    canvas.bind("<Configure>", configure_canvas)

    def configure_inner(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    inner.bind("<Configure>", configure_inner)

    # Mousewheel scrolling
    def _on_mousewheel(event):
        canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    def _bind_mousewheel(event):
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def _unbind_mousewheel(event):
        canvas.unbind_all("<MouseWheel>")

    inner.bind("<Enter>", _bind_mousewheel)
    inner.bind("<Leave>", _unbind_mousewheel)

    # ── Section: Camera Settings (FIX for IP BUG) ──────────────────────────
    sec_cam = make_card(inner)
    sec_cam.pack(fill=X, padx=24, pady=(20, 12))

    tk.Label(sec_cam, text="  📸  Camera Settings (DroidCam)", bg="#6366F1", fg="white",
             font=FONT_H2, anchor="w", pady=10).pack(fill=X)

    cam_form = tk.Frame(sec_cam, bg=CARD)
    cam_form.pack(fill=X, padx=20, pady=12)

    def lbl(parent, text, row, col=0):
        tk.Label(parent, text=text, bg=CARD, fg=TEXT, font=FONT_LABEL,
                 anchor="w").grid(row=row, column=col, sticky="w",
                                  padx=(0, 10), pady=8)

    def entry_simple(parent, row, col=1, width=20):
        e = tk.Entry(parent, width=width, bg=WHITE, fg=TEXT, font=FONT_ENTRY,
                     relief=SOLID, bd=1, insertbackground=ACCENT)
        e.grid(row=row, column=col, sticky="w", pady=8)
        return e

    lbl(cam_form, "DroidCam IP", 0);  txt_ip   = entry_simple(cam_form, 0)
    lbl(cam_form, "DroidCam Port", 0, 2); txt_port = entry_simple(cam_form, 0, 3, width=8)

    # Load current
    cfg = load_camera_config()
    txt_ip.insert(0, cfg.get("droidcam_ip", "192.168.1.7"))
    txt_port.insert(0, cfg.get("droidcam_port", "4747"))

    notif_cam = tk.Label(sec_cam, text="", bg=CARD, fg=ACCENT2, font=FONT_SMALL)
    notif_cam.pack(fill=X, padx=20, pady=(0, 4))

    def update_cam():
        ip = txt_ip.get().strip()
        port = txt_port.get().strip()
        if not ip or not port:
            notif_cam.config(text="⚠  IP and Port are required.", fg=DANGER); return
        if save_camera_config(ip, port):
            notif_cam.config(text=f"✅  Camera updated to {ip}:{port} successfully.", fg=ACCENT2)
            try:
                status_cam_lbl.config(text=f"  🟢  System Ready  |  DroidCam: http://{ip}:{port}")
            except:
                pass
        else:
            notif_cam.config(text="❌  Failed to save settings.", fg=DANGER)

    btn_cam = make_styled_btn(sec_cam, "💾  Save Camera Settings", update_cam, bg="#6366F1")
    btn_cam.pack(padx=20, pady=(0, 16), anchor="w")
    add_hover(btn_cam, "#6366F1", "#4F46E5")

    # ── Section: Register Student ──────────────────────────────────────────
    sec1 = make_card(inner)
    sec1.pack(fill=X, padx=24, pady=(20, 12))

    tk.Label(sec1, text="  📝  Register New Student", bg=ACCENT, fg="white",
             font=FONT_H2, anchor="w", pady=10).pack(fill=X)

    form1 = tk.Frame(sec1, bg=CARD)
    form1.pack(fill=X, padx=20, pady=12)

    def lbl(parent, text, row, col=0):
        tk.Label(parent, text=text, bg=CARD, fg=TEXT, font=FONT_LABEL,
                 anchor="w").grid(row=row, column=col, sticky="w",
                                  padx=(0, 10), pady=8)

    def entry(parent, row, col=1, width=28, validate=None):
        e = tk.Entry(parent, width=width, bg=WHITE, fg=TEXT, font=FONT_ENTRY,
                     relief=SOLID, bd=1, insertbackground=ACCENT)
        e.grid(row=row, column=col, sticky="w", pady=8)
        if validate:
            e["validate"] = "key"
            e["validatecommand"] = (e.register(testVal), "%P", "%d")
        return e

    lbl(form1, "Enrollment No", 0); txt_enroll = entry(form1, 0, validate=True)
    lbl(form1, "Full Name", 1);     txt_name   = entry(form1, 1)
    lbl(form1, "Stream", 2)
    var_stream = StringVar()
    curr_streams = get_streams()
    if curr_streams: var_stream.set(curr_streams[0])
    
    cb_reg_stream = ttk.Combobox(form1, textvariable=var_stream, values=curr_streams,
                                 font=FONT_ENTRY, width=26, state="readonly")
    cb_reg_stream.grid(row=2, column=1, sticky="w", pady=8)
    stream_combos.append(cb_reg_stream)
    lbl(form1, "Semester", 3)
    var_sem = StringVar(value="1")
    sem_frame = tk.Frame(form1, bg=CARD)
    sem_frame.grid(row=3, column=1, sticky="w", pady=8)
    for s in range(1, 9):
        tk.Radiobutton(sem_frame, text=str(s), variable=var_sem, value=str(s),
                       bg=CARD, fg=TEXT, font=FONT_BODY,
                       selectcolor=WHITE, activebackground=CARD).pack(side=LEFT)

    notif1 = tk.Label(sec1, text="", bg=CARD, fg=ACCENT2, font=FONT_SMALL,
                      wraplength=700, anchor="w", justify="left")
    notif1.pack(fill=X, padx=20, pady=(0, 4))

    btn_row1 = tk.Frame(sec1, bg=CARD)
    btn_row1.pack(padx=20, pady=(4, 16), anchor="w")

    def take_image():
        l1 = txt_enroll.get().strip()
        l2 = txt_name.get().strip()
        stream = var_stream.get()
        sem    = var_sem.get()
        if not l1 or not l2:
            err_screen(); return
        notif1.config(text="📸  Opening camera… make sure DroidCam is running.", fg=ACCENT)
        def _run():
            takeImage.TakeImage(l1, l2, stream, sem,
                                HAARCASCADE_PATH, TRAIN_IMAGE_PATH,
                                notif1, err_screen, text_to_speech)
        threading.Thread(target=_run, daemon=True).start()
        txt_enroll.delete(0, "end")
        txt_name.delete(0, "end")

    def train_model():
        notif1.config(text="⚙  Training model… this may take a moment.", fg=WARN)
        def _run():
            trainImage.TrainImage(HAARCASCADE_PATH, TRAIN_IMAGE_PATH,
                                  TRAIN_IMAGE_LABEL_PATH, notif1, text_to_speech)
        threading.Thread(target=_run, daemon=True).start()

    b1 = make_styled_btn(btn_row1, "📸  Capture Faces", take_image)
    b1.pack(side=LEFT, padx=(0, 12))
    add_hover(b1)
    b2 = make_styled_btn(btn_row1, "🧠  Train Model", train_model,
                         bg=ACCENT2)
    b2.pack(side=LEFT)
    add_hover(b2, ACCENT2, "#047857")

    separator(inner, fill=X, padx=24, pady=4)

    # ── Section: Edit / View Students ─────────────────────────────────────
    sec2 = make_card(inner)
    sec2.pack(fill=X, padx=24, pady=(0, 12))

    tk.Label(sec2, text="  ✏️  Edit Student Details", bg=WARN, fg="white",
             font=FONT_H2, anchor="w", pady=10).pack(fill=X)

    tbl_frame = tk.Frame(sec2, bg=CARD)
    tbl_frame.pack(fill=BOTH, padx=20, pady=12)

    cols = ("Enrollment", "Name", "Stream", "Semester")
    tree = ttk.Treeview(tbl_frame, columns=cols, show="headings", height=6,
                        selectmode="browse")
    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=160, anchor="center", stretch=True)

    style = ttk.Style()
    style.configure("Treeview", font=FONT_BODY, rowheight=28,
                    background=WHITE, fieldbackground=WHITE, foreground=TEXT)
    style.configure("Treeview.Heading", font=FONT_LABEL, background=BG)
    style.map("Treeview", background=[("selected", ACCENT)])

    tsb = tk.Scrollbar(tbl_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=tsb.set)
    tree.pack(side=LEFT, fill=BOTH, expand=True)
    tsb.pack(side=RIGHT, fill=Y)

    def refresh_table():
        tree.delete(*tree.get_children())
        try:
            df = pd.read_csv(STUDENT_DETAIL_PATH, dtype=str).fillna("")
            for _, r in df.iterrows():
                tree.insert("", "end", values=(
                    r.get("Enrollment",""), r.get("Name",""),
                    r.get("Stream",""), r.get("Semester","")))
        except Exception:
            pass

    refresh_table()

    edit_form = tk.Frame(sec2, bg=CARD)
    edit_form.pack(fill=X, padx=20, pady=(0, 4))

    ef_labels = ["Enrollment", "Name", "Stream", "Semester"]
    ef_vars   = [StringVar() for _ in ef_labels]
    selected_student_id = [None] # Track original ID for renaming/editing

    for i, (lbl_txt, var) in enumerate(zip(ef_labels, ef_vars)):
        tk.Label(edit_form, text=lbl_txt, bg=CARD, fg=TEXT, font=FONT_LABEL,
                 anchor="w").grid(row=0, column=i*2, padx=(0,4), pady=4)
        if lbl_txt == "Stream":
            w = ttk.Combobox(edit_form, textvariable=var, values=get_streams(),
                             font=FONT_ENTRY, width=18, state="readonly")
            stream_combos.append(w)
        elif lbl_txt == "Semester":
            w = ttk.Combobox(edit_form, textvariable=var,
                             values=[str(x) for x in range(1,9)],
                             font=FONT_ENTRY, width=8, state="readonly")
        else:
            w = tk.Entry(edit_form, textvariable=var, bg=WHITE, fg=TEXT,
                         font=FONT_ENTRY, relief=SOLID, bd=1, width=18)
        w.grid(row=0, column=i*2+1, padx=(0,12), pady=4)

    def on_select(event):
        sel = tree.selection()
        if not sel: return
        vals = tree.item(sel[0], "values")
        selected_student_id[0] = vals[0] # Remember original enrollment
        for var, val in zip(ef_vars, vals):
            var.set(val)

    tree.bind("<<TreeviewSelect>>", on_select)

    notif2 = tk.Label(sec2, text="", bg=CARD, fg=ACCENT2, font=FONT_SMALL)
    notif2.pack(fill=X, padx=20, pady=(0, 4))

    edit_btn_row = tk.Frame(sec2, bg=CARD)
    edit_btn_row.pack(padx=20, pady=(0, 16), anchor="w")

    def save_edit():
        old_enroll = selected_student_id[0]
        if not old_enroll:
            notif2.config(text="⚠  Select a student from the table first.", fg=DANGER); return
            
        new_enroll = ef_vars[0].get().strip()
        new_name   = ef_vars[1].get().strip()
        
        if not new_enroll or not new_name:
            notif2.config(text="⚠  Enrollment & Name cannot be empty.", fg=DANGER); return
            
        try:
            df = pd.read_csv(STUDENT_DETAIL_PATH, dtype=str).fillna("")
            idx = df.index[df["Enrollment"] == old_enroll].tolist()
            
            # Fallback if selected student ID was manually deleted from CSV
            if not idx:
                idx = df.index[df["Enrollment"] == new_enroll].tolist()

            if idx:
                # Get old name for folder renaming
                orig_name = df.at[idx[0], "Name"]
                
                # Update CSV data
                df.at[idx[0], "Enrollment"] = new_enroll
                df.at[idx[0], "Name"]       = new_name
                df.at[idx[0], "Stream"]     = ef_vars[2].get().strip()
                df.at[idx[0], "Semester"]   = ef_vars[3].get().strip()
                
                # Rename TrainingImage folder if ID or Name changed
                if old_enroll != new_enroll or orig_name != new_name:
                    old_path = os.path.join(TRAIN_IMAGE_PATH, f"{old_enroll}_{orig_name}")
                    new_path = os.path.join(TRAIN_IMAGE_PATH, f"{new_enroll}_{new_name}")
                    if os.path.exists(old_path):
                        try:
                            os.rename(old_path, new_path)
                        except Exception as e:
                            print(f"[Rename Error] {e}")

                df.to_csv(STUDENT_DETAIL_PATH, index=False)
                notif2.config(text=f"✅  Updated {new_enroll} successfully.", fg=ACCENT2)
                selected_student_id[0] = new_enroll # Update selection tracker
                refresh_table()
            else:
                notif2.config(text="⚠  Record not found in database.", fg=DANGER)
        except Exception as e:
            notif2.config(text=f"Error: {e}", fg=DANGER)

    def delete_student():
        enroll = ef_vars[0].get().strip()
        if not enroll:
            notif2.config(text="⚠  Select a student first.", fg=DANGER); return
        if not messagebox.askyesno("Confirm Delete", f"Delete student {enroll}?\nThis cannot be undone."): return
        try:
            df = pd.read_csv(STUDENT_DETAIL_PATH, dtype=str).fillna("")
            df = df[df["Enrollment"] != enroll]
            df.to_csv(STUDENT_DETAIL_PATH, index=False)
            notif2.config(text=f"🗑  Deleted {enroll}.", fg=WARN)
            refresh_table()
            for var in ef_vars: var.set("")
        except Exception as e:
            notif2.config(text=f"Error: {e}", fg=DANGER)

    b3 = make_styled_btn(edit_btn_row, "💾  Save Changes", save_edit)
    b3.pack(side=LEFT, padx=(0, 12))
    add_hover(b3)
    b4 = make_styled_btn(edit_btn_row, "🔄  Refresh", refresh_table, bg="#64748B")
    b4.pack(side=LEFT, padx=(0, 12))
    add_hover(b4, "#64748B", "#475569")
    b5 = make_styled_btn(edit_btn_row, "🗑  Delete Student", delete_student, bg=DANGER)
    b5.pack(side=LEFT)
    add_hover(b5, DANGER, "#B91C1C")

    separator(inner, fill=X, padx=24, pady=4)

    # --- Management Functions ---
    def refresh_stream_combo():
        streams = get_streams()
        sync_all_streams()
        if streams and not adm_stream_var.get(): adm_stream_var.set(streams[0])
        refresh_subject_list()

    def refresh_subject_list(*args):
        try:
            stream, sem = adm_stream_var.get(), adm_sem_var.get()
            subs = get_subjects_for_stream_sem(stream, sem)
            subj_listbox.delete(0, END)
            for s in subs: subj_listbox.insert(END, s)
        except: pass

    def add_stream():
        name = new_stream_entry.get().strip()
        if not name or name == "New Stream Name…": return
        data = load_sem_subjects()
        if name in data: return
        data[name] = {str(i): [] for i in range(1, 9)}; save_sem_subjects(data)
        new_stream_entry.delete(0, END); refresh_stream_combo()
        notif_sem.config(text=f"✅ Stream '{name}' added.", fg=ACCENT2)

    def add_subject():
        st, sm, sb = adm_stream_var.get(), adm_sem_var.get(), new_subj_entry.get().strip()
        if not st or not sb or sb == "New Subject Name…": return
        data = load_sem_subjects(); data[st][sm].append(sb); save_sem_subjects(data); refresh_subject_list()
        new_subj_entry.delete(0, END); new_subj_entry.insert(0, "")
        notif_sem.config(text=f"✅ Subject '{sb}' added.", fg=ACCENT2)

    def remove_selected_subject():
        st, sm, sel = adm_stream_var.get(), adm_sem_var.get(), subj_listbox.curselection()
        if sel:
            sb = subj_listbox.get(sel[0])
            if messagebox.askyesno("Confirm Delete", f"Are you sure you want to remove the subject '{sb}'?"):
                data = load_sem_subjects()
                data[st][sm].remove(sb)
                save_sem_subjects(data)
                refresh_subject_list()
                notif_sem.config(text=f"🗑 Removed subject '{sb}'.", fg=WARN)

    def delete_stream():
        st = adm_stream_var.get()
        if st and messagebox.askyesno("Confirm", f"Delete stream '{st}' and all subjects?"):
            data = load_sem_subjects(); data.pop(st, None); save_sem_subjects(data); refresh_stream_combo()
            notif_sem.config(text=f"🗑 Deleted stream '{st}'.", fg=WARN)

    def open_update_stream():
        sw = Toplevel(window); sw.title("Update Stream"); sw.configure(bg=WHITE); sw.grab_set()
        w, h = 380, 300
        x = window.winfo_x() + (window.winfo_width() // 2) - (w // 2)
        y = window.winfo_y() + (window.winfo_height() // 2) - (h // 2)
        sw.geometry(f"{w}x{h}+{x}+{y}")
        tk.Label(sw, text="✏️ Update Stream", bg=WHITE, font=FONT_H2).pack(pady=15)
        sel_v = StringVar(); cb = ttk.Combobox(sw, textvariable=sel_v, values=get_streams(), state="readonly", width=30)
        cb.set("Select Stream..."); cb.pack(pady=5)
        ent = tk.Entry(sw, font=FONT_ENTRY, width=32, relief=SOLID, bd=1, fg=TEXT_MUTED)
        ent.insert(0, "Enter New Name..."); ent.pack(pady=10)
        ent.bind("<FocusIn>", lambda e: (ent.delete(0, END), ent.config(fg=TEXT)) if ent.get() == "Enter New Name..." else None)
        def save():
            o, n = sel_v.get(), ent.get().strip()
            if o and n and n != "Enter New Name...":
                data = load_sem_subjects(); data[n] = data.pop(o); save_sem_subjects(data)
                try:
                    df = pd.read_csv(STUDENT_DETAIL_PATH, dtype=str).fillna("")
                    df.loc[df["Stream"] == o, "Stream"] = n
                    df.to_csv(STUDENT_DETAIL_PATH, index=False)
                except: pass
                refresh_stream_combo(); sw.destroy()
                notif_sem.config(text=f"✅ Stream Updated: {o} ➔ {n}", fg=ACCENT2)
        make_styled_btn(sw, "💾 Save Changes", save, bg=ACCENT2).pack(pady=10)

    def open_update_subject():
        sw = Toplevel(window); sw.title("Update Subject"); sw.configure(bg=WHITE); sw.grab_set()
        w, h = 400, 350
        x = window.winfo_x() + (window.winfo_width() // 2) - (w // 2)
        y = window.winfo_y() + (window.winfo_height() // 2) - (h // 2)
        sw.geometry(f"{w}x{h}+{x}+{y}")
        tk.Label(sw, text="📚 Update Subject", bg=WHITE, font=FONT_H2).pack(pady=10)
        st_v = StringVar(); st_cb = ttk.Combobox(sw, textvariable=st_v, values=get_streams(), state="readonly", width=35)
        st_cb.set("Select Stream..."); st_cb.pack(pady=5)
        sb_v = StringVar(); sb_cb = ttk.Combobox(sw, textvariable=sb_v, state="readonly", width=35)
        sb_cb.set("Select Subject..."); sb_cb.pack(pady=5)
        st_cb.bind("<<ComboboxSelected>>", lambda e: sb_cb.config(values=[s for sm in range(1,9) for s in get_subjects_for_stream_sem(st_v.get(), str(sm))]))
        ent = tk.Entry(sw, font=FONT_ENTRY, width=35, relief=SOLID, bd=1, fg=TEXT_MUTED)
        ent.insert(0, "Enter New Name..."); ent.pack(pady=10)
        ent.bind("<FocusIn>", lambda e: (ent.delete(0, END), ent.config(fg=TEXT)) if ent.get() == "Enter New Name..." else None)
        def save():
            st, o, n = st_v.get(), sb_v.get(), ent.get().strip()
            if st and o and n and n != "Enter New Name...":
                data = load_sem_subjects()
                for sm in range(1,9):
                    if o in data[st][str(sm)]: data[st][str(sm)][data[st][str(sm)].index(o)] = n
                save_sem_subjects(data)
                old_p, new_p = os.path.join(ATTENDANCE_PATH, o), os.path.join(ATTENDANCE_PATH, n)
                if os.path.exists(old_p):
                    try:
                        for f in os.listdir(old_p):
                            if f.startswith(o): os.rename(os.path.join(old_p,f), os.path.join(old_p,f.replace(o,n,1)))
                        os.rename(old_p, new_p)
                    except: pass
                refresh_subject_list(); sw.destroy()
                notif_sem.config(text=f"✅ Subject Updated: {o} ➔ {n}", fg=ACCENT2)
        make_styled_btn(sw, "💾 Save Changes", save, bg=ACCENT2).pack(pady=10)

    # ── Section: Manage Structure ──────────────────────────────────────────
    sec_sem = make_card(inner)
    sec_sem.pack(fill=X, padx=24, pady=(0, 12))
    tk.Label(sec_sem, text="  📚  Manage Academic Structure",
             bg="#0EA5E9", fg="white", font=FONT_H2, anchor="w", pady=10).pack(fill=X)

    mng_main = tk.Frame(sec_sem, bg=CARD)
    mng_main.pack(fill=X, padx=20, pady=15)

    left_col = tk.Frame(mng_main, bg=CARD)
    left_col.pack(side=LEFT, fill=BOTH, expand=True)

    grid_f = tk.Frame(left_col, bg=CARD)
    grid_f.pack(fill=X, pady=(0, 10))

    tk.Label(grid_f, text="Add Stream:", bg=CARD, fg=TEXT, font=FONT_LABEL).grid(row=0, column=2, sticky="w", padx=(12, 4))
    tk.Label(grid_f, text="Add Subject:", bg=CARD, fg=TEXT, font=FONT_LABEL).grid(row=0, column=3, sticky="w", padx=(12, 4))
    tk.Label(grid_f, text="Stream:", bg=CARD, fg=TEXT, font=FONT_LABEL).grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
    adm_stream_var = StringVar()
    adm_stream_combo = ttk.Combobox(grid_f, textvariable=adm_stream_var, font=FONT_ENTRY, width=18, state="readonly")
    adm_stream_combo.grid(row=1, column=1, sticky="w", pady=4); stream_combos.append(adm_stream_combo)
    new_stream_entry = tk.Entry(grid_f, bg=WHITE, fg=TEXT, font=FONT_ENTRY, relief=SOLID, bd=1, width=20)
    new_stream_entry.grid(row=1, column=2, sticky="w", padx=(12, 4), pady=4)
    new_stream_entry.insert(0, "New Stream Name…"); new_stream_entry.bind("<FocusIn>", lambda e: new_stream_entry.delete(0, END) if new_stream_entry.get() == "New Stream Name…" else None)
    new_subj_entry = tk.Entry(grid_f, bg=WHITE, fg=TEXT, font=FONT_ENTRY, relief=SOLID, bd=1, width=20)
    new_subj_entry.grid(row=1, column=3, sticky="w", padx=(12, 4), pady=4)
    new_subj_entry.insert(0, "New Subject Name…"); new_subj_entry.bind("<FocusIn>", lambda e: new_subj_entry.delete(0, END) if new_subj_entry.get() == "New Subject Name…" else None)
    tk.Label(grid_f, text="Semester:", bg=CARD, fg=TEXT, font=FONT_LABEL).grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
    adm_sem_var = StringVar(value="1")
    adm_sem_combo = ttk.Combobox(grid_f, textvariable=adm_sem_var, values=[str(i) for i in range(1, 9)], font=FONT_ENTRY, width=5, state="readonly")
    adm_sem_combo.grid(row=2, column=1, sticky="w", pady=4)

    list_f = tk.Frame(left_col, bg=CARD)
    list_f.pack(fill=X, pady=(10, 0))
    tk.Label(list_f, text="Subjects List:", bg=CARD, fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w")
    subj_listbox = Listbox(list_f, font=FONT_BODY, bg=WHITE, fg=TEXT, relief=SOLID, bd=1, height=8)
    subj_listbox.pack(fill=X, pady=5)

    act_p = tk.Frame(mng_main, bg=CARD)
    act_p.pack(side=LEFT, fill=Y, padx=(25, 0))
    tk.Label(act_p, text="Actions:", bg=CARD, fg=TEXT, font=FONT_LABEL).pack(anchor="w", pady=(0, 5))
    make_styled_btn(act_p, "➕  Add Stream", add_stream, bg="#0F766E", width=18).pack(pady=1)
    make_styled_btn(act_p, "🗑  Delete Stream", delete_stream, bg="#B91C1C", width=18).pack(pady=1)
    separator(act_p, fill=X, pady=8)
    make_styled_btn(act_p, "➕  Add Subject", add_subject, bg="#0EA5E9", width=18).pack(pady=1)
    make_styled_btn(act_p, "❌  Remove Subject", remove_selected_subject, bg=DANGER, width=18).pack(pady=1)
    separator(act_p, fill=X, pady=8)
    make_styled_btn(act_p, "✏️  Update Stream", open_update_stream, bg="#6366F1", width=18).pack(pady=1)
    make_styled_btn(act_p, "✏️  Update Subject", open_update_subject, bg="#8B5CF6", width=18).pack(pady=1)

    notif_sem = tk.Label(sec_sem, text="", bg=CARD, font=FONT_SMALL); notif_sem.pack(pady=(0, 10))
    adm_stream_combo.bind("<<ComboboxSelected>>", refresh_subject_list)
    adm_sem_combo.bind("<<ComboboxSelected>>", refresh_subject_list)
    refresh_stream_combo()

    separator(inner, fill=X, padx=24, pady=4)

    # ── Quick subject Access ───────────────────────────────────────────────
    sec3 = make_card(inner)
    sec3.pack(fill=X, padx=24, pady=(0, 20))
    tk.Label(sec3, text="  📂  Quick Subject Folder Access", bg="#7C3AED", fg="white",
             font=FONT_H2, anchor="w", pady=10).pack(fill=X)

    qa_inner = tk.Frame(sec3, bg=CARD)
    qa_inner.pack(fill=X, padx=20, pady=14)

    tk.Label(qa_inner, text="Subject Name:", bg=CARD, fg=TEXT, font=FONT_LABEL).pack(side=LEFT)
    qa_entry = tk.Entry(qa_inner, bg=WHITE, fg=TEXT, font=FONT_ENTRY,
                        relief=SOLID, bd=1, width=22)
    qa_entry.pack(side=LEFT, padx=10)

    def open_folder():
        sub = qa_entry.get().strip()
        path = os.path.join(ATTENDANCE_PATH, sub)
        if sub and os.path.exists(path):
            os.startfile(path)
        else:
            messagebox.showwarning("Not Found",
                f"No attendance folder found for subject: '{sub}'")

    b6 = make_styled_btn(qa_inner, "📂  Open Folder", open_folder, bg="#7C3AED")
    b6.pack(side=LEFT, padx=(0, 12))
    add_hover(b6, "#7C3AED", "#6D28D9")

    def list_subjects():
        try:
            data = load_sem_subjects()
            if not data:
                messagebox.showinfo("Subjects", "No subjects defined in 'Manage Streams' section yet.")
                return
                
            sw = Toplevel()
            sw.title("Subject Explorer")
            sw.configure(bg=WHITE)
            sw.geometry("500x550")
            sw.grab_set()
            
            tk.Label(sw, text="📂 Attendance Folder Explorer", bg=WHITE, fg=TEXT,
                     font=("Segoe UI", 16, "bold"), pady=15).pack()
            
            instr = tk.Label(sw, text="Double-click a subject to open its history folder", 
                             bg=WHITE, fg=TEXT_MUTED, font=FONT_SMALL)
            instr.pack(pady=(0, 10))
            
            # --- TREEVIEW FOR CATEGORIZED VIEW ---
            tree_frame = tk.Frame(sw, bg=WHITE, bd=1, relief=SOLID)
            tree_frame.pack(fill=BOTH, expand=True, padx=20, pady=(0, 20))
            
            columns = ("Name", "Status")
            tree = ttk.Treeview(tree_frame, columns=columns, show="tree headings", selectmode="browse")
            
            # Column headers
            tree.heading("#0", text="Streams / Subjects")
            tree.heading("Name", text="") 
            tree.heading("Status", text="Data Status")
            
            tree.column("#0", width=300, anchor="w")
            tree.column("Name", width=0, stretch=False) 
            tree.column("Status", width=120, anchor="center")
            
            # Styling for Treeview
            style = ttk.Style()
            # style.theme_use("clam")
            style.configure("Treeview", font=FONT_BODY, rowheight=32)
            style.configure("Treeview.Heading", font=FONT_LABEL)
            
            sb = tk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=sb.set)
            
            tree.pack(side=LEFT, fill=BOTH, expand=True)
            sb.pack(side=RIGHT, fill=Y)

            # Insert Data
            for stream in sorted(data.keys()):
                # Stream Node (Closed by default)
                stream_node = tree.insert("", "end", text=f" 📁  {stream}", open=False, values=("", "---"))
                
                # Collect all subjects for this stream
                stream_subs = set()
                for sem, sub_list in data[stream].items():
                    for s in sub_list:
                        if s: stream_subs.add(s)
                
                for s in sorted(list(stream_subs)):
                    path = os.path.join(ATTENDANCE_PATH, s)
                    exists = os.path.exists(path)
                    status = "✅ Records Exist" if exists else "⚪ No Data"
                    icon = "📄"
                    tree.insert(stream_node, "end", text=f"      {icon}  {s}", values=(s, status))

            def on_action(event=None):
                sel = tree.selection()
                if not sel: return
                item = tree.item(sel[0])
                sub_name = item['values'][0]
                if sub_name:
                    path = os.path.join(ATTENDANCE_PATH, sub_name)
                    if os.path.exists(path):
                        os.startfile(path)
                    else:
                        messagebox.showinfo("No Folder", 
                            f"Attendance record for '{sub_name}' hasn't been created yet.")
                else:
                    # Toggle expand/collapse if clicking a stream
                    tree.item(sel[0], open=not tree.item(sel[0], 'open'))

            tree.bind("<Double-1>", on_action)
            
            btn_row = tk.Frame(sw, bg=WHITE)
            btn_row.pack(fill=X, padx=20, pady=(0, 20))
            
            b_open = make_styled_btn(btn_row, "📂  Open Selected Folder", on_action, bg=ACCENT)
            b_open.pack(side=RIGHT)
            add_hover(b_open)
                                     
        except Exception as e:
            messagebox.showerror("Error", f"Could not explore subjects: {e}")

    b7 = make_styled_btn(qa_inner, "📋  List All Subjects", list_subjects, bg="#64748B")
    b7.pack(side=LEFT)
    add_hover(b7, "#64748B", "#475569")

# Global refresh hooks for tabs
_refresh_attendance_tab = lambda: None
_refresh_reports_tab    = lambda: None

# ─────────────────────────────────────────────────────────────────────────────
#  TAKE ATTENDANCE TAB
# ─────────────────────────────────────────────────────────────────────────────
def build_attendance_tab(parent):
    global _refresh_attendance_tab_sems

    inner = tk.Frame(parent, bg=BG)
    inner.pack(fill=BOTH, expand=True)

    card = make_card(inner)
    card.pack(fill=BOTH, padx=24, pady=24, expand=True)

    tk.Label(card, text="  📡  Take Attendance", bg=ACCENT, fg="white",
             font=FONT_H2, anchor="w", pady=10).pack(fill=X)

    # ── Master Container for Form & List ─────────────────────────────────────
    main_container = tk.Frame(card, bg=CARD)
    main_container.pack(fill=BOTH, expand=True, padx=20, pady=(0, 16))

    # LEFT: Session Settings
    left_col = tk.Frame(main_container, bg=CARD)
    left_col.pack(side=LEFT, fill=BOTH, expand=True)

    form = tk.Frame(left_col, bg=CARD)
    form.pack(fill=X, pady=(16, 0))

    # ── Row 0: Stream ──────────────────────────────────────────────────────
    tk.Label(form, text="Stream:", bg=CARD, fg=TEXT, font=FONT_LABEL).grid(
        row=0, column=0, sticky="w", pady=8, padx=(0, 12))
    att_stream_var = StringVar()
    att_stream_combo = ttk.Combobox(form, textvariable=att_stream_var,
                                    font=FONT_ENTRY, width=24, state="readonly")
    att_stream_combo.grid(row=0, column=1, sticky="w", pady=8)
    stream_combos.append(att_stream_combo)

    # ── Row 1: Semester ────────────────────────────────────────────────────
    tk.Label(form, text="Semester:", bg=CARD, fg=TEXT, font=FONT_LABEL).grid(
        row=1, column=0, sticky="w", pady=8, padx=(0, 12))
    sem_att_var = StringVar(value="1")
    sem_att_combo = ttk.Combobox(form, textvariable=sem_att_var,
                                 values=[str(i) for i in range(1, 9)],
                                 font=FONT_ENTRY, width=8, state="readonly")
    sem_att_combo.grid(row=1, column=1, sticky="w", pady=8)

    # ── Row 2: Subject (filtered by stream + semester) ─────────────────────
    tk.Label(form, text="Subject:", bg=CARD, fg=TEXT, font=FONT_LABEL).grid(
        row=2, column=0, sticky="w", pady=8, padx=(0, 12))
    subj_att_var = StringVar()
    subj_att_combo = ttk.Combobox(form, textvariable=subj_att_var,
                                  font=FONT_ENTRY, width=24, state="readonly")
    subj_att_combo.grid(row=2, column=1, sticky="w", pady=8)

    def refresh_att_subjects(*args):
        stream = att_stream_var.get()
        sem    = sem_att_var.get()
        subs   = get_subjects_for_stream_sem(stream, sem)
        subj_att_combo["values"] = subs
        # Preserve selection if valid
        curr = subj_att_var.get()
        if subs:
            if curr not in subs:
                subj_att_var.set(subs[0])
        else:
            subj_att_var.set("")

    def refresh_att_streams(*args):
        streams = get_streams()
        att_stream_combo["values"] = streams
        curr = att_stream_var.get()
        if streams:
            if curr not in streams:
                att_stream_var.set(streams[0])
        refresh_att_subjects()

    att_stream_combo.bind("<<ComboboxSelected>>", refresh_att_subjects)
    sem_att_combo.bind("<<ComboboxSelected>>",    refresh_att_subjects)
    refresh_att_streams()

    # Expose hook so admin panel triggers cascade refresh
    _refresh_attendance_tab = refresh_att_streams


    # ── Row 3: Duration ──────────────────────────────────────────────────────
    tk.Label(form, text="Session Duration:", bg=CARD, fg=TEXT, font=FONT_LABEL).grid(
        row=3, column=0, sticky="w", pady=8)
    timer_var = IntVar(value=5)
    timer_frame = tk.Frame(form, bg=CARD)
    timer_frame.grid(row=3, column=1, sticky="w")
    for mins, txt in [(5, "5 min"), (10, "10 min"), (15, "15 min"), (20, "20 min")]:
        tk.Radiobutton(timer_frame, text=txt, variable=timer_var, value=mins,
                       bg=CARD, fg=TEXT, font=FONT_BODY, selectcolor=WHITE,
                       activebackground=CARD).pack(side=LEFT, padx=(0, 16))

    # RIGHT: Confirmation List
    right_col = tk.Frame(main_container, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
    right_col.pack(side=RIGHT, fill=BOTH, expand=True, padx=(24, 0))

    tk.Label(right_col, text="  ✅  Confirmed Attendance", bg="#F8FAFC", fg=TEXT,
             font=FONT_LABEL, anchor="w", pady=8).pack(fill=X)
    separator(right_col, fill=X)

    names_frame = tk.Frame(right_col, bg=WHITE)
    names_frame.pack(fill=BOTH, expand=True, padx=4, pady=4)

    names_list = tk.Listbox(names_frame, font=FONT_H3, fg=ACCENT2, bg=WHITE,
                            relief=FLAT, bd=0, highlightthickness=0,
                            selectbackground=BG, selectforeground=TEXT)
    names_list.pack(side=LEFT, fill=BOTH, expand=True)

    n_sb = tk.Scrollbar(names_frame, orient="vertical", command=names_list.yview)
    names_list.configure(yscrollcommand=n_sb.set)
    n_sb.pack(side=RIGHT, fill=Y)


    separator(card, fill=X, padx=20, pady=4)

    notif = tk.Label(card, text="", bg=CARD, fg=ACCENT2, font=FONT_SMALL,
                     wraplength=700, anchor="w", justify="left")
    notif.pack(fill=X, padx=20, pady=4)

    countdown_lbl = tk.Label(card, text="", bg=CARD, fg=ACCENT, font=FONT_H3)
    countdown_lbl.pack(padx=20, anchor="w")

    pb_style = ttk.Style()
    pb_style.configure("blue.Horizontal.TProgressbar", troughcolor=BORDER,
                       background=ACCENT, thickness=10)
    pb = ttk.Progressbar(card, style="blue.Horizontal.TProgressbar",
                         orient="horizontal", mode="determinate")
    pb.pack(fill=X, padx=20, pady=(4, 16))

    btn_row = tk.Frame(card, bg=CARD)
    btn_row.pack(padx=20, pady=(0, 20), anchor="w")

    def fill_attendance():
        stream = att_stream_var.get().strip()
        sub    = subj_att_var.get().strip()
        mins   = timer_var.get()
        sem    = sem_att_var.get()
        if not stream:
            notif.config(text="⚠  Please select a stream.", fg=DANGER); return
        if not sub:
            notif.config(text="⚠  Please select a subject.", fg=DANGER); return
        if not os.path.exists(TRAIN_IMAGE_LABEL_PATH):
            notif.config(text="❌  Model not trained. Register & train first.", fg=DANGER); return
        notif.config(
            text=f"⏳  Starting: {stream} → Sem {sem} → {sub}...",
            fg=ACCENT)
        automaticAttedance.subjectChoose(text_to_speech, sub, mins * 60,
                                         notif, countdown_lbl, pb, stream, sem, names_list)

    b_start = make_styled_btn(btn_row, "▶  Start Session", fill_attendance)
    b_start.pack(side=LEFT, padx=(0, 12))
    add_hover(b_start)

    def open_folder():
        sub = subj_att_var.get().strip()
        path = os.path.join(ATTENDANCE_PATH, sub)
        if sub and os.path.exists(path):
            os.startfile(path)
        else:
            messagebox.showwarning("Not Found",
                f"No folder for '{sub}'. Take attendance first.")

    b_folder = make_styled_btn(btn_row, "📂  Open Folder", open_folder, bg="#64748B")
    b_folder.pack(side=LEFT)
    add_hover(b_folder, "#64748B", "#475569")

# ─────────────────────────────────────────────────────────────────────────────
#  REPORTS TAB
# ─────────────────────────────────────────────────────────────────────────────
def build_reports_tab(parent):
    inner = tk.Frame(parent, bg=BG)
    inner.pack(fill=BOTH, expand=True)

    card = make_card(inner)
    card.pack(fill=X, padx=24, pady=24) # Smaller vertical pady

    tk.Label(card, text="  📈  Attendance Insights (Range Analysis)", bg="#64748B", fg="white",
             font=FONT_H3, anchor="w", pady=8).pack(fill=X)

    filters = tk.Frame(card, bg=CARD)
    filters.pack(fill=X, padx=20, pady=16)

    # ── Filter Row 1: Stream, Semester & Subject ───────────────────────────
    tk.Label(filters, text="Stream:", bg=CARD, fg=TEXT, font=FONT_SMALL).grid(row=0, column=0, sticky="w")
    rep_stream_var = StringVar()
    rep_stream_combo = ttk.Combobox(filters, textvariable=rep_stream_var, values=get_streams(),
                                    font=FONT_SMALL, width=15, state="readonly")
    rep_stream_combo.grid(row=0, column=1, padx=(4, 12), pady=4)
    stream_combos.append(rep_stream_combo)

    tk.Label(filters, text="Semester:", bg=CARD, fg=TEXT, font=FONT_SMALL).grid(row=0, column=2, sticky="w")
    rep_sem_var = StringVar(value="All")
    rep_sem_combo = ttk.Combobox(filters, textvariable=rep_sem_var, 
                                 values=["All","1","2","3","4","5","6","7","8"],
                                 font=FONT_SMALL, width=6, state="readonly")
    rep_sem_combo.grid(row=0, column=3, padx=(4, 12), pady=4)

    tk.Label(filters, text="Subject:", bg=CARD, fg=TEXT, font=FONT_SMALL).grid(row=0, column=4, sticky="w")
    rep_sub_var = StringVar()
    rep_sub_combo = ttk.Combobox(filters, textvariable=rep_sub_var, font=FONT_SMALL,
                                 width=18, state="readonly")
    rep_sub_combo.grid(row=0, column=5, padx=(4, 0), pady=4)

    # ── Filter Row 2: Date Range ────────────────────────────────────────
    tk.Label(filters, text="Start Date:", bg=CARD, fg=TEXT, font=FONT_SMALL).grid(row=1, column=0, sticky="w", pady=(10, 0))
    start_date_var = StringVar()
    start_date_combo = ttk.Combobox(filters, textvariable=start_date_var, font=FONT_SMALL,
                                     width=22, state="readonly")
    start_date_combo.grid(row=1, column=1, padx=(4, 16), pady=(10, 0))

    tk.Label(filters, text="End Date:", bg=CARD, fg=TEXT, font=FONT_SMALL).grid(row=1, column=2, sticky="w", pady=(10, 0))
    end_date_var = StringVar()
    end_date_combo = ttk.Combobox(filters, textvariable=end_date_var, font=FONT_SMALL,
                                   width=22, state="readonly")
    end_date_combo.grid(row=1, column=3, padx=(4, 0), pady=(10, 0))

    notif_r = tk.Label(card, text="", bg=CARD, fg=ACCENT2, font=FONT_SMALL)
    notif_r.pack(fill=X, padx=20, pady=(0, 10))

    # ── Button Row ──────────────────────────────────────────────────────
    btn_row = tk.Frame(card, bg=CARD)
    btn_row.pack(fill=X, padx=20, pady=(0, 16))

    # --- STATS ROW ---
    stats_frame = tk.Frame(inner, bg=BG)
    stats_frame.pack(fill=X, padx=24, pady=(0, 20))

    def make_stat_card(parent, label, color):
        f = tk.Frame(parent, bg=WHITE, bd=0, highlightbackground=BORDER, highlightthickness=1)
        f.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 15))
        tk.Label(f, text=label, bg=WHITE, fg=TEXT_MUTED, font=FONT_SMALL).pack(pady=(12, 0))
        val_lbl = tk.Label(f, text="--", bg=WHITE, fg=color, font=("Segoe UI", 20, "bold"))
        val_lbl.pack(pady=(0, 12))
        return val_lbl

    lbl_total = make_stat_card(stats_frame, "Total Students", TEXT)
    lbl_avg   = make_stat_card(stats_frame, "Avg. Attendance", ACCENT)
    lbl_good  = make_stat_card(stats_frame, "Above 75%", ACCENT2)
    lbl_poor  = make_stat_card(stats_frame, "Below 50%", DANGER)

    def refresh_sub_list(e=None):
        strm = rep_stream_var.get()
        sem = rep_sem_var.get()
        alldata = load_sem_subjects()
        strm_data = alldata.get(strm, {})
        subs = []
        
        if sem == "All":
            for s_sem, subject_list in strm_data.items():
                subs.extend(subject_list)
        else:
            subs = strm_data.get(sem, [])
            
        subs = sorted(list(set(subs)))
        rep_sub_combo["values"] = subs
        
        # Preserve selection if valid
        curr = rep_sub_var.get()
        if subs:
            if curr not in subs:
                rep_sub_var.set(subs[0])
        else:
            rep_sub_var.set("")

    def refresh_dates(e=None):
        dates = show_attendance.get_all_dates()
        start_date_combo["values"] = dates
        end_date_combo["values"] = dates
        if dates:
            curr_s = start_date_var.get()
            curr_e = end_date_var.get()
            if not curr_s or curr_s not in dates: start_date_var.set(dates[0])
            if not curr_e or curr_e not in dates: end_date_var.set(dates[-1])

    rep_stream_combo.bind("<<ComboboxSelected>>", refresh_sub_list)
    rep_sem_combo.bind("<<ComboboxSelected>>",    refresh_sub_list)
    # Expose hook
    _refresh_reports_tab = lambda: (rep_stream_combo.config(values=get_streams()), refresh_sub_list(), refresh_dates())

    # NOTE: Hover sync removed as it was causing selection reset bugs. 
    # Global sync handle these via sync_all_streams().

    # Table
    tbl_card = make_card(inner)
    tbl_card.pack(fill=BOTH, expand=True, padx=24, pady=(0, 24))

    # --- SEARCH BAR ---
    search_frame = tk.Frame(tbl_card, bg=WHITE)
    search_frame.pack(fill=X, padx=15, pady=(15, 5))
    tk.Label(search_frame, text="  🔍  Quick Search:", bg=WHITE, fg=TEXT_MUTED, font=FONT_SMALL).pack(side=LEFT)
    search_var = StringVar()
    search_ent = tk.Entry(search_frame, textvariable=search_var, font=FONT_SMALL, width=40,
                          relief=SOLID, bd=1)
    search_ent.pack(side=LEFT, padx=12)
    tk.Label(search_frame, text="(Name or Enrollment)", bg=WHITE, fg="#94A3B8", font=("Segoe UI", 9)).pack(side=LEFT)

    rep_tree = ttk.Treeview(tbl_card, show="headings", height=12)
    tsb = tk.Scrollbar(tbl_card, orient="vertical", command=rep_tree.yview)
    rep_tree.configure(yscrollcommand=tsb.set)
    rep_tree.pack(side=LEFT, fill=BOTH, expand=True, padx=(15, 0), pady=(5, 15))
    tsb.pack(side=RIGHT, fill=Y, pady=(5, 15), padx=(0, 15))

    # Cache for search filtering
    all_report_data = []

    def on_search(*args):
        query = search_var.get().lower().strip()
        rep_tree.delete(*rep_tree.get_children())
        
        for row_vals, tag in all_report_data:
            # Check if query matches any column value
            if not query or any(query in str(v).lower() for v in row_vals):
                rep_tree.insert("", "end", values=row_vals, tags=(tag,))

    search_var.trace("w", on_search)

    def generate_report():
        strm = rep_stream_var.get()
        sem = rep_sem_var.get()
        sub = rep_sub_var.get()
        start = start_date_var.get()
        end = end_date_var.get()
        search_var.set("") # Clear search on new report
        
        if not sub or not start or not end:
            notif_r.config(text="⚠ Select all filters first.", fg=DANGER); return
        
        stats = show_attendance.generate_and_show(sub, start, end, rep_tree, notif_r, stream=strm, semester=sem)
        
        # Fill cache for search
        all_report_data.clear()
        for item in rep_tree.get_children():
            all_report_data.append((rep_tree.item(item, 'values'), rep_tree.item(item, 'tags')[0] if rep_tree.item(item, 'tags') else ""))

        if stats:
            lbl_total.config(text=str(stats['total']))
            lbl_avg.config(text=str(stats['avg']))
            lbl_good.config(text=str(stats['good']))
            lbl_poor.config(text=str(stats['poor']))
        else:
            for lbl in [lbl_total, lbl_avg, lbl_good, lbl_poor]: lbl.config(text="--")

    b_gen = make_styled_btn(btn_row, "📊  Generate Range Report", generate_report, bg=ACCENT2)
    b_gen.pack(side=LEFT, padx=(0, 12))
    add_hover(b_gen, ACCENT2, "#047857")

    def export_report():
        sub = rep_sub_var.get().strip()
        strm = rep_stream_var.get().strip()
        sem = rep_sem_var.get().strip()
        start = start_date_var.get()
        end = end_date_var.get()
        
        if not sub:
            notif_r.config(text="⚠  Select a subject first.", fg=DANGER); return
            
        try:
            notif_r.config(text="⏳  Generating logs report...", fg=ACCENT)
            # NEW: Gets Date-wise raw logs (5 columns)
            src_path = show_attendance.export_report(sub, start, end, strm, sem)
            
            suggested_name = f"Attendance_Logs_{strm}_{sub}_Sem{sem}.xlsx"
            
            filesave = filedialog.asksaveasfilename(
                initialfile=suggested_name,
                defaultextension=".xlsx",
                filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")],
                title="Save Date-wise Attendance Logs"
            )
            
            if filesave:
                import shutil
                shutil.copy(src_path, filesave)
                notif_r.config(text=f"✅ Date-wise logs exported successfully!", fg=ACCENT2)
                messagebox.showinfo("Export Successful", f"Date-wise logs saved at:\n{filesave}\n\n(Columns: Enrollment, Name, Date, Time, Status)")
            else:
                notif_r.config(text="⚠ Export cancelled.", fg=WARN)
                
        except Exception as e:
            notif_r.config(text=f"Error: {e}", fg=DANGER)
            messagebox.showerror("Export Error", f"Could not export logs: {e}")

    b_exp = make_styled_btn(btn_row, "💾  Export Report", export_report, bg="#7C3AED")
    b_exp.pack(side=LEFT, padx=(0, 12))
    add_hover(b_exp, "#7C3AED", "#6D28D9")

    def download_full_history():
        sub = rep_sub_var.get().strip()
        strm = rep_stream_var.get().strip()
        sem = rep_sem_var.get().strip()
        
        if not sub:
            notif_r.config(text="⚠  Select a subject first.", fg=DANGER); return
        
        try:
            notif_r.config(text="⏳  Fetching summary report...", fg=ACCENT)
            # NEW: Gets the Summary report (8 columns)
            src_path = show_attendance.export_full_history(sub)
            
            suggested_name = f"Attendance_Summary_{strm}_{sub}_Sem{sem}.xlsx"
            
            filesave = filedialog.asksaveasfilename(
                initialfile=suggested_name,
                defaultextension=".xlsx",
                filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")],
                title="Save Attendance Summary Report"
            )
            
            if filesave:
                import shutil
                shutil.copy(src_path, filesave)
                notif_r.config(text=f"✅ Summary report exported successfully!", fg=ACCENT2)
                messagebox.showinfo("Export Successful",
                    f"Summary report saved at:\n{filesave}\n\n"
                    f"📊 Columns: Enrollment, Name, Stream, Semester, Total, Attended, %, Status")
            else:
                notif_r.config(text="⚠ Export cancelled.", fg=WARN)
                
        except Exception as e:
            notif_r.config(text=f"Error: {e}", fg=DANGER)
            messagebox.showerror("Export Error", f"Could not export summary: {e}")

    b_of = make_styled_btn(btn_row, "📥  Download Full History", download_full_history, bg="#64748B")
    b_of.pack(side=LEFT)
    add_hover(b_of, "#64748B", "#475569")

    # ── Final Initialization ───────────────────────────────────────────
    refresh_dates()
    if get_streams():
        rep_stream_var.set(get_streams()[0])
        refresh_sub_list()

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────
window = Tk()
window.title("CLASS VISION – Smart Attendance Management System")
window.geometry("1100x700")
window.configure(background=BG)
window.resizable(True, True)
try:
    window.iconbitmap("AMS.ico")
except Exception:
    pass

# ── Top Header bar (Perfectly Centered & Aligned) ──────────────────────────
header = tk.Frame(window, bg=HEADER_BG, height=80)
header.pack(fill=X)
header.pack_propagate(False)

# Custom function for True Smooth Diagonal Gradient Text (Supports Multi-Colors)
def make_gradient_title(parent, text, font_size, colors, bg_hex):
    from PIL import Image, ImageDraw, ImageFont, ImageTk
    def h2r(h): return tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    rgb_colors = [h2r(c) for c in colors]
    bg_c = h2r(bg_hex)
    
    try: font = ImageFont.truetype("segoeuib.ttf", font_size + 10)
    except:
        try: font = ImageFont.truetype("arialbd.ttf", font_size + 10)
        except: font = ImageFont.load_default()

    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = (bbox[2] - bbox[0]) + 10, (bbox[3] - bbox[1]) + 15
    
    grad_src = Image.new("RGB", (w, h))
    for x in range(w):
        for y in range(h):
            # Calculate diagonal factor (0.0 to 1.0)
            factor = (x + y) / (w + h)
            
            # Interpolate between multiple colors
            num_colors = len(rgb_colors)
            if num_colors > 1:
                # Segment index
                pos = factor * (num_colors - 1)
                idx = int(pos)
                if idx >= num_colors - 1:
                    r, g, b = rgb_colors[-1]
                else:
                    inner_factor = pos - idx
                    c1, c2 = rgb_colors[idx], rgb_colors[idx+1]
                    r = int(c1[0] + (c2[0]-c1[0]) * inner_factor)
                    g = int(c1[1] + (c2[1]-c1[1]) * inner_factor)
                    b = int(c1[2] + (c2[2]-c1[2]) * inner_factor)
            else:
                r, g, b = rgb_colors[0]
                
            grad_src.putpixel((x, y), (r, g, b))
    
    mask = Image.new("L", (w, h), 0)
    draw_m = ImageDraw.Draw(mask)
    draw_m.text((0, 0), text, font=font, fill=255)
    
    final_img = Image.new("RGB", (w, h), bg_c)
    final_img.paste(grad_src, (0, 0), mask=mask)
    
    photo = ImageTk.PhotoImage(final_img)
    lbl = tk.Label(parent, image=photo, bg=bg_hex, bd=0, highlightthickness=0)
    lbl.image = photo
    return lbl

# Container for Title and Subtitle to handle alignment better
logo_frame = tk.Frame(header, bg=HEADER_BG)
logo_frame.place(relx=0, rely=0.44, anchor="w", x=24)

# Icon (Simple White)
tk.Label(logo_frame, text="🎓", bg=HEADER_BG, fg="white",
         font=("Segoe UI", 30)).pack(side=LEFT, padx=(0, 15))

# Text block
text_frame = tk.Frame(logo_frame, bg=HEADER_BG)
text_frame.pack(side=LEFT)

# Title with Smooth Diagonal Gradient (Green -> Blue)
title_lbl_smooth = make_gradient_title(text_frame, "CLASS VISION", 27, ["#00F260", "#0575E6"], HEADER_BG)
title_lbl_smooth.grid(row=0, column=0, sticky="w", pady=(8, 0))

tk.Label(text_frame, text="Smart Attendance Management System", bg=HEADER_BG, 
         fg="#CBD5E1", font=("Segoe UI", 10, "bold"), padx=0, pady=0).grid(row=1, column=0, sticky="nw", pady=(0, 0))

time_lbl = tk.Label(header, text="", bg=HEADER_BG, fg="#CBD5E1", font=FONT_BODY)
time_lbl.place(relx=1.0, rely=0.5, x=-24, anchor="e")

def update_clock():
    now = datetime.datetime.now().strftime("%A, %d %b %Y   %H:%M:%S")
    time_lbl.config(text=now)
    window.after(1000, update_clock)
update_clock()

# ── Notebook (tabs) ────────────────────────────────────────────────────────
nb_style = ttk.Style()
nb_style.theme_use("clam")
nb_style.configure("TNotebook", background=BG, borderwidth=0)
nb_style.configure("TNotebook.Tab", font=FONT_BTN, padding=[16, 8],
                   background=BORDER, foreground=TEXT_MUTED)
nb_style.map("TNotebook.Tab",
             background=[("selected", WHITE)],
             foreground=[("selected", ACCENT)],
             expand=[("selected", [1, 1, 1, 0])])

nb = ttk.Notebook(window)
nb.pack(fill=BOTH, expand=True, padx=0, pady=0)

tab_attend = tk.Frame(nb, bg=BG)
tab_admin  = tk.Frame(nb, bg=BG)
tab_report = tk.Frame(nb, bg=BG)

nb.add(tab_attend, text="  📡  Take Attendance  ")
nb.add(tab_admin,  text="  🔐  Admin Panel  ")
nb.add(tab_report, text="  📊  Reports  ")

# ── Status bar ─────────────────────────────────────────────────────────────
status_bar = tk.Frame(window, bg=HEADER_BG, height=28)
status_bar.pack(fill=X, side=BOTTOM)

# Initial load for status bar
cfg_init = load_camera_config()
init_ip = cfg_init.get("droidcam_ip", "192.168.1.7")
init_port = cfg_init.get("droidcam_port", "4747")

status_cam_lbl = tk.Label(status_bar, text=f"  🟢  System Ready  |  DroidCam: http://{init_ip}:{init_port}",
         bg=HEADER_BG, fg="#64748B", font=FONT_SMALL)
status_cam_lbl.pack(side=LEFT, pady=4)

tk.Label(status_bar, text="CLASS VISION v3.0  ",
         bg=HEADER_BG, fg="#334155", font=FONT_SMALL).pack(side=RIGHT, pady=4)

# ── Build Tabs ─────────────────────────────────────────────────────────────
build_attendance_tab(tab_attend)
build_admin_tab(tab_admin)
build_reports_tab(tab_report)

window.mainloop()
