import os
import cv2
import numpy as np
import pandas as pd
import datetime
import time
import threading

import json

HAARCASCADE_PATH    = "haarcascade_frontalface_default.xml"
TRAIN_LABEL_PATH    = "TrainingImageLabel/Trainner.yml"
STUDENT_DETAIL_PATH = "StudentDetails/studentdetails.csv"
ATTENDANCE_PATH     = "Attendance"
CAMERA_CONFIG_PATH  = "./camera_config.json"

def _load_cam_srcs():
    try:
        with open(CAMERA_CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        ip = cfg.get("droidcam_ip", "192.168.1.7")
        port = cfg.get("droidcam_port", "4747")
        return [
            f"http://{ip}:{port}/mjpegfeed",
            f"http://{ip}:{port}/video",
            1, 0
        ]
    except Exception:
        return ["http://192.168.1.7:4747/mjpegfeed", 1, 0]

def _open_camera():
    srcs = _load_cam_srcs()
    for src in srcs:
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            # Set MJPEG codec
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Wait for camera to stabilize
            time.sleep(0.5)
            
            good_frames = 0
            for _ in range(10):
                ret, frame = cap.read()
                if ret and frame is not None and not _is_corrupted(frame):
                    good_frames += 1
            
            if good_frames >= 3:
                print(f"[Camera] Connected: {src}")
                return cap
        cap.release()
    return None


def _is_corrupted(frame):
    """Detect corrupted/green/striped frames from DroidCam."""
    if frame is None: return True
    
    # 1. Check for green dominance
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    green_ratio = cv2.countNonZero(green_mask) / (frame.shape[0] * frame.shape[1])
    if green_ratio > 0.5: return True
    
    # 2. Check for horizontal stripes/noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    row_averages = np.mean(gray, axis=1)
    row_diffs = np.abs(np.diff(row_averages))
    if np.max(row_diffs) > 80:
        return True
        
    return False


def _window_closed(win_name):
    """Check if OpenCV window was closed via X button."""
    try:
        return cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
    except Exception:
        return True

def _safe(label, text, color="#2563EB"):
    try:
        if label and label.winfo_exists():
            label.after(0, lambda: label.config(text=text, fg=color))
    except Exception:
        pass

def _safe_pb(pb, value):
    try:
        if pb and pb.winfo_exists():
            pb.after(0, lambda: pb.config(value=value))
    except Exception:
        pass

def _safe_countdown(label, text):
    try:
        if label and label.winfo_exists():
            label.after(0, lambda: label.config(text=text))
    except Exception:
        pass


def _safe_list_update(listbox, text, clear=False):
    try:
        if listbox and listbox.winfo_exists():
            if clear:
                listbox.after(0, lambda: listbox.delete(0, 'end'))
            else:
                listbox.after(0, lambda: listbox.insert('end', text))
                listbox.after(0, lambda: listbox.see('end'))
    except Exception:
        pass

def subjectChoose(text_to_speech, subject, duration_secs,
                  notif_label, countdown_label, progress_bar, req_stream, req_sem, names_list=None):
    """
    Called directly from attendance.py.
    No Tkinter window is created here; UI elements are passed in.
    Runs the recognition loop in a background thread.
    """
    def _run():
        try:
            # Stable Presentation Settings (8x8 grid)
            recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
            if not os.path.exists(TRAIN_LABEL_PATH):
                _safe(notif_label, "❌  Model not found. Train first.", "#DC2626")
                return
            recognizer.read(TRAIN_LABEL_PATH)

            face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
            df = pd.read_csv(STUDENT_DETAIL_PATH, dtype=str)
            df = df.astype(str)

            cam = _open_camera()
            if cam is None:
                _safe(notif_label, "❌  Camera not found. Check DroidCam.", "#DC2626")
                return

            _safe_list_update(names_list, "", clear=True)
            added_names = set()

            col_names  = ["Enrollment", "Name"]
            attendance = pd.DataFrame(columns=col_names)
            cv_font    = cv2.FONT_HERSHEY_SIMPLEX

            # [NEW] Multi-frame validation: Track how many times each ID is seen
            validation_counter = {} 
            VALIDATION_THRESHOLD = 15 # Frames needed to confirm attendance

            start_time      = time.time()
            last_frame_time = time.time()

            _safe(notif_label,
                  f"Camera active! Scanning for '{subject}'...", "#059669")

            WIN_NAME = "CLASS VISION - Taking Attendance (ESC to stop)"

            while True:
                elapsed  = time.time() - start_time
                remain   = max(0, int(duration_secs - elapsed))
                progress = min(100, (elapsed / duration_secs) * 100)

                _safe_pb(progress_bar, progress)
                _safe_countdown(countdown_label,
                                f"Time remaining: {remain}s")

                if elapsed >= duration_secs:
                    break

                ret, im = cam.read()
                if not ret or im is None:
                    if time.time() - last_frame_time > 10:
                        _safe(notif_label, "❌  Camera signal lost.", "#DC2626")
                        break
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # Skip corrupted/green frames
                if _is_corrupted(im):
                    cv2.waitKey(1)
                    continue

                last_frame_time = time.time()
                gray  = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2, 5)

                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    # Resize and equalize to match training normalization
                    face_roi = cv2.resize(face_roi, (200, 200))
                    face_roi = cv2.equalizeHist(face_roi)

                    Id, conf = recognizer.predict(face_roi)
                    # Ultra-Strict threshold (50) for final presentation safety
                    recognized = conf < 50

                    if recognized and 0 <= Id < len(df):
                        row  = df.iloc[Id]
                        real_enroll = row["Enrollment"]
                        disp_name   = row["Name"]
                        student_stream = str(row.get("Stream", ""))

                        # 1. Anti-Spoofing Check (Liveness)
                        face_img = gray[y:y+h, x:x+w]
                        laplacian_var = cv2.Laplacian(face_img, cv2.CV_64F).var()
                        
                        # Check glare (Phone screens often have bright reflection spots)
                        hist = cv2.calcHist([face_img], [0], None, [256], [0, 256])
                        glare_ratio = sum(hist[240:]) / (sum(hist) + 1e-5)

                        is_spoof = laplacian_var < 45 or glare_ratio > 0.15

                        if is_spoof:
                            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 100, 255), 3) # Red-Orange
                            cv2.putText(im, f"! Spoof (C:{int(conf)})", (x, y-12),
                                        cv_font, 0.55, (0, 100, 255), 2)
                        elif student_stream.strip().lower() != req_stream.strip().lower():
                            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 165, 255), 3) # Orange
                            cv2.putText(im, f"X Stream (C:{int(conf)})", (x, y-12),
                                        cv_font, 0.55, (0, 165, 255), 2)
                        else:
                            # Genuine face detected
                            enroll = str(real_enroll)
                            validation_counter[enroll] = validation_counter.get(enroll, 0) + 1
                            
                            # Only mark present if seen enough times to be SURE
                            if validation_counter[enroll] >= VALIDATION_THRESHOLD:
                                attendance.loc[len(attendance)] = [real_enroll, disp_name]
                                
                                # Update UI Listbox
                                entry_text = f" {real_enroll} - {disp_name}"
                                if entry_text not in added_names:
                                    added_names.add(entry_text)
                                    _safe_list_update(names_list, entry_text)

                            cv2.rectangle(im, (x, y), (x+w, y+h), (5, 150, 50), 3)
                            # Show confirmation progress on screen
                            confirm_pct = min(100, int((validation_counter[enroll]/VALIDATION_THRESHOLD)*100))
                            cv2.putText(im, f"CONFIRMING {confirm_pct}%", (x, y+h+20),
                                        cv_font, 0.5, (5, 150, 50), 1)
                            cv2.putText(im, f"OK {disp_name} ({int(conf)})", (x, y-12),
                                        cv_font, 0.65, (5, 150, 50), 2)


                    else:
                        # Red box + cross
                        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 50, 220), 3)
                        cv2.line(im, (x+8, y+8), (x+w-8, y+h-8), (0, 50, 220), 3)
                        cv2.line(im, (x+w-8, y+8), (x+8, y+h-8), (0, 50, 220), 3)
                        cv2.putText(im, f"X Unknown ({int(conf)})", (x, y-12),
                                    cv_font, 0.65, (0, 50, 220), 2)

                attendance = attendance.drop_duplicates(["Enrollment"], keep="first")

                # Timer overlay at top
                mins_r, secs_r = divmod(remain, 60)
                overlay = f"Subject: {subject}  |  {mins_r:02d}:{secs_r:02d} remaining"
                cv2.rectangle(im, (0, 0), (640, 38), (30, 41, 59), -1)
                cv2.putText(im, overlay, (10, 26),
                            cv_font, 0.62, (245, 197, 24), 2)

                cv2.imshow(WIN_NAME, im)
                key = cv2.waitKey(30) & 0xFF
                if key == 27 or _window_closed(WIN_NAME):
                    break

            cam.release()
            cv2.destroyAllWindows()

            _safe_countdown(countdown_label, "")
            _safe_pb(progress_bar, 100)

            if len(attendance) == 0:
                _safe(notif_label,
                      "⚠  No faces recognized. Check training & lighting.", "#D97706")
                text_to_speech("No faces recognized.")
                return

            # ── Save CSV & Excel with accurate datetime ──────────────────────
            ts        = time.time()
            dt        = datetime.datetime.fromtimestamp(ts)
            date_str  = dt.strftime("%Y-%m-%d")
            time_str  = dt.strftime("%H-%M-%S")  # file-safe
            disp_time = dt.strftime("%H:%M:%S")  # human readable

            path_dir  = os.path.join(ATTENDANCE_PATH, subject)
            os.makedirs(path_dir, exist_ok=True)
            file_name_base = f"{subject}_{date_str}_{time_str}"
            file_path = os.path.join(path_dir, f"{file_name_base}.csv")
            excel_path = os.path.join(path_dir, f"{file_name_base}.xlsx")

            attendance["Date"]     = dt.strftime("%d-%m-%y") # Shorter format to avoid ### in Excel
            attendance["Time"]     = disp_time
            attendance["Status"]   = "Present"
            attendance["Stream"]   = req_stream
            attendance["Semester"] = req_sem
            
            # Save CSV for app internal use
            attendance.to_csv(file_path, index=False)
            
            # Save Excel for user convenience with auto-adjusted columns
            try:
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    attendance.to_excel(writer, index=False, sheet_name='Attendance')
                    worksheet = writer.sheets['Attendance']
                    for idx, col in enumerate(attendance.columns):
                        max_len = max(attendance[col].astype(str).map(len).max(), len(col)) + 2
                        worksheet.column_dimensions[chr(65+idx)].width = max_len
            except Exception as e:
                print(f"[ExcelExport] Error: {e}")

            msg = (f"✅  Attendance saved! {len(attendance)} student(s) marked present "
                   f"at {disp_time} on {date_str}.")
            _safe(notif_label, msg, "#059669")
            text_to_speech(f"Attendance saved for {subject}. "
                           f"{len(attendance)} students marked present.")

        except Exception as ex:
            _safe(notif_label, f"Error: {ex}", "#DC2626")
            print(f"[AutoAttendance] Error: {ex}")
            cv2.destroyAllWindows()

    threading.Thread(target=_run, daemon=True).start()
