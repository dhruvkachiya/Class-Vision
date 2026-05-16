import csv
import os
import cv2
import numpy as np
import pandas as pd
import datetime
import time

import json

CAMERA_CONFIG_PATH = "./camera_config.json"

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
    
    # 1. Check for green dominance (common DroidCam glitch)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    green_ratio = cv2.countNonZero(green_mask) / (frame.shape[0] * frame.shape[1])
    if green_ratio > 0.5: return True
    
    # 2. Check for horizontal stripes/noise
    # We look at the variance of the average brightness of each row
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    row_averages = np.mean(gray, axis=1)
    # If there's a huge jump between adjacent rows (noise/lines), it's corrupted
    row_diffs = np.abs(np.diff(row_averages))
    if np.max(row_diffs) > 80: # Sudden brightness jump in lines
        return True
        
    return False


def _window_closed(win_name):
    """Check if OpenCV window was closed via X button."""
    try:
        return cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
    except Exception:
        return True


def _update_label(label, msg, color="#2563EB"):
    try:
        if label and label.winfo_exists():
            label.configure(text=msg, fg=color)
    except Exception:
        pass


def TakeImage(l1, l2, stream, semester,
              haarcasecade_path, trainimage_path,
              message, err_screen, text_to_speech):

    if not l1 or not l2:
        err_screen()
        return

    try:
        cam = _open_camera()
        if cam is None:
            _update_label(message, "❌  Camera not found. Check DroidCam.", "#DC2626")
            return

        detector   = cv2.CascadeClassifier(haarcasecade_path)
        Enrollment = str(l1).strip()
        Name       = str(l2).strip()

        # [NEW] FACE DUPLICATE CHECK - Prevent registering the same face with a different ID
        TRAIN_LABEL_PATH = "TrainingImageLabel/Trainner.yml"
        if os.path.exists(TRAIN_LABEL_PATH):
            _update_label(message, "🔍  Checking if face is already registered…", "#2563EB")
            try:
                # Stable Presentation Settings (8x8 grid)
                recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
                recognizer.read(TRAIN_LABEL_PATH)
                df_check = pd.read_csv("StudentDetails/studentdetails.csv", dtype=str)
                df_check = df_check.astype(str)

                # Multi-frame reliability check
                WIN_NAME = "Register Student - Press Q to quit early"
                check_count = 0
                detections = {} # Track how many times each ID is seen

                while check_count < 40: # Scan for 40 valid face frames
                    ret, img = cam.read()
                    if not ret or img is None:
                        continue
                    # Skip corrupted frames
                    if _is_corrupted(img):
                        continue
                    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(gray, 1.3, 5)

                    if len(faces) == 0:
                        cv2.rectangle(img, (0, 0), (640, 40), (0, 0, 255), -1)
                        cv2.putText(img, "FACE NOT DETECTED - PLEASE SHOW FACE", (10, 26), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    else:
                        for (x, y, w, h) in faces:
                            cv2.rectangle(img, (x, y), (x+w, y+h), (37, 99, 235), 2)
                            face_roi = gray[y:y+h, x:x+w]
                            face_roi = cv2.resize(face_roi, (200, 200))
                            face_roi = cv2.equalizeHist(face_roi)

                            Id, conf = recognizer.predict(face_roi)
                            if conf < 75: 
                                detections[Id] = detections.get(Id, 0) + 1
                        
                        check_count += 1
                        cv2.rectangle(img, (0, 0), (640, 40), (37, 99, 235), -1)
                        cv2.putText(img, f"Scanning Registration... ({check_count}/40)", (10, 26), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.imshow(WIN_NAME, img)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), ord('Q'), 27) or _window_closed(WIN_NAME):
                        cam.release()
                        cv2.destroyAllWindows()
                        return

                    # FINAL DECISION: If an ID was seen in at least 8 frames, show a warning
                    for Id, count in detections.items():
                        if count >= 8: # Reliable detection
                            if 0 <= Id < len(df_check):
                                existing_enroll = str(df_check.iloc[Id]["Enrollment"])
                                existing_name   = str(df_check.iloc[Id]["Name"])

                                # [CHANGE] Only Warn on screen, Don't speak or Block for presentation
                                if str(existing_enroll).strip() != str(Enrollment).strip():
                                    _update_label(message, f"Warning: Face looks like {existing_name}", "#D97706")
                                    time.sleep(1) 
                                    break
                                else:
                                    print(f"[TakeImage] Updating existing student: {existing_name}")
                                    break
            except Exception as e:
                print(f"[TakeImage] Skip duplicate check: {e}")

        sampleNum  = 0
        directory  = f"{Enrollment}_{Name}"
        path       = os.path.join(trainimage_path, directory)

        if os.path.exists(path):
            # Wipe old images so re-registration works
            import shutil
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

        _update_label(message, f"📸  Capturing images for {Name}… (0/50)", "#2563EB")

        WIN_NAME = "Register Student - Press Q to quit early"
        last_frame_time = time.time()
        while True:
            ret, img = cam.read()
            if not ret or img is None:
                if time.time() - last_frame_time > 10:
                    _update_label(message, "❌  Camera signal lost.", "#DC2626")
                    break
                cv2.waitKey(1)
                continue

            # Skip corrupted/green frames
            if _is_corrupted(img):
                cv2.waitKey(1)
                continue

            last_frame_time = time.time()
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            # [FIX] Ensure only one face is captured at a time to prevent mixing training data
            if len(faces) == 0:
                cv2.putText(img, "FACE NOT DETECTED", (180, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                _update_label(message, "⚠  Please show your face to start capture.", "#D97706")
            elif len(faces) > 1:
                cv2.putText(img, "WARNING: Multiple faces detected!", (50, 450), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                _update_label(message, "⚠  Multiple faces detected. Please stand alone.", "#DC2626")
            elif len(faces) == 1:
                for (x, y, w, h) in faces:
                    sampleNum += 1
                    pct = int((sampleNum / 50) * 100)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (37, 99, 235), 2)
                    cv2.putText(img, f"{Name}  {pct}%",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (37, 99, 235), 2)
                    
                    # Store raw crop (normalization happens during training/recognition)
                    face_roi = gray[y:y+h, x:x+w]
                    
                    cv2.imwrite(
                        os.path.join(path, f"{Name}_{Enrollment}_{sampleNum}.jpg"),
                        face_roi
                    )
                    _update_label(message, f"Capturing... {sampleNum}/50", "#2563EB")

            # Progress bar on frame
            cv2.rectangle(img, (0, 460), (int(640*(sampleNum/50)), 480), (37,99,235), -1)
            cv2.imshow(WIN_NAME, img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or _window_closed(WIN_NAME):
                break
            if sampleNum >= 50:
                break

        cam.release()
        cv2.destroyAllWindows()

        if sampleNum == 0:
            _update_label(message, "⚠  No face detected. Improve lighting & try again.", "#D97706")
            text_to_speech("No face detected.")
            return

        # Save / update student CSV
        csv_path = "StudentDetails/studentdetails.csv"
        try:
            df = pd.read_csv(csv_path, dtype=str)
        except Exception:
            df = pd.DataFrame(columns=["Enrollment","Name","Stream","Semester"])

        # Force string types and replace/add the student record
        df = df.astype(str)
        df = df[df["Enrollment"] != Enrollment] # Remove duplicate ID if exists
        
        new_row = pd.DataFrame([[Enrollment, Name, stream, semester]],
                               columns=["Enrollment","Name","Stream","Semester"])
        df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(csv_path, index=False)

        res = f"✅  Saved {sampleNum} images for {Name}. Click 'Train Model' now."
        _update_label(message, res, "#059669")
        text_to_speech(f"Images saved for {Name}. Please train the model.")

    except Exception as ex:
        _update_label(message, f"Error: {ex}", "#DC2626")
        print(f"[TakeImage] Error: {ex}")
