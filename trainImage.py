import os
import cv2
import numpy as np
from PIL import Image


def TrainImage(haarcasecade_path, trainimage_path, trainimagelabel_path,
               message, text_to_speech):

    def _update(msg, color="#f5c518"):
        try:
            message.configure(text=msg, fg=color)
        except Exception:
            pass

    try:
        _update("⚙  Loading training data…", "#00d4ff")

        # Stable Presentation Settings (8x8 grid)
        recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        detector   = cv2.CascadeClassifier(haarcasecade_path)

        faces, Ids = getImagesAndLabels(trainimage_path)

        if len(faces) == 0:
            _update("❌  No training images found. Register a student first.", "#ff4444")
            text_to_speech("No training images found. Please register a student first.")
            return

        _update(f"⚙  Training on {len(faces)} images…", "#00d4ff")
        recognizer.train(faces, np.array(Ids))

        # ensure directory exists
        os.makedirs(os.path.dirname(trainimagelabel_path), exist_ok=True)
        recognizer.save(trainimagelabel_path)

        unique_students = len(set(Ids))
        res = f"✅  Model trained on {len(faces)} images from {unique_students} student(s)."
        _update(res, "#44ff88")
        text_to_speech("Model trained successfully.")

    except Exception as ex:
        _update(f"❌  Training error: {ex}", "#ff4444")
        print(f"[TrainImage] Error: {ex}")


def getImagesAndLabels(path):
    import pandas as pd
    faces, Ids = [], []

    if not os.path.exists(path):
        return faces, Ids

    try:
        df = pd.read_csv("StudentDetails/studentdetails.csv")
        df["Enrollment"] = df["Enrollment"].astype(str)
        enrollment_to_id = {row['Enrollment']: idx for idx, row in df.iterrows()}
    except Exception:
        enrollment_to_id = {}

    student_dirs = [os.path.join(path, d) for d in os.listdir(path)
                    if os.path.isdir(os.path.join(path, d))]

    for student_dir in student_dirs:
        for img_file in os.listdir(student_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(student_dir, img_file)
            try:
                parts = os.path.splitext(img_file)[0].split("_")
                # filename: Name_Enrollment_SampleNum.jpg
                enrollment_str = str(parts[1])
                
                if enrollment_str not in enrollment_to_id:
                    continue
                
                mapped_id = enrollment_to_id[enrollment_str]
                pil_img = Image.open(img_path).convert("L")
                img_np  = np.array(pil_img, "uint8")
                
                # [NEW] Normalize face for consistency
                img_np = cv2.resize(img_np, (200, 200))
                img_np = cv2.equalizeHist(img_np)
                
                faces.append(img_np)
                Ids.append(mapped_id)
            except Exception as ex:
                print(f"[TrainImage] Skipping {img_file}: {ex}")
                continue

    return faces, Ids
