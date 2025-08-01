import streamlit as st
import zipfile
import os
import cv2
import pytesseract
import tempfile
import shutil
import re
import numpy as np
import ffmpeg
from datetime import timedelta

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # à adapter selon installation

def extract_frames(video_path, fps_sample=1):
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25  # fallback
    interval = int(fps / fps_sample)
    frames = []
    frame_count = 0
    success, image = vidcap.read()
    while success:
        if frame_count % interval == 0:
            frames.append((frame_count, image.copy()))
        success, image = vidcap.read()
        frame_count += 1
    vidcap.release()
    return frames, fps

def detect_bib_number(frame, bib_numbers):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    found = []
    for bib in bib_numbers:
        if re.search(rf'\b{bib}\b', text):
            found.append(bib)
    return found

def get_video_segments(timestamps, padding=2, min_gap=60):
    segments = []
    if not timestamps:
        return segments
    timestamps.sort()
    start = timestamps[0] - padding
    end = timestamps[0] + padding
    for t in timestamps[1:]:
        if t - end > min_gap:
            segments.append((max(0, start), end))
            start = t - padding
        end = t + padding
    segments.append((max(0, start), end))
    return segments

def extract_video_segment(video_path, start_sec, end_sec, output_path):
    (
        ffmpeg
        .input(video_path, ss=start_sec, t=end_sec - start_sec)
        .output(output_path, codec="copy")
        .run(overwrite_output=True, quiet=True)
    )

def main():
    st.title("Détection de passages vidéo par numéro de dossard")

    zip_file = st.file_uploader("Chargez un fichier .zip contenant des vidéos MP4", type="zip")
    bib_input = st.text_input("Entrez les numéros de dossard séparés par des virgules")

    if zip_file and bib_input:
        bib_numbers = [b.strip() for b in bib_input.split(",") if b.strip()]

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "videos.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_file.read())

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            video_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".mp4")]

            result_dir = os.path.join(tmpdir, "result")
            os.makedirs(result_dir, exist_ok=True)

            for video_path in video_files:
                st.write(f"Analyse de {os.path.basename(video_path)}...")

                # Lecture initiale du FPS pour déterminer fps_sample
                vidcap = cv2.VideoCapture(video_path)
                fps = vidcap.get(cv2.CAP_PROP_FPS)
                vidcap.release()
                if fps == 0:
                    fps = 25
                fps_sample = fps / 6

                frames, _ = extract_frames(video_path, fps_sample=fps_sample)

                timestamps_detected = []
                for frame_idx, frame in frames:
                    seconds = frame_idx / fps
                    detected = detect_bib_number(frame, bib_numbers)
                    if detected:
                        timestamps_detected.append(seconds)

                st.write(f"{len(timestamps_detected)} détections pour {os.path.basename(video_path)}")

                segments = get_video_segments(timestamps_detected, padding=2, min_gap=60)
                for i, (start, end) in enumerate(segments):
                    out_path = os.path.join(result_dir, f"{os.path.basename(video_path).replace('.mp4','')}_segment_{i+1}.mp4")
                    extract_video_segment(video_path, start, end, out_path)

            # Création d'un zip avec les extraits
            output_zip = os.path.join(tmpdir, "extraits_dossards.zip")
            with zipfile.ZipFile(output_zip, "w") as zipf:
                for root, _, files in os.walk(result_dir):
                    for file in files:
                        zipf.write(os.path.join(root, file), arcname=file)

            with open(output_zip, "rb") as f:
                st.download_button("Télécharger les extraits", f, file_name="extraits_dossards.zip")

if __name__ == "__main__":
    main()
