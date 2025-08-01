import cv2
from ultralytics import YOLO
import threading
import tkinter as tk
from tkinter import filedialog
import time

# === Variables de contrôle ===
running = False
video_path = None
use_webcam = False
use_ip_camera = False
ip_camera_url = ""

# === Variables pour stocker les max ===
max_persons = 0
max_bikes = 0
max_suitcases = 0
max_knives = 0

person_start_time = None

def detection_video():
    global running, video_path, use_webcam, use_ip_camera, ip_camera_url
    global max_persons, max_bikes, max_suitcases, max_knives
    global person_start_time

    running = True
    model = YOLO("yolov8n.pt")

    max_persons = max_bikes = max_suitcases = max_knives = 0
    person_start_time = None

    if use_webcam:
        cap = cv2.VideoCapture(0)
    elif use_ip_camera:
        cap = cv2.VideoCapture(ip_camera_url)
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la source vidéo.")
        return

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results[0].boxes

        person_count = sum(1 for box in detections if int(box.cls) == 0)
        bike_count = sum(1 for box in detections if int(box.cls) == 1)
        suitcase_count = sum(1 for box in detections if int(box.cls) == 28)
        knife_count = sum(1 for box in detections if int(box.cls) == 44)

        max_persons = max(max_persons, person_count)
        max_bikes = max(max_bikes, bike_count)
        max_suitcases = max(max_suitcases, suitcase_count)
        max_knives = max(max_knives, knife_count)

        # === Affichage vidéo annotée ===
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, f"Personnes : {person_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Velos : {bike_count}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"Valises : {suitcase_count}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        cv2.putText(annotated_frame, f"Couteaux : {knife_count}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Détection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n=== Résumé de la détection ===")
    print(f"Nombre max de personnes détectées simultanément : {max_persons}")
    print(f"Nombre max de vélos détectés simultanément : {max_bikes}")
    print(f"Nombre max de valises détectées simultanément : {max_suitcases}")
    print(f"Nombre max de couteaux détectés simultanément : {max_knives}\n")

    root.quit()

def launch_webcam():
    global use_webcam, use_ip_camera, video_path
    use_webcam = True
    use_ip_camera = False
    video_path = None
    threading.Thread(target=detection_video).start()

def select_video():
    global use_webcam, use_ip_camera, video_path
    file = filedialog.askopenfilename(title="Choisir une vidéo MP4", filetypes=[("Fichiers MP4", "*.mp4")])
    if file:
        use_webcam = False
        use_ip_camera = False
        video_path = file
        threading.Thread(target=detection_video).start()

def launch_ip_camera():
    global use_webcam, use_ip_camera, ip_camera_url, video_path
    ip_camera_url = "http://192.168.208.165:8080/video"
    use_webcam = False
    use_ip_camera = True
    video_path = None
    threading.Thread(target=detection_video).start()

def stop_detection():
    global running
    running = False

root = tk.Tk()
root.title("Détection objets - Webcam, Vidéo ou Caméra IP")
root.geometry("450x260")

tk.Label(root, text="Sélectionne une source de vidéo :", font=("Arial", 14)).pack(pady=10)
tk.Button(root, text="Utiliser Webcam", command=launch_webcam, font=("Arial", 12)).pack(pady=5)
tk.Button(root, text="Choisir une vidéo MP4", command=select_video, font=("Arial", 12)).pack(pady=5)
tk.Button(root, text="Utiliser flux téléphone IP", command=launch_ip_camera, font=("Arial", 12)).pack(pady=5)
tk.Button(root, text="Arrêter", command=stop_detection, bg="red", fg="white", font=("Arial", 12)).pack(pady=10)

root.mainloop()
