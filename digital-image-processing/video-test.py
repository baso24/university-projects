import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

def is_fallen(p1, p2):
    # Se la distanza verticale è minore di quella orizzontale -> Fall detection
    return abs(p1[1] - p2[1]) < abs(p1[0] - p2[0])

def run_video(model_path, video_path, conf_threshold):
    print(f"Caricamento modello: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Errore caricamento modello: {e}")
        return

    if not os.path.exists(video_path):
        print(f"Errore: Il file video '{video_path}' non esiste.")
        return

    print(f"Apertura video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Errore: Impossibile aprire il video.")
        return

    print("Avvio elaborazione video. Premi 'q' per uscire anticipatamente.")

    # 0: testa, 1: braccia, 2: torso, 3: gambe, 4: piedi
    class_colors = {
        0: (0, 255, 255),   # Giallo (Testa)
        1: (0, 255, 0),     # Verde (Braccia)
        2: (255, 0, 0),     # Blu (Torso)
        3: (0, 165, 255),   # Arancione (Gambe)
        4: (128, 0, 128)    # Viola (Piedi)
    }

    cv2.namedWindow('YOLO Video Fall Detection', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fine del video o errore lettura frame.")
            break

        # Inferenza YOLO
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        result = results[0]

        # Copia del frame per creare l'overlay delle maschere
        overlay = frame.copy()
        
        # Struttura per accumulare i momenti
        class_moments = {k: {'m10': 0.0, 'm01': 0.0, 'm00': 0.0} for k in class_colors}
        
        if result.masks:
            masks_xy = result.masks.xy
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for i, poly in enumerate(masks_xy):
                cls_id = classes[i]
                color = class_colors.get(cls_id, (200, 200, 200))
                
                # Disegna maschera piena sull'overlay se il poligono è valido
                if len(poly) > 0:
                    pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts], color)
                    
                    # Calcola momenti per i centroidi
                    M = cv2.moments(pts)
                    if M["m00"] != 0:
                        if cls_id in class_moments:
                            class_moments[cls_id]['m10'] += M["m10"]
                            class_moments[cls_id]['m01'] += M["m01"]
                            class_moments[cls_id]['m00'] += M["m00"]

        # Calcolo centroidi unici per classe (escludendo braccia)
        final_centroids = {}
        for cid, m in class_moments.items():
            if cid == 1: # Skip braccia per il calcolo dello scheletro
                continue
            if m['m00'] != 0:
                cX = int(m['m10'] / m['m00'])
                cY = int(m['m01'] / m['m00'])
                final_centroids[cid] = (cX, cY)

        # Blending trasparenza
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Disegno skeleton: Testa(0) -> Torso(2) -> Gambe(3) -> Piedi(4)
        skeleton_links = [(0, 2), (2, 3), (3, 4)]
        for cls_a, cls_b in skeleton_links:
            if cls_a in final_centroids and cls_b in final_centroids:
                cv2.line(frame, final_centroids[cls_a], final_centroids[cls_b], (255, 255, 255), 2)

        # Disegno centroidi
        for pt in final_centroids.values():
            cv2.circle(frame, pt, 6, (0, 0, 0), -1)     # Bordo nero
            cv2.circle(frame, pt, 4, (255, 0, 0), -1)   # Centro rosso

        fall_detected = False
        pair_found = False

        # Case 1: Testa (0) e Torso (2)
        if 0 in final_centroids and 2 in final_centroids:
            pair_found = True
            if is_fallen(final_centroids[0], final_centroids[2]):
                fall_detected = True

        # Case 2: Testa (0) e Gambe (3)
        if not fall_detected and 0 in final_centroids and 3 in final_centroids:
            pair_found = True
            if is_fallen(final_centroids[0], final_centroids[3]):
                fall_detected = True

        # Case 3: Torso (2) e Gambe (3)
        if not fall_detected and 2 in final_centroids and 3 in final_centroids:
            pair_found = True
            if is_fallen(final_centroids[2], final_centroids[3]):
                fall_detected = True

        if pair_found:
            if fall_detected:
                status_text = "FALL DETECTED!"
                status_color = (0, 0, 255) # Rosso
            else:
                status_text = "NO FALL DETECTED"
                status_color = (0, 255, 0) # Verde
        else:
            status_text = "INSUFFICIENT DATA"
            status_color = (0, 255, 255) # Giallo

        # Visualizzazione stato in alto a sinistra
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        cv2.imshow('YOLO Video Fall Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ========================================== main ==========================================
if __name__ == "__main__":
    # Percorso del modello
    MODEL_PATH = 'runs/segment/body_parts12/weights/best.pt'
    
    # Percorsi dei video da voler testare
    VIDEO_PATH_1 = 'digital-image-processing/test-dataset/videos/video_caduta.avi'
    VIDEO_PATH_2 = 'digital-image-processing/test-dataset/videos/video_caduta_2.avi'
    VIDEO_PATH_3 = 'digital-image-processing/test-dataset/videos/video_caduta_3.avi'
    VIDEO_PATH_4 = 'digital-image-processing/test-dataset/videos/video_caduta.mp4'
    
    CONF_THRESHOLD = 0.4  # Soglia di confidenza
    
    run_video(MODEL_PATH, VIDEO_PATH_1, CONF_THRESHOLD)
    #run_video(MODEL_PATH, VIDEO_PATH_2, CONF_THRESHOLD)
    run_video(MODEL_PATH, VIDEO_PATH_3, CONF_THRESHOLD)
    #run_video(MODEL_PATH, VIDEO_PATH_4, CONF_THRESHOLD)