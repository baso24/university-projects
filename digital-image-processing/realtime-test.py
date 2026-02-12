import cv2
import numpy as np
from ultralytics import YOLO
import time

def run(model_path, conf_threshold):
    print(f"Caricamento modello: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Errore caricamento modello: {e}")
        return

    # Apre la webcam (indice 0 di solito è la webcam integrata)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Errore: Impossibile aprire la webcam.")
        return

    print("Avvio webcam... Premi 'q' per uscire.")

    # 0: testa, 1: braccia, 2: torso, 3: gambe, 4: piedi
    class_colors = {
        0: (0, 255, 255),   # Giallo (Testa)
        1: (0, 255, 0),     # Verde (Braccia)
        2: (255, 0, 0),     # Blu (Torso)
        3: (0, 165, 255),   # Arancione (Gambe)
        4: (128, 0, 128)    # Viola (Piedi)
    }

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Errore lettura frame.")
            break

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
        prev_frame_time = new_frame_time

        # Inferenza YOLO
        # verbose=False evita di stampare log per ogni frame
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        result = results[0]

        # Copia del frame per creare l'overlay delle maschere
        overlay = frame.copy()
        
        # Struttura per accumulare i momenti (come in test.py)
        class_moments = {k: {'m10': 0.0, 'm01': 0.0, 'm00': 0.0} for k in class_colors}
        
        if result.masks:
            masks_xy = result.masks.xy
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for i, poly in enumerate(masks_xy):
                cls_id = classes[i]
                color = class_colors.get(cls_id, (200, 200, 200))
                
                # Disegna maschera piena sull'overlay
                pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], color)
                
                # Calcola momenti per i centroidi
                M = cv2.moments(pts)
                if M["m00"] != 0:
                    if cls_id in class_moments:
                        class_moments[cls_id]['m10'] += M["m10"]
                        class_moments[cls_id]['m01'] += M["m01"]
                        class_moments[cls_id]['m00'] += M["m00"]

        # Calcolo centroidi unici per classe (escludendo braccia come in test.py)
        final_centroids = {}
        for cid, m in class_moments.items():
            if cid == 1: # Skip braccia per il calcolo dello scheletro
                continue
            if m['m00'] != 0:
                cX = int(m['m10'] / m['m00'])
                cY = int(m['m01'] / m['m00'])
                final_centroids[cid] = (cX, cY)

        # Blending trasparenza (sovrappone le maschere colorate all'immagine originale)
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

        # Logica Fall Detection (identica a test.py)
        status_text = "WAITING..."
        status_color = (200, 200, 200) # Grigio

        # Verifica se sono presenti le componenti necessarie: Testa(0), Torso(2), Gambe(3)
        if {0, 2, 3}.issubset(final_centroids.keys()):
            head_x, head_y = final_centroids[0]
            torso_x, torso_y = final_centroids[2]
            legs_x, legs_y = final_centroids[3]

            # Se la distanza orizzontale è maggiore di quella verticale per i segmenti chiave
            is_fallen = (abs(head_y - torso_y) < abs(head_x - torso_x)) and \
                        (abs(head_y - legs_y) < abs(head_x - legs_x))
            
            if is_fallen:
                status_text = "FALL DETECTED!"
                status_color = (0, 0, 255) # Rosso
            else:
                status_text = "NO FALL DETECTED"
                status_color = (0, 255, 0) # Verde
        else:
            status_text = "PARTIAL DETECTION"
            status_color = (0, 165, 255) # Arancione

        # Visualizzazione fps
        cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 180, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.imshow('YOLO Real-time Fall Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Percorso del modello
    MODEL_PATH = 'runs/segment/body_parts8/weights/best.pt'
    
    CONF_THRESHOLD = 0.4  # Soglia di confidenza
    
    run(MODEL_PATH, CONF_THRESHOLD)