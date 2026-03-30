import cv2
import numpy as np
from ultralytics import YOLO
import time

def is_fallen(p1, p2):
    # Se la distanza verticale è minore di quella orizzontale -> Caduta
    return abs(p1[1] - p2[1]) < abs(p1[0] - p2[0])

def run(model_path, conf_threshold):
    print(f"Caricamento modello: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Errore caricamento modello: {e}")
        return

    # Apre la webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Errore: Impossibile aprire la webcam.")
        return

    # Inizializzazione Background Subtractor (MOG2) per isolare i pixel in movimento
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    # Kernel per le operazioni morfologiche (pulizia della maschera)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Variabili per il calcolo FPS
    prev_time = time.time()
    fps_smooth = 0.0

    # Cache dell'ultima Bounding Box di movimento valida
    last_bbox = None

    print("Avvio webcam. Premi 'q' per uscire.")

    # 0: testa, 1: braccia, 2: torso, 3: gambe, 4: piedi
    class_colors = {
        0: (0, 255, 255),   # Giallo (Testa)
        1: (0, 255, 0),     # Verde (Braccia)
        2: (255, 0, 0),     # Blu (Torso)
        3: (0, 165, 255),   # Arancione (Gambe)
        4: (128, 0, 128)    # Viola (Piedi)
    }

    cv2.namedWindow('YOLO Real Time Fall Detection', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Errore lettura frame.")
            break

         # --- 1. MOTION DETECTION E CALCOLO ROI ---
        fgMask = backSub.apply(frame)
        
        # Pulizia morfologica per rimuovere rumore (es. sfarfallio della luce)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)

        # Trova i contorni degli oggetti in movimento
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_x, min_y = frame.shape[1], frame.shape[0]
        max_x, max_y = 0, 0
        motion_detected = False

        # Calcola un'unica Bounding Box che racchiuda tutto il movimento significativo
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Soglia area per scartare piccoli artefatti
                x, y, w, h = cv2.boundingRect(contour)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)
                motion_detected = True

        if motion_detected:
            # Aggiungiamo un padding dinamico per non tagliare parti del corpo sui bordi
            pad = 50
            min_x = max(0, min_x - pad)
            min_y = max(0, min_y - pad)
            max_x = min(frame.shape[1], max_x + pad)
            max_y = min(frame.shape[0], max_y + pad)
            last_bbox = (min_x, min_y, max_x, max_y)
        elif last_bbox is None:
            # Fallback iniziale se non c'è ancora stato alcun movimento
            last_bbox = (0, 0, frame.shape[1], frame.shape[0])

        x1, y1, x2, y2 = last_bbox
        
        # Estrazione della Region of Interest (ROI)
        roi = frame[y1:y2, x1:x2]

        # Disegno la BBox del tracking movimento per debug visuale
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, "Motion ROI", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- 2. INFERENZA YOLO (SOLO SULLA ROI) ---
        # N.B. YOLO riceve solo la porzione ritagliata, molto più piccola e priva di background
        results = model.predict(roi, conf=conf_threshold, verbose=False)
        result = results[0]

        overlay = frame.copy()
        class_moments = {k: {'m10': 0.0, 'm01': 0.0, 'm00': 0.0} for k in class_colors}
        
        if result.masks:
            masks_xy = result.masks.xy
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for i, poly in enumerate(masks_xy):
                cls_id = classes[i]
                color = class_colors.get(cls_id, (200, 200, 200))
                
                if len(poly) > 0:
                    # TRASLAZIONE DEI PUNTI: Mappa le coordinate YOLO (relative alla ROI) 
                    # nel sistema di riferimento del frame originale
                    poly_shifted = poly + np.array([x1, y1])
                    
                    pts = np.array(poly_shifted, np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts], color)
                    
                    M = cv2.moments(pts)
                    if M["m00"] != 0:
                        if cls_id in class_moments:
                            class_moments[cls_id]['m10'] += M["m10"]
                            class_moments[cls_id]['m01'] += M["m01"]
                            class_moments[cls_id]['m00'] += M["m00"]

        final_centroids = {}
        for cid, m in class_moments.items():
            if cid == 1:
                continue
            if m['m00'] != 0:
                cX = int(m['m10'] / m['m00'])
                cY = int(m['m01'] / m['m00'])
                final_centroids[cid] = (cX, cY)

        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        skeleton_links = [(0, 2), (2, 3), (3, 4)]
        for cls_a, cls_b in skeleton_links:
            if cls_a in final_centroids and cls_b in final_centroids:
                cv2.line(frame, final_centroids[cls_a], final_centroids[cls_b], (255, 255, 255), 2)

        for pt in final_centroids.values():
            cv2.circle(frame, pt, 6, (0, 0, 0), -1)    
            cv2.circle(frame, pt, 4, (255, 0, 0), -1)  

        # --- LOGICA FALL DETECTION ---
        status_text = "WAITING..."
        status_color = (200, 200, 200) # Grigio

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
            status_color = (0, 255, 255)

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

         # --- 3. CALCOLO E STAMPA FPS ---
        curr_time = time.time()
        # Tempo trascorso in secondi per compiere un iterazione intera
        time_diff = curr_time - prev_time 
        fps = 1.0 / time_diff if time_diff > 0 else 0.0
        prev_time = curr_time

        # Calcolo Media Mobile Esponenziale (EMA) per evitare fluttuazioni eccessive a schermo
        if fps_smooth == 0.0:
            fps_smooth = fps
        else:
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps 

        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow('YOLO Real Time Fall Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ========================================== main ==========================================
if __name__ == "__main__":
    # Percorso del modello
    MODEL_PATH = 'runs/segment/body_parts11/weights/best.pt'
    
    CONF_THRESHOLD = 0.4  # Soglia di confidenza
    
    run(MODEL_PATH, CONF_THRESHOLD)