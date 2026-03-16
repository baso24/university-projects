import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

import numpy as np
from collections import deque

# Classe per rilevare anomalie posturali basate sui versori normalizzati tra i centroidi delle perti del corpo rilevate
# Si prendono in considerazione 3 versori: Testa->Busto, Testa->Gambe, Busto->Gambe
# Durante la fase di calibrazione si costruisce media e covarianza di questi vettori in condizioni di normalità (postura eretta)
# Finita la calibrazione si misura la distanza di Mahalanobis per il numero di coppie di centroidi che vengono rilevati nel frame
# Se la distanza supera una certa soglia si marca il frame come sospetto
# Se in una finestra temporale di frames consecuitivi la maggioranza dei frame è sospetta allora si segnala una possibile caduta
class PosturalAnomalyDetector:
    # baseline_frames: numero di frame richiesti per costruire la baseline inizial, devono essere presenti tutte e 6 le componenti dei versori
    # anomaly_thresh: soglia di distanza di Mahalanobis normalizzata per considerare un frame come anomalo (2.0 uquivale a 2 deviazioni standard)
    # time_window: numero di frame consecutivi da considerare per la finestra di rilevamento caduta
    def __init__(self, calibration_frames, anomaly_thresh, time_window):
        self.calibration_frames = calibration_frames
        self.anomaly_thresh = anomaly_thresh
        # buffer circolare per tenere traccia degli ultimi N frame per rilevare sequenze di anomalie
        self.anomaly_buffer = deque(maxlen=time_window)
        
        self.calibration_features = [] # Lista per accumulare i vettori di feature durante la fase di calibrazione
        self.mean = None # Vettore medio dei feature calcolato durante la calibrazione
        self.cov = None # Matrice di covarianza dei feature calcolata durante la calibrazione
        self.is_calibrated = False # Flag per indicare se la calibrazione è stata completata

    def extract_features(self, centroids):
        features = []
        
        # Versore Testa (0) -> Busto (2)
        if 0 in centroids and 2 in centroids:
            distance = np.array(centroids[2]) - np.array(centroids[0])
            norm = np.linalg.norm(distance)
            verser = distance / norm
            features.extend(verser.tolist())
        else:
            features.extend([np.nan, np.nan]) # Uso NaN per i dati mancanti
        
        # Versore Testa (0) -> Gambe (3)
        if 0 in centroids and 3 in centroids:
            distance = np.array(centroids[3]) - np.array(centroids[0])
            norm = np.linalg.norm(distance)
            versor = distance / norm
            features.extend(versor.tolist())
        else:
            features.extend([np.nan, np.nan])

        # Versore Busto (2) -> Gambe (3)
        if 2 in centroids and 3 in centroids:
            distance = np.array(centroids[3]) - np.array(centroids[2])
            norm = np.linalg.norm(distance)
            versor = distance / norm
            features.extend(versor.tolist())
        else:
            features.extend([np.nan, np.nan])
            
        return np.array(features)

    def process_frame(self, centroids):
        feat = self.extract_features(centroids)
        
        # Tolgo i NaN per contare quante coppie di centroidi validi abbiamo (ogni coppia contribuisce con 2 feature, x e y)
        valid_mask = ~np.isnan(feat)
        # Conta quante coppie di centroidi validi abbiamo
        valid_count = np.sum(valid_mask)
        
        # Se non c'è nessuna coppia di detection utile, saltiamo il frame
        if valid_count == 0:
            self.anomaly_buffer.append(False) # Consideriamo il frame come "non anomalo" per non falsare la finestra temporale, ma in realtà è un dato mancante
            status = "DATI INSUFFICIENTI"
            return False, status, 0.0

        # Fase di calibrazione
        # Esigiamo frame "perfetti" (tutte e 6 le feature) solo per la calibrazione.
        if not self.is_calibrated:
            if valid_count == 6:
                self.calibration_features.append(feat)
                
                if len(self.calibration_features) >= self.calibration_frames:
                    X = np.array(self.calibration_features)
                    self.mean = np.mean(X, axis=0)
                    self.cov = np.cov(X, rowvar=False)
                    # self.cov += np.eye(self.cov.shape[0]) * 1e-4 # Regolarizzazione
                    
                    self.is_calibrated = True
                    
                    np.set_printoptions(precision=4, suppress=True, linewidth=120)
                    print("\n" + "="*50)
                    print(">>> CALIBRAZIONE COMPLETATA <<<")
                    print(f"Media:\n{self.mean}\nCovarianza:\n{self.cov}")
                    print("="*50 + "\n")
                    
                return False, f"CALIBRAZIONE ({len(self.calibration_features)}/{self.calibration_frames})", 0.0
            else:
                return False, "ATTESA FRAME COMPLETO PER CALIBRAZIONE", 0.0

        # Fase di inferenza (la calibrazione è stata completata, possiamo valutare l'anomalia anche su frame con feature parziali)
        valid_indices = np.where(valid_mask)[0]
        
        # Estraiamo le sotto-strutture basandoci SOLO sulle feature visibili
        feat_sub = feat[valid_indices]
        mean_sub = self.mean[valid_indices]
        
        # np.ix_ permette di estrarre la sottomatrice incrociando righe e colonne valide
        # In pratica in base ai versori che abbiamo ottenuto dalle classi che stiamo segmenentando nel frame corrente
        # Costuiamo una matrice di covarianza "ridotta" che considera solo queste classi attive
        cov_sub = self.cov[np.ix_(valid_indices, valid_indices)]
        
        # Inversa della matrice di covarianza ridotta, necessaria per il calcolo della distanza di Mahalanobis
        cov_sub_inv = np.linalg.inv(cov_sub)

        # Calcolo della distanza di Mahalanobis solo sulle feature valide
        mean_diff = feat_sub - mean_sub
        mahalanobis_distance_squared = np.dot(np.dot(mean_diff, cov_sub_inv), mean_diff.T)
        
        # Normalizziamo la distanza per il numero di gradi di libertà attivi
        mahalanobis_distance_normalized = np.sqrt(mahalanobis_distance_squared / valid_count)

        # Un frame è considerato anomalo se la distanza di Mahalanobis normalizzata supera la soglia definita
        is_anomalous = mahalanobis_distance_normalized > self.anomaly_thresh
        self.anomaly_buffer.append(is_anomalous)

        # Contiamo quante anomalie ci sono nella finestra temporale e se superano una certa percentuale (es. 70%) allora segnaliamo una possibile caduta
        anomalies_in_window = sum(self.anomaly_buffer)
        fall_detected = anomalies_in_window >= (self.anomaly_buffer.maxlen * 0.7)

        if fall_detected:
            status = "CADUTA RILEVATA!"
        elif is_anomalous:
            status = f"POSIZIONE SOSPETTA"
        else:
            status = f"POSIZIONE NORMALE"

        return fall_detected, status, mahalanobis_distance_normalized

# Logica di fal detection molto semplice usata in precedenza
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

    print("Avvio elaborazione video. Premi 'q' per uscire anticipatamente.")

    # Inizializzazione Background Subtractor (MOG2) per isolare i pixel in movimento
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    # Kernel per le operazioni morfologiche (pulizia della maschera)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Variabili per il calcolo FPS
    prev_time = time.time()
    fps_smooth = 0.0

    # Cache dell'ultima Bounding Box di movimento valida
    last_bbox = None

    class_colors = {
        0: (0, 255, 255),   # Giallo (Testa)
        1: (0, 255, 0),     # Verde (Braccia)
        2: (255, 0, 0),     # Blu (Torso)
        3: (0, 165, 255),   # Arancione (Gambe)
        4: (128, 0, 128)    # Viola (Piedi)
    }

    cv2.namedWindow('YOLO Video Fall Detection', cv2.WINDOW_NORMAL)

    # 100 frame di calibrazione a ~30fps sono circa 3 secondi di "postura normale"
    anomaly_detector = PosturalAnomalyDetector(baseline_frames=100, anomaly_thresh=2, time_window=15)

    while True:
        ret, frame = cap.read()
        if not ret:
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

        # --- VECCHIA LOGICA FALL DETECTION ---
        """"
        fall_detected = False
        pair_found = False

        if 0 in final_centroids and 2 in final_centroids:
            pair_found = True
            if is_fallen(final_centroids[0], final_centroids[2]): fall_detected = True

        if not fall_detected and 0 in final_centroids and 3 in final_centroids:
            pair_found = True
            if is_fallen(final_centroids[0], final_centroids[3]): fall_detected = True

        if not fall_detected and 2 in final_centroids and 3 in final_centroids:
            pair_found = True
            if is_fallen(final_centroids[2], final_centroids[3]): fall_detected = True

        if pair_found:
            if fall_detected:
                status_text = "FALL DETECTED!"
                status_color = (0, 0, 255)
            else:
                status_text = "NO FALL DETECTED"
                status_color = (0, 255, 0)
        else:
            status_text = "INSUFFICIENT DATA"
            status_color = (0, 255, 255)

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        """""

        # --- LOGICA FALL DETECTION (ANOMALY DETECTION) ---
        fall_detected, status_text, mahalanobis_distance_score = anomaly_detector.process_frame(final_centroids)

        if fall_detected:
            status_color = (0, 0, 255) # Rosso
        elif "SOSPETTA" in status_text:
            status_color = (0, 165, 255) # Arancione
        elif "CALIBRAZIONE" in status_text or "INSUFFICIENTI" in status_text:
            status_color = (0, 255, 255) # Giallo
        else:
            status_color = (0, 255, 0) # Verde

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Stampa del valore della distanza di Mahalanobis normalizzata dopo che è stata fatta la calibrazione
        if anomaly_detector.is_calibrated:
            cv2.putText(frame, f"Anomaly Score: {mahalanobis_distance_score:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- CALCOLO FPS ---
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

        cv2.imshow('YOLO Video Fall Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    MODEL_PATH = 'runs/segment/body_parts12/weights/best.pt'
    VIDEO_PATH_1 = 'digital-image-processing/test-dataset/videos/video_caduta_1.avi'

    CONF_THRESHOLD = 0.2
    run_video(MODEL_PATH, VIDEO_PATH_1, CONF_THRESHOLD)