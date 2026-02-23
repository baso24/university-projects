import cv2
import numpy as np
import time
from ultralytics import YOLO

def is_fallen(p1, p2):
    # Se la distanza verticale è minore di quella orizzontale -> Fall detection
    return abs(p1[1] - p2[1]) < abs(p1[0] - p2[0])

class BodyPartsFallDetector:
    def __init__(self, model_path='yolov8n-seg.pt'):
        self.model = YOLO(model_path)
        self.class_map = {0: 'head', 1: 'torso', 2: 'arms', 3: 'legs', 4: 'feet'}
        
        # Inizializza il Background Subtractor
        # history=500: impara lo sfondo su 500 frame
        # varThreshold=25: soglia sensibilità
        # detectShadows=False: Disabilito per performance (non calcola ombre)
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
        
        # Buffer per Optical Flow (per la logica di caduta)
        self.prev_gray = None

    def get_centroid(self, mask_poly):
        contour = mask_poly.astype(np.int32)
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return np.array([cx, cy])
        return None

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        # Ottieni FPS originali del video
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0: video_fps = 30 # Fallback
        wait_ms = int(1000 / video_fps)

        cv2.namedWindow('YOLO Video Fall Detection', cv2.WINDOW_NORMAL) # Finestra ridimensionabile
        
        prev_frame_time = 0
        new_frame_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Calcolo FPS elaborazione
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time
            proc_fps_text = f"Proc FPS: {int(fps)}"

            # 1. Calcolo Maschera di Movimento (MOG2) - Ottimizzazione: Resize
            scale_factor = 0.5
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            fg_mask = self.backSub.apply(small_frame)
            
            # Pulizia morfologica
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # Kernel ridotto per frame piccolo
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)
            
            # Rimuovi le ombre (Non necessario con detectShadows=False)
            # _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            
            # 2. Trova la ROI del movimento (Bounding Box dinamico)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            min_x, min_y = frame.shape[1], frame.shape[0]
            max_x, max_y = 0, 0
            motion_detected = False
            
            for cnt in contours:
                if cv2.contourArea(cnt) > 150: # Filtro rumore (adattato alla scala ridotta)
                    motion_detected = True
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Riscala coordinate al frame originale
                    x = int(x / scale_factor)
                    y = int(y / scale_factor)
                    w = int(w / scale_factor)
                    h = int(h / scale_factor)

                    # Aggiorna min/max globali
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x + w)
                    max_y = max(max_y, y + h)
            
            # Se non c'è movimento significativo, mostriamo il frame così com'è
            if not motion_detected:
                cv2.imshow('YOLO Video Fall Detection', frame)
                if cv2.waitKey(wait_ms) & 0xFF == ord('q'): break
                continue
                
            # Aggiungi Padding alla ROI
            padding = 50
            h_img, w_img = frame.shape[:2]
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(w_img, max_x + padding)
            max_y = min(h_img, max_y + padding)
            
            # Ritaglio (Zoom)
            crop_frame = frame[min_y:max_y, min_x:max_x]
            
            if crop_frame.size == 0: continue 

            # Inferenza YOLO su CROP
            # retina_masks=False velocizza notevolmente l'inferenza
            results = self.model.predict(crop_frame, verbose=False, retina_masks=False, conf=0.5)
            
            # Disegno bounding box del movimento sul frame originale
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

            # 0: testa, 1: braccia, 2: torso, 3: gambe, 4: piedi
            class_colors = {
                0: (0, 255, 255),   # Giallo (Testa)
                1: (0, 255, 0),     # Verde (Braccia)
                2: (255, 0, 0),     # Blu (Torso)
                3: (0, 165, 255),   # Arancione (Gambe)
                4: (128, 0, 128)    # Viola (Piedi)
            }
            
            class_moments = {k: {'m10': 0.0, 'm01': 0.0, 'm00': 0.0} for k in class_colors}
            
            # Overlay per disegnare le maschere
            overlay = frame.copy()

            if results[0].masks is not None:
                masks_xy = results[0].masks.xy
                classes = results[0].boxes.cls.cpu().numpy()
                
                for mask_poly, cls_id in zip(masks_xy, classes):
                    if len(mask_poly) == 0: continue
                    
                    cls_id = int(cls_id)
                    color = class_colors.get(cls_id, (200, 200, 200))
                    
                    # Converti coordinate: Da CROP a FRAME ORIGINALE
                    mask_poly_global = mask_poly + np.array([min_x, min_y])
                    mask_pts = mask_poly_global.astype(np.int32)
                    
                    # Disegna maschera piena sull'overlay
                    cv2.fillPoly(overlay, [mask_pts], color)
                    
                    # Calcola momenti sulla maschera globale
                    M = cv2.moments(mask_pts)
                    if M["m00"] != 0:
                        if cls_id in class_moments:
                            class_moments[cls_id]['m10'] += M["m10"]
                            class_moments[cls_id]['m01'] += M["m01"]
                            class_moments[cls_id]['m00'] += M["m00"]

            # Blending trasparenza
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
            
            # Calcolo centroidi unici
            final_centroids = {}
            for cid, m in class_moments.items():
                if cid == 1: continue 
                if m['m00'] != 0:
                    cX = int(m['m10'] / m['m00'])
                    cY = int(m['m01'] / m['m00'])
                    final_centroids[cid] = (cX, cY)

            # Skeleton
            skeleton_links = [(0, 2), (2, 3), (3, 4)]
            for cls_a, cls_b in skeleton_links:
                if cls_a in final_centroids and cls_b in final_centroids:
                    cv2.line(frame, final_centroids[cls_a], final_centroids[cls_b], (255, 255, 255), 2)

            # Disegno centroidi
            for pt in final_centroids.values():
                cv2.circle(frame, pt, 6, (0, 0, 0), -1)
                cv2.circle(frame, pt, 4, (255, 0, 0), -1)

            # Logica caduta
            fall_detected = False
            pair_found = False
            
            check_pairs = [(0, 2), (0, 3), (2, 3)]
            for (p1, p2) in check_pairs:
                if not fall_detected and p1 in final_centroids and p2 in final_centroids:
                    pair_found = True
                    if is_fallen(final_centroids[p1], final_centroids[p2]):
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
            
            # Scritta resized (più piccola)
            cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # FPS Text
            cv2.putText(frame, proc_fps_text, (frame.shape[1] - 130, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) 

            cv2.imshow('YOLO Video Fall Detection', frame)

            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
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
    
    detector = BodyPartsFallDetector(MODEL_PATH)
    detector.process_video(VIDEO_PATH_1)
    detector.process_video(VIDEO_PATH_2)
    detector.process_video(VIDEO_PATH_3)