import cv2
import time
import os
import numpy as np
from ultralytics import YOLO

def normalize_keypoints(keypoints, img_width, img_height):
    normalized = keypoints.copy()
    normalized[:, 0] /= img_width
    normalized[:, 1] /= img_height
    return normalized

def get_keypoints():
    model = YOLO('assets/yolo11n-pose.pt') 
    
    KEYPOINT_NAMES = {
        0: "Naso", 1: "Occhio Sx", 2: "Occhio Dx", 3: "Orecchio Sx", 4: "Orecchio Dx",
        5: "Spalla Sx", 6: "Spalla Dx", 7: "Gomito Sx", 8: "Gomito Dx",
        9: "Polso Sx", 10: "Polso Dx", 11: "Anca Sx", 12: "Anca Dx",
        13: "Ginocchio Sx", 14: "Ginocchio Dx", 15: "Caviglia Sx", 16: "Caviglia Dx"
    }
    
    KEYPOINT_CONF_THRESHOLD = 0.5 # Soglia di confidenza per i punti chiave (70%)
    PERSON_CONF_THRESHOLD = 0.5  # Soglia di confidenza per il rilevamento delle persone (50%)

    assets_path = 'assets/fall-01-cam0-rgb'
    image_files = sorted([f for f in os.listdir(assets_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    sequence_data = [] # Lista per raccogliere i dati sequenziali per la LSTM

    for image_file in image_files:
        frame = cv2.imread(os.path.join(assets_path, image_file))
        if frame is None:
            continue
        height, width = frame.shape[:2]

        results = model(frame, verbose=False)  # esegue la stima della posa sul frame corrente
        print(f"--- Risultati per {image_file} ---")
        
        # Inizializza vettore features: 17 keypoints * 3 (x_norm, y_norm, conf)
        # Se non viene trovata nessuna persona, questo vettore rimarrà pieno di zeri (Zero Padding)
        frame_features = np.zeros((17, 3))
        person_found = False

        if results[0].keypoints is not None and results[0].keypoints.xy.numel() > 0:
            all_keypoints = results[0].keypoints.xy.cpu().numpy()
            all_confs = results[0].keypoints.conf.cpu().numpy() 
            all_boxes_conf = results[0].boxes.conf.cpu().numpy()
            
            # Seleziona la persona con la confidenza del box più alta (assumiamo single-person tracking per la LSTM)
            best_person_idx = -1
            max_conf = -1
            
            for i, conf in enumerate(all_boxes_conf):
                if conf >= PERSON_CONF_THRESHOLD and conf > max_conf:
                    max_conf = conf
                    best_person_idx = i
            
            if best_person_idx != -1:
                person_found = True
                person_kpts = all_keypoints[best_person_idx]
                person_confs = all_confs[best_person_idx]
                
                norm_kpts = normalize_keypoints(person_kpts, width, height)
                print(f"PERSONA {best_person_idx+1} selezionata (Conf Box: {max_conf:.2f})")
                
                for kp_idx, (kp, conf) in enumerate(zip(person_kpts, person_confs)):
                    # Gestione punti chiave: se conf < soglia, lasciamo a 0 (o usiamo il valore raw se preferito)
                    if conf >= KEYPOINT_CONF_THRESHOLD:
                        n_x, n_y = norm_kpts[kp_idx]
                        frame_features[kp_idx] = [n_x, n_y, conf]
                        print(f"  {KEYPOINT_NAMES[kp_idx]}: ({kp[0]:.1f}, {kp[1]:.1f}) [Norm: {n_x:.4f}, {n_y:.4f}] Conf: {conf:.2f}")

        if not person_found:
            print("Nessuna persona valida rilevata. Frame inserito come zeri.")
        
        # Appiattiamo il vettore (es. 17*3 = 51 features) e aggiungiamo alla sequenza
        sequence_data.append(frame_features.flatten())
        print("\n")

    # Convertiamo in array numpy pronto per l'LSTM: (Time_Steps, Features)
    lstm_input = np.array(sequence_data)
    print(f"Generazione completata. Shape dati LSTM: {lstm_input.shape}")
    
    if len(lstm_input) > 0:
        print("\n--- Verifica Struttura Dati (Primo Frame) ---")
        print(f"Ogni riga (frame) ha {lstm_input.shape[1]} valori (17 keypoints * 3 features).")
        print("I valori sono appiattiti in sequenza: [x0, y0, conf0, x1, y1, conf1, ...]")
        
        # Reshape per mostrare che x e y ci sono ancora
        sample_reshaped = lstm_input[0].reshape(17, 3)
        print("\nEsempio primo frame riformattato (17 keypoints x [x, y, conf]):")
        print(sample_reshaped)
        
        print(lstm_input)

if __name__ == "__main__":
    get_keypoints()