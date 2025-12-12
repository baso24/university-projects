import cv2
import time
from ultralytics import YOLO

def run_pose_estimation_filtered():
    model = YOLO('yolo11n-pose.pt') 
    cap = cv2.VideoCapture(0)
    
    KEYPOINT_NAMES = {
        0: "Naso", 1: "Occhio Sx", 2: "Occhio Dx", 3: "Orecchio Sx", 4: "Orecchio Dx",
        5: "Spall∏ Sx", 6: "Spalla Dx", 7: "Gomito Sx", 8: "Gomito Dx",
        9: "Polso Sx", 10: "Polso Dx", 11: "Anca Sx", 12: "Anca Dx",
        13: "Ginocchio Sx", 14: "Ginocchio Dx", 15: "Caviglia Sx", 16: "Caviglia Dx"
    }

    last_print_time = 0
    print_interval = 0.2
    
    KEYPOINT_CONF_THRESHOLD = 0.8 # Soglia di confidenza per i punti chiave (80%)
    PERSON_CONF_THRESHOLD = 0.8  # Soglia di confidenza per il rilevamento delle persone (70%)

    while True:
        success, frame = cap.read()  # legge un frame dalla webcam, success indica se la lettura è andata a buon fine, frame è l'immagine catturata
        if not success:
            break

        results = model(frame, verbose=False)  # esegue la stima della posa sul frame corrente
        annotated_frame = results[0].plot() # disegna i risultati sul frame

        current_time = time.time()
        
        if current_time - last_print_time > print_interval:
            print("-" * 30)
            
            # Verifica che ci siano punti chiave rilevati
            if results[0].keypoints is not None and results[0].keypoints.xy.numel() > 0:
                
                # Estraiamo coordinate (xy) e confidenza (conf)
                # Spostiamo tutto su CPU e convertiamo in numpy
                all_keypoints = results[0].keypoints.xy.cpu().numpy()
                all_confs = results[0].keypoints.conf.cpu().numpy() 
                
                for i, (person_kpts, person_confs) in enumerate(zip(all_keypoints, all_confs)):
                    # Controllo se la confidenza della persona è sopra la soglia
                    if results[0].boxes[i].conf >= PERSON_CONF_THRESHOLD:
                        print(f"PERSONA {i+1} (Solo punti visibili > {int(KEYPOINT_CONF_THRESHOLD*100)}%):")
                        
                        visible_points_count = 0
                        
                        for idx, ((x, y), conf) in enumerate(zip(person_kpts, person_confs)):
                            # Controllo se la confidenza dei punti è sopra la soglia e le coordinate non sono (0,0)
                            if conf >= KEYPOINT_CONF_THRESHOLD and (x != 0 or y != 0):
                                part_name = KEYPOINT_NAMES.get(idx, f"Punto {idx}")
                                print(f"  {part_name}: Conf={conf:.2f} | X={int(x)}, Y={int(y)}")
                                visible_points_count += 1
                        
                        if visible_points_count == 0:
                            print("  (Nessun punto del corpo sufficientemente visibile)")

            else:
                print("Nessuna persona rilevata.")
            
            last_print_time = current_time

        cv2.imshow("YOLO11 Pose - ", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pose_estimation_filtered()