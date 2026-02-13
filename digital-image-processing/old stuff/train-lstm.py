import cv2
import time
import os
from ultralytics import YOLO

def normalize_keypoints(keypoints, img_width, img_height):
    """
    Normalizza le coordinate dei keypoints tra 0 e 1 dividendo per le dimensioni dell'immagine.
    """
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

    for image_file in image_files:
        frame = cv2.imread(os.path.join(assets_path, image_file))
        if frame is None:
            continue
        height, width = frame.shape[:2]

        results = model(frame, verbose=False)  # esegue la stima della posa sul frame corrente
        print(f"--- Risultati per {image_file} ---")
        if results[0].keypoints is not None and results[0].keypoints.xy.numel() > 0:
            all_keypoints = results[0].keypoints.xy.cpu().numpy()
            all_confs = results[0].keypoints.conf.cpu().numpy() 
            
            person_found = False
            for i, (person_kpts, person_confs) in enumerate(zip(all_keypoints, all_confs)):
                if results[0].boxes[i].conf >= PERSON_CONF_THRESHOLD:
                    person_found = True
                    norm_kpts = normalize_keypoints(person_kpts, width, height)
                    print(f"PERSONA {i+1} (Solo punti visibili > {int(KEYPOINT_CONF_THRESHOLD*100)}%):")
                    for kp_idx, (kp, conf) in enumerate(zip(person_kpts, person_confs)):
                        if conf >= KEYPOINT_CONF_THRESHOLD:
                            n_x, n_y = norm_kpts[kp_idx]
                            print(f"  {KEYPOINT_NAMES[kp_idx]}: ({kp[0]:.1f}, {kp[1]:.1f}) [Norm: {n_x:.4f}, {n_y:.4f}] Conf: {conf:.2f}")
            if not person_found:
                print(f"Persone rilevate ma con confidenza inferiore a {PERSON_CONF_THRESHOLD}")
        else:
            print("Nessun punto chiave rilevato.")
        print("\n")

    


if __name__ == "__main__":
    get_keypoints()