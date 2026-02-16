import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

def get_centroids_vector(result):
    """
    Calcola i centroidi delle classi segmentate e restituisce un vettore appiattito.
    Classi: 0: testa, 1: braccia, 2: torso, 3: gambe, 4: piedi
    Output: lista [x0, y0, x1, y1, x2, y2, x3, y3, x4, y4]
    """
    class_ids = [0, 1, 2, 3, 4]
    
    # Struttura per accumulare i momenti per ogni classe
    class_moments = {k: {'m10': 0.0, 'm01': 0.0, 'm00': 0.0} for k in class_ids}
    
    if result.masks:
        masks_xy = result.masks.xy
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for i, poly in enumerate(masks_xy):
            cls_id = classes[i]
            
            if cls_id not in class_moments:
                continue
                
            # Converte poligono in punti per cv2.moments
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            M = cv2.moments(pts)
            
            # Accumula momenti (gestisce maschere frammentate)
            if M["m00"] != 0:
                class_moments[cls_id]['m10'] += M["m10"]
                class_moments[cls_id]['m01'] += M["m01"]
                class_moments[cls_id]['m00'] += M["m00"]

    # Calcolo coordinate centroidi
    centroids_vector = []
    centroids_map = {}

    for cid in class_ids:
        m = class_moments[cid]
        if m['m00'] != 0:
            cX = int(m['m10'] / m['m00'])
            cY = int(m['m01'] / m['m00'])
            centroids_map[cid] = (cX, cY)
            centroids_vector.extend([cX, cY])
        else:
            centroids_map[cid] = None
            # Se manca il centroide, inseriamo -1, -1 come valore "sentinella"
            centroids_vector.extend([-1, -1])

    return centroids_vector, centroids_map

def normalize_vector(vector, shape):
    h, w = shape
    norm_vector = []
    for j in range(0, len(vector), 2):
        # Se il valore è -1 (dato mancante), lo manteniamo tale anche dopo la normalizzazione
        if vector[j] == -1:
            norm_vector.extend([-1.0, -1.0])
        else:
            norm_vector.append(vector[j] / w if w > 0 else 0)
            norm_vector.append(vector[j+1] / h if h > 0 else 0)
    return norm_vector

def load_classifier_model(path):
    # Definizione identica a quella usata nel training
    model = nn.Sequential(
        nn.Linear(10, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )
    
    try:
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
    except Exception as e:
        print(f"Errore caricamento classificatore: {e}")
        return None


def run_test(yolo_model_path, classifier_model_path, test_images_dir):
    try:
        yolo_model = YOLO(yolo_model_path)
    except Exception as e:
        print(f"Errore caricamento YOLO: {e}")
        return

    classifier_model = load_classifier_model(classifier_model_path)
    if classifier_model is None:
        print("Impossibile caricare il classificatore.")
        return

    test_images_path = Path(test_images_dir)
    if not test_images_path.exists():
        print(f"Errore: La cartella immagini {test_images_path} non esiste.")
        return

    # Prende tutti i file
    image_files = sorted(list(test_images_path.glob('*'))) 
    # Estensioni
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

    if not image_files:
        print("Nessuna immagine trovata nella cartella di test.")
        return

    print(f"Trovate {len(image_files)} immagini")

    for img_path in image_files:

        # Applicazione YOLO
        results = yolo_model.predict(source=str(img_path), conf=0.25, verbose=False)
        result = results[0]

        # Calcolo centroidi
        vector_raw, centroids_map = get_centroids_vector(result)
        
        # Normalizzazione vettore di centroidi
        vector_norm = normalize_vector(vector_raw, result.orig_shape)

        # Applicazione classificatore
        input_tensor = torch.tensor(vector_norm, dtype=torch.float32).unsqueeze(0) # Batch size 1
        
        with torch.no_grad():
            output = classifier_model(input_tensor)
            prob = output.item()

        # Output del classificatore
        is_fall = prob > 0.5
        label_str = "FALL" if is_fall else "NO FALL"
        color_str = "red" if is_fall else "green"

        # Visualizzazione
        annotated_img = result.plot() # BGR
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        # Disegno i centroidi
        h, w = result.orig_shape
        for cid, coords in centroids_map.items():
            if coords:
                cv2.circle(annotated_img_rgb, coords, 8, (255, 255, 255), -1)
                cv2.circle(annotated_img_rgb, coords, 5, (0, 0, 0), -1)

        plt.figure(figsize=(10, 9))
        plt.imshow(annotated_img_rgb)
        plt.axis('off')
        
        # Titolo
        title_text = f"File: {img_path.name}\nPrediction: {label_str}\nConfidence: {prob:.4f}"
        plt.title(title_text, fontsize=14, color=color_str, fontweight='bold')

        # Coordinate vettore normalizzato visualizzate sotto l'immagine
        norm_vec_str = "[" + ", ".join([f"{x:.4f}" for x in vector_norm]) + "]"
        info_text = (
            f"Input Vector:\n{norm_vec_str}\n\n"
            "Classes Map: [0:Head, 1:Arms, 2:Torso, 3:Legs, 4:Feet]"
        )
        plt.figtext(0.5, 0.08, info_text, wrap=True, horizontalalignment='center', fontsize=10)

        plt.tight_layout(rect=[0, 0.2, 1, 0.85])
        plt.show()

if __name__ == "__main__":
    # Path modello
    YOLO_PATH = 'runs/segment/body_parts12/weights/best.pt'
    
    # Path classificatore
    CURRENT_DIR = Path(__file__).resolve().parent
    CLASSIFIER_PATH = CURRENT_DIR / 'classifier.pth'
    
    # Path delle immagini di test
    IMAGES_DIR = 'digital-image-processing/test-dataset/images'

    run_test(YOLO_PATH, CLASSIFIER_PATH, IMAGES_DIR)
