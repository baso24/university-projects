import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pathlib import Path
from ultralytics import YOLO

def test_image_from_validationSet(model_path, images_dir, conf_threshold):
    # Carica modello
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Errore caricamento modello: {e}")
        return

    img_path_obj = Path(images_dir)
    if not img_path_obj.exists():
        print(f"Errore: La cartella {images_dir} non esiste.")
        return

    # Cerca file immagini
    image_files = list(img_path_obj.glob('*.jpg'))
    if not image_files:
        print("Nessuna immagine trovata.")
        return

    random_image = random.choice(image_files)

    # Applico il modello
    results = model.predict(source=str(random_image), conf=conf_threshold, save=False, verbose=False)
    
    # Visualizzazione
    show_result(results[0], f"Test on: {random_image.name}")

def test_specific_image(model_path, image_name, conf_threshold):
    
    # Verifica esistenza file
    target_path = Path(image_name)
    if not target_path.exists():
        print(f"ERRORE: Il file '{image_name}' non è stato trovato nella directory corrente!")
        print(f"Cercato in: {target_path.absolute()}")
        return

    # Carica modello
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Errore caricamento modello: {e}")
        return

    # Applico il modello
    results = model.predict(source=str(target_path), conf=conf_threshold, save=False, verbose=False)

    # Visualizzazione
    show_result(results[0], f"Test on: {image_name}")

    # Visualizzazione analisi segmentazione
    show_segmentation_analysis(results[0], f"Segmentation analysis: {image_name}")

def show_result(result, title):
    # Carica l'immagine originale
    orig_img = cv2.imread(result.path)
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # Immagine originale
    orig_img = result.orig_img
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # Plot annotato da YOLO (in BGR)
    annotated_frame = result.plot()
    # BGR -> RGB per Matplotlib
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Creazione della figura con due subplot affiancati
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Mostra l'immagine originale
    axes[0].imshow(orig_img_rgb)
    axes[0].set_title("Original photo")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_frame_on(False)
    axes[0].set_xlabel(result.path, fontsize=10)

    # Mostra il risultato dell'annotazione
    axes[1].imshow(annotated_frame_rgb)
    axes[1].set_title("Result")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_frame_on(False)
    axes[1].set_xlabel(title, fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
def show_segmentation_analysis(result, title):
    if not result.masks:
        return

    # Preparazione Immagine Base (RGB)
    orig_img = result.orig_img.copy()
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    overlay = img_rgb.copy()

    # Definizione Colori (RGB) e Nomi
    # 0: testa, 1: braccia, 2: torso, 3: gambe, 4: piedi
    class_colors = {
        0: (255, 255, 0),   # Giallo
        1: (0, 255, 0),     # Verde
        2: (0, 0, 255),     # Blu
        3: (255, 165, 0),   # Arancione
        4: (128, 0, 128)    # Viola
    }
    
    class_moments = {k: {'m10': 0.0, 'm01': 0.0, 'm00': 0.0} for k in class_colors}
    masks_xy = result.masks.xy
    classes = result.boxes.cls.cpu().numpy().astype(int)

    # Disegno maschere e calcolo centroidi
    for i, poly in enumerate(masks_xy):
        cls_id = classes[i]
        color = class_colors.get(cls_id, (200, 200, 200))
        
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)
        
        M = cv2.moments(pts)
        if M["m00"] != 0:
            if cls_id in class_moments:
                class_moments[cls_id]['m10'] += M["m10"]
                class_moments[cls_id]['m01'] += M["m01"]
                class_moments[cls_id]['m00'] += M["m00"]

    # Calcolo centroidi unici per classe
    final_centroids = {}
    for cid, m in class_moments.items():
        if cid == 1: # Skip braccia
            continue
        if m['m00'] != 0:
            cX = int(m['m10'] / m['m00'])
            cY = int(m['m01'] / m['m00'])
            final_centroids[cid] = (cX, cY)

    img_final = cv2.addWeighted(overlay, 0.5, img_rgb, 0.5, 0)

    # Disegno skeleton
    # Ci dobbiamo assicurare che ci sia una sola persona nella foto per avere un risultato coerente.
    skeleton_links = [
        (0, 2), # Testa - Torso
        (2, 3), # Torso - Gambe
        (3, 4)  # Gambe - Piedi
    ]

    for cls_a, cls_b in skeleton_links:
        if cls_a in final_centroids and cls_b in final_centroids:
            pt_a = final_centroids[cls_a]
            pt_b = final_centroids[cls_b]
            cv2.line(img_final, pt_a, pt_b, (255, 255, 255), 2)

    # Disegno i centroidi
    for pt in final_centroids.values():
        cv2.circle(img_final, pt, 6, (0, 0, 0), -1)     # Bordo nero
        cv2.circle(img_final, pt, 4, (255, 0, 0), -1)   # Centro rosso

    # Plot con legenda
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [4, 1]})
    
    ax1.imshow(img_final)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_frame_on(False)
    # Il titolo con il nome del file va sotto l'immagine
    ax1.set_xlabel(title.replace("Segmentation analysis: ", ""), fontsize=12)

    legend_elements = []
    sorted_ids = sorted([k for k in class_colors.keys() if k in classes])
    
    for k in sorted_ids:
        c = class_colors[k]
        name = result.names[k].capitalize()
        if k in final_centroids:
            cx, cy = final_centroids[k]
            label_text = f"{name}\nX={cx}, Y={cy}"
        else:
            # Se una parte chiave (testa, torso, gambe) non è rilevata, segnalalo nella legenda
            if k in {0, 2, 3}:
                label_text = f"{name}\n(non rilevato)"
            else:
                label_text = name
        legend_elements.append(Patch(facecolor=np.array(c)/255, edgecolor='black', label=label_text))
    
    ax2.legend(handles=legend_elements, loc='center', title="Legenda e Coordinate", fontsize=12)
    ax2.axis('off')

    is_fall, pair_found = fall_detection(final_centroids)

    if pair_found:
        if is_fall:
            plt.suptitle("POSSIBLE FALL DETECTED!", color='red', fontsize=16)
        else:
            plt.suptitle("UNDETECTED FALL", color='green', fontsize=16)
    else:
        plt.suptitle("Not enough components for fall analysis", color='orange', fontsize=16)

    # Aggiusto il layout per far spazio a suptitle e xlabel
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def is_fallen_check(p1, p2):
    # Se la distanza verticale è minore di quella orizzontale -> Fall detection
    return abs(p1[1] - p2[1]) < abs(p1[0] - p2[0])

def fall_detection(centroids):
    fall_detected = False
    pair_found = False

    # Case 1: Testa (0) e Torso (2)
    if 0 in centroids and 2 in centroids:
        pair_found = True
        if is_fallen_check(centroids[0], centroids[2]):
            fall_detected = True

    # Case 2: Testa (0) e Gambe (3)
    if not fall_detected and 0 in centroids and 3 in centroids:
        pair_found = True
        if is_fallen_check(centroids[0], centroids[3]):
            fall_detected = True

    # Case 3: Torso (2) e Gambe (3)
    if not fall_detected and 2 in centroids and 3 in centroids:
        pair_found = True
        if is_fallen_check(centroids[2], centroids[3]):
            fall_detected = True

    return fall_detected, pair_found

# ========================================== main ==========================================
if __name__ == "__main__":
    
    # Path al modello che si desidera utilizzare per il test
    MODEL_PATH = 'runs/segment/body_parts12/weights/best.pt' 
    
    # Path per il test random, immagine presa dal dataset di validazione
    VAL_IMAGES_PATH = 'assets/cihp-DatasetNinja/processed/images/val'
    
    # Path per il test su immagini specifiche
    TEST_IMAGE_PATH = 'digital-image-processing/test-dataset/images/caduta.jpg'
    TEST_IMAGE_PATH_3 = 'digital-image-processing/test-dataset/images/inpiedi.jpg'

    # Test random su immagine casuale del dataset di validazione
    test_image_from_validationSet(MODEL_PATH, VAL_IMAGES_PATH, conf_threshold=0.25)

    # Test specifico su "caduta.jpg"
    test_specific_image(MODEL_PATH, TEST_IMAGE_PATH, conf_threshold=0.25)
    
    # Test specifico su "inpiedi.png"
    test_specific_image(MODEL_PATH, TEST_IMAGE_PATH_3, conf_threshold=0.25)