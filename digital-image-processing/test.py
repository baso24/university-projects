import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pathlib import Path
from ultralytics import YOLO

def test_random_image(model_path, images_dir, conf_threshold):
    """
    Pesca un'immagine casuale dal dataset di validazione e mostra il risultato.
    """
    print(f"\n--- TEST RANDOM DAL DATASET ---")
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
    print(f"Processando random: {random_image.name}")

    # Inferenza
    results = model.predict(source=str(random_image), conf=conf_threshold, save=False, verbose=False)
    
    # Visualizzazione
    show_result(results[0], f"Random Val: {random_image.name}")

def test_specific_image(model_path, image_name, conf_threshold):
    """
    Testa il modello su un file specifico nella directory corrente.
    """
    print(f"\n--- TEST SPECIFICO SU '{image_name}' ---")
    
    # 1. Verifica esistenza file
    target_path = Path(image_name)
    if not target_path.exists():
        print(f"ERRORE: Il file '{image_name}' non è stato trovato nella directory corrente!")
        print(f"Cercato in: {target_path.absolute()}")
        return

    # 2. Carica Modello
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Errore caricamento modello: {e}")
        return

    # 3. Inferenza
    results = model.predict(source=str(target_path), conf=conf_threshold, save=False, verbose=False)

    # 4. Visualizzazione
    if not results:
        print("Nessun risultato generato.")
        return

    show_result(results[0], f"Test Specifico: {image_name}")
    show_segmentation_analysis(results[0], f"Analisi Segmentazione: {image_name}")

def show_result(result, title):
    # Plot annotato da YOLO (in BGR)
    annotated_frame = result.plot()
    
    # BGR -> RGB per Matplotlib
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))

    plt.imshow(annotated_frame_rgb)
    plt.axis('off')
    plt.title(title)
    plt.show()

def show_segmentation_analysis(result, title):
    if not result.masks:
        return

    # 1. Preparazione Immagine Base (RGB)
    orig_img = result.orig_img.copy()
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    overlay = img_rgb.copy()

    # 2. Definizione Colori (RGB) e Nomi
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

    # 3. Disegno Maschere e Calcolo Centroidi
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

    # Calcolo centroidi unici per classe (media pesata)
    final_centroids = {}
    for cid, m in class_moments.items():
        if cid == 1: # Skip braccia (nessuna distinzione dx/sx)
            continue
        if m['m00'] != 0:
            cX = int(m['m10'] / m['m00'])
            cY = int(m['m01'] / m['m00'])
            final_centroids[cid] = (cX, cY)

    # Blending trasparenza
    img_final = cv2.addWeighted(overlay, 0.5, img_rgb, 0.5, 0)

    # 4. Disegno Skeleton Semplificato (Singolo Soggetto)
    # Connessioni: Testa-Torso, Braccia-Torso, Torso-Gambe, Gambe-Piedi
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

    # Plot con Legenda
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [4, 1]})
    
    ax1.imshow(img_final)
    ax1.axis('off')
    ax1.set_title(title)

    legend_elements = []
    sorted_ids = sorted([k for k in class_colors.keys() if k in classes])
    
    for k in sorted_ids:
        c = class_colors[k]
        name = result.names[k].capitalize()
        if k in final_centroids:
            cx, cy = final_centroids[k]
            label_text = f"{name}\nX={cx}, Y={cy}"
        else:
            label_text = name
        legend_elements.append(Patch(facecolor=np.array(c)/255, edgecolor='black', label=label_text))
    
    ax2.legend(handles=legend_elements, loc='center', title="Legenda & Coordinate", fontsize=12)
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    MODEL_PATH = 'runs/segment/body_parts6/weights/best.pt' 
    
    # Path per il test random
    VAL_IMAGES_PATH = 'assets/cihp-DatasetNinja/processed/images/val'
    
    # Path per il test specifico
    TEST_IMAGE_PATH = 'digital-image-processing/caduta.jpg'
    TEST_IMAGE_PATH_2 = 'digital-image-processing/caduta.png'
    TEST_IMAGE_PATH_3 = 'digital-image-processing/inpiedi.png'

    # Test random (giusto per confronto)
    test_random_image(MODEL_PATH, VAL_IMAGES_PATH, conf_threshold=0.25)

    # Test specifico su "caduta.jpg"
    test_specific_image(MODEL_PATH, TEST_IMAGE_PATH, conf_threshold=0.25)
    
    # Test specifico su "caduta.png"
    test_specific_image(MODEL_PATH, TEST_IMAGE_PATH_2, conf_threshold=0.25)
    
    # Test specifico su "inpiedi.png"
    test_specific_image(MODEL_PATH, TEST_IMAGE_PATH_3, conf_threshold=0.25)