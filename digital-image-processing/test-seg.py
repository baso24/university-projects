import random
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

def test_random_image(model_path, images_dir, conf_threshold=0.25):
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
    image_files = list(img_path_obj.glob('*.jpg')) + list(img_path_obj.glob('*.png'))
    if not image_files:
        print("Nessuna immagine trovata.")
        return

    random_image = random.choice(image_files)
    print(f"Processando random: {random_image.name}")

    # Inferenza
    results = model.predict(source=str(random_image), conf=conf_threshold, save=False, verbose=False)
    
    # Visualizzazione
    _show_result(results[0], f"Random Val: {random_image.name}")

def test_specific_image(model_path, image_name="caduta.jpg", conf_threshold=0.25):
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
    # Nota: su pose complesse come le cadute, a volte abbassare la conf aiuta a vedere cosa "pensa" il modello
    print(f"Analizzando l'immagine...")
    results = model.predict(source=str(target_path), conf=conf_threshold, save=False, verbose=False)

    # 4. Visualizzazione
    if not results:
        print("Nessun risultato generato.")
        return

    _show_result(results[0], f"Test Specifico: {image_name}")

def _show_result(result, title):
    """
    Helper function per visualizzare i risultati con Matplotlib
    """
    # Plot annotato da YOLO (in BGR)
    annotated_frame = result.plot()
    
    # BGR -> RGB per Matplotlib
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_frame_rgb)
    plt.axis('off')
    plt.title(title)
    plt.show()

    # Info console
    print(f"Oggetti rilevati: {len(result.boxes)}")
    if result.masks:
        print(f"Classi segmentate: {result.boxes.cls.cpu().numpy()}")
    else:
        print("Nessuna segmentazione trovata.")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    
    MODEL_PATH = 'runs/segment/body_parts2/weights/best.pt' 
    
    # Path per il test random
    VAL_IMAGES_PATH = 'assets/cihp-DatasetNinja/processed/images/val'

    # 1. Esegui il test random (giusto per confronto)
    test_random_image(MODEL_PATH, VAL_IMAGES_PATH, conf_threshold=0.3)

    # 2. Esegui il test specifico su "caduta.jpg"
    test_specific_image(MODEL_PATH, "digital-image-processing/caduta.jpg", conf_threshold=0.2)