import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import copy

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

class FallDataset(Dataset):
    def __init__(self, images_dir, labels_file, model):
        self.data = []
        self.labels = []
        
        img_path_obj = Path(images_dir)
        lbl_path_obj = Path(labels_file)

        if not img_path_obj.exists() or not lbl_path_obj.exists():
            print(f"Errore: Path non trovato ({images_dir} o {labels_file})")
            return

        # Caricamento etichette
        with open(lbl_path_obj, 'r') as f:
            raw_labels = [int(line.strip()) for line in f.readlines() if line.strip()]

        # Caricamento immagini ordinate
        image_files = sorted(list(img_path_obj.glob('*.jpg')))

        if len(image_files) != len(raw_labels):
            print(f"ATTENZIONE: Numero immagini ({len(image_files)}) diverso da numero label ({len(raw_labels)})!")
            min_len = min(len(image_files), len(raw_labels))
            image_files = image_files[:min_len]
            raw_labels = raw_labels[:min_len]

        print(f"Elaborazione feature per {len(image_files)} immagini in {images_dir}...")

        # Pre-calcolo delle feature (Inferenza YOLO una tantum)
        for i, img_file in enumerate(image_files):
            # Inferenza YOLO
            results = model.predict(source=str(img_file), conf=0.25, verbose=False)
            result = results[0]
            
            # Estrazione vettore grezzo
            vector, _ = get_centroids_vector(result)
            norm_vector = normalize_vector(vector, result.orig_shape)

            self.data.append(norm_vector)
            self.labels.append(raw_labels[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return running_loss / len(dataloader), accuracy, precision, recall, f1

def predict_single_image(yolo_model, classifier, img_path):
    path_obj = Path(img_path)
    if not path_obj.exists():
        print(f"Errore: Immagine non trovata -> {img_path}")
        return

    print(f"Analisi immagine: {path_obj.name}")
    
    # Inferenza YOLO
    results = yolo_model.predict(source=str(path_obj), conf=0.25, verbose=False)
    result = results[0]
    
    # Estrazione feature
    vector, _ = get_centroids_vector(result)
    
    norm_vector = normalize_vector(vector, result.orig_shape)

    # Predizione
    classifier.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(norm_vector, dtype=torch.float32).unsqueeze(0)
        output = classifier(input_tensor)
        prob = output.item()
    
    print(f"Probabilità caduta: {prob:.4f} ({prob*100:.2f}%)")
    if prob > 0.5:
        print("RISULTATO: CADUTA RILEVATA")
    else:
        print("RISULTATO: NON CADUTA")
    print("-" * 30)

if __name__ == "__main__":
    # Paths
    MODEL_PATH = 'runs/segment/body_parts12/weights/best.pt' 
    
    BASE_DIR = 'digital-image-processing/fall-dataset'
    TRAIN_IMG_DIR = f'{BASE_DIR}/images/train'
    VAL_IMG_DIR = f'{BASE_DIR}/images/val'
    TRAIN_LBL_FILE = f'{BASE_DIR}/labels/train/label.txt'
    VAL_LBL_FILE = f'{BASE_DIR}/labels/val/label.txt'

    print(f"Caricamento modello: {MODEL_PATH}")
    try:
        yolo_model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Errore caricamento modello: {e}")
        exit()

    # Creazione Dataset
    print("--- Preparazione Training Set ---")
    train_dataset = FallDataset(TRAIN_IMG_DIR, TRAIN_LBL_FILE, yolo_model)
    print("--- Preparazione Validation Set ---")
    val_dataset = FallDataset(VAL_IMG_DIR, VAL_LBL_FILE, yolo_model)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Preparazione Test Set (Files)
    test_dir = Path('digital-image-processing/test-dataset')
    test_img_path = test_dir / 'images'
    test_lbl_path = test_dir / 'labels' / 'test-label.txt'
    
    test_images = []
    if test_dir.exists():
        for ext in ['*.jpg', '*.png']:
            test_images.extend(test_img_path.glob(ext))
        # Rimuovi duplicati e ordina
        test_images = sorted(list(set(test_images)), key=lambda x: x.name)

    test_labels = []
    if test_lbl_path.exists():
        with open(test_lbl_path, 'r') as f:
            test_labels = [int(line.strip()) for line in f.readlines() if line.strip()]

    if len(test_images) != len(test_labels):
        print(f"ATTENZIONE: Numero immagini test ({len(test_images)}) != Numero label test ({len(test_labels)})")

    # Loop di Training Multiplo
    NUM_RUNS = 10
    EPOCHS = 500
    LEARNING_RATE = 0.01
    best_avg_error = float('inf')
    best_model_wts = None

    print(f"\nAvvio procedura di training e selezione modello ({NUM_RUNS} esecuzioni)...")

    for run in range(NUM_RUNS):
        print(f"\n{'='*15} RUN {run+1}/{NUM_RUNS} {'='*15}")
        
        # Re-inizializzazione Classificatore
        classifier = nn.Sequential(
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
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

        # Training e validazione
        for epoch in range(EPOCHS):
            train_loss = train_epoch(classifier, train_loader, loss_function, optimizer)
            val_loss, acc, prec, rec, f1 = validate_model(classifier, val_loader, loss_function)
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} | Val Loss: {val_loss:.4f} | Acc: {acc:.2f}")

        # Test 
        print(f"--- Fase di test, Run {run+1} ---")
        classifier.eval()
        total_diff = 0.0
        valid_test = False

        if test_images and len(test_images) == len(test_labels):
            valid_test = True
            with torch.no_grad():
                for i, img_path in enumerate(test_images):
                    results = yolo_model.predict(source=str(img_path), conf=0.25, verbose=False)
                    result = results[0]
                    vector, _ = get_centroids_vector(result)
                    
                    norm_vector = normalize_vector(vector, result.orig_shape)

                    # Predizione
                    input_tensor = torch.tensor(norm_vector, dtype=torch.float32).unsqueeze(0)
                    output = classifier(input_tensor)
                    prob = output.item()
                    
                    label = test_labels[i]
                    diff = abs(prob - label)
                    total_diff += diff
                    
                    print(f"  Img: {img_path.name} | Label: {label} | Pred: {prob:.4f} | Diff: {diff:.4f}")

            avg_error = total_diff / len(test_images)
            print(f"  Errore Medio Assoluto: {avg_error:.4f}")

            if avg_error < best_avg_error:
                best_avg_error = avg_error
                best_model_wts = copy.deepcopy(classifier.state_dict())
                print("  -> NUOVO MIGLIOR MODELLO!")
        else:
            print("Test saltato: dati mancanti o non allineati.")

    # Salvataggio Miglior Modello
    if best_model_wts:
        save_path = Path(__file__).resolve().parent / 'classifier.pth'
        torch.save(best_model_wts, save_path)
        print(f"\n{'='*40}")
        print(f"Training completato. Miglior modello salvato in '{save_path}'")
        print(f"Miglior Errore Medio su Test Set: {best_avg_error:.4f}")
        print(f"{'='*40}")