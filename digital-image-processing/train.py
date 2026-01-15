"""
Progetto: Pose Estimation and Fall Detection - Training Pipeline
Autore: Student - AI and Automation Engineering
"""

import os
import json
import base64
import zlib
import yaml
import random
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import albumentations as A

class BodyPartDatasetPreprocessor:
    def __init__(self, dataset_path, output_path, subset_ratio=1.0):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.subset_ratio = subset_ratio
        
        # Mapping classi (Semplificato per il tuo task)
        # 0: Background, 1: Testa, 2: Braccia, 3: Torso, 4: Gambe, 5: Piedi
        self.target_classes = ['testa', 'braccia', 'torso', 'gambe', 'piedi']
        self.debug_counter = 0

    def get_image_paths(self):
        """
        Recupera le immagini specificamente dalla cartella 'imm'
        """
        
        # Cerca cartella immagini (imm o img o images)
        img_dir = None
        for d in ['imm', 'img', 'images']:
            if (self.dataset_path / d).exists():
                img_dir = self.dataset_path / d
                break
        
        if img_dir is None:
            print(f"ERRORE CRITICO: Nessuna cartella immagini (imm/img) trovata in {self.dataset_path}!")
            return []

        # Cerca estensioni comuni
        exts = ['*.jpg', '*.jpeg', '*.png']
        all_images = []
        for ext in exts:
            found = list(img_dir.glob(ext))
            all_images.extend(found)
            
        print(f"Trovate {len(all_images)} immagini in {img_dir.name}")
        
        # Subset logic
        if self.subset_ratio < 1.0:
            subset_size = int(len(all_images) * self.subset_ratio)
            # Shuffle deterministico per riproducibilità se serve, altrimenti random
            random.shuffle(all_images)
            all_images = all_images[:subset_size]
            print(f"Utilizzo un subset del {self.subset_ratio*100}%: {len(all_images)} immagini")
            
        return all_images

    def parse_dataset_ninja_json(self, json_path, width, height):
        """
        Parser specifico per il formato DatasetNinja / Supervisely
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            yolo_annotations = []
            
            # DatasetNinja di solito usa 'objects' o 'shapes'
            objects = data.get('objects', data.get('shapes', []))
            
            for obj in objects:
                # Recupera la label
                label = obj.get('classTitle', obj.get('label', '')).lower()
                polygons = []

                # --- GESTIONE BITMAP (Supervisely/DatasetNinja) ---
                if 'bitmap' in obj and 'data' in obj['bitmap']:
                    try:
                        bitmap_data = obj['bitmap']['data']
                        origin = obj['bitmap']['origin'] # [x, y]
                        
                        # Decode base64 + zlib -> PNG -> Mask
                        compressed_bytes = base64.b64decode(bitmap_data)
                        decompressed_bytes = zlib.decompress(compressed_bytes)
                        nparr = np.frombuffer(decompressed_bytes, np.uint8)
                        mask_crop = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                        
                        if mask_crop is not None:
                            # Gestione canali (Alpha o Grayscale)
                            if len(mask_crop.shape) > 2:
                                mask_crop = mask_crop[:, :, 3] if mask_crop.shape[2] == 4 else cv2.cvtColor(mask_crop, cv2.COLOR_BGR2GRAY)
                            
                            # Trova contorni
                            _, mask_bin = cv2.threshold(mask_crop, 127, 255, cv2.THRESH_BINARY)
                            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            for cnt in contours:
                                if len(cnt) < 3: continue
                                # Offset coordinate con origin
                                pts = cnt.reshape(-1, 2).astype(float)
                                pts[:, 0] += origin[0]
                                pts[:, 1] += origin[1]
                                polygons.append(pts.tolist())
                    except Exception as e:
                        if self.debug_counter < 10:
                            print(f"Errore bitmap {label}: {e}")
                            self.debug_counter += 1

                # --- GESTIONE POLIGONI (LabelMe/Standard) ---
                elif 'points' in obj:
                    if isinstance(obj['points'], dict) and 'exterior' in obj['points']:
                        polygons.append(obj['points']['exterior'])
                    elif isinstance(obj['points'], list):
                        polygons.append(obj['points'])
                
                if not polygons:
                    # Skip silenzioso se non troviamo geometria valida
                    continue

                # Mapping Classi (Logica custom basata sui nomi CIHP)
                class_id = -1
                if any(x in label for x in ['hat', 'hair', 'face', 'sunglass', 'scarf']): class_id = 0 # Testa
                elif any(x in label for x in ['arm', 'glove', 'hand']): class_id = 1 # Braccia
                elif any(x in label for x in ['coat', 'dress', 'upper', 'torso', 'jumpsuit']): class_id = 2 # Torso
                elif any(x in label for x in ['leg', 'pants', 'skirt']): class_id = 3 # Gambe
                elif any(x in label for x in ['shoe', 'sock', 'boot']): class_id = 4 # Piedi

                if class_id != -1:
                    for points in polygons:
                        # Normalizzazione YOLO (x_norm y_norm)
                        flat_points = []
                        for pt in points:
                            # Clamp points to image dimensions to avoid errors
                            x = min(max(pt[0], 0), width)
                            y = min(max(pt[1], 0), height)
                            flat_points.extend([x / width, y / height])
                        
                        if len(flat_points) >= 6:
                            segment_str = f"{class_id} " + " ".join(f"{p:.6f}" for p in flat_points)
                            yolo_annotations.append(segment_str)
                else:
                    if self.debug_counter < 20:
                        print(f"Label ignorata: '{label}' (File: {json_path.name})")
                        self.debug_counter += 1
            
            return yolo_annotations

        except Exception as e:
            print(f"Errore lettura JSON {json_path.name}: {e}")
            return []

    def process_dataset(self, train_split=0.8):
        print("\nInizio pre-processing...")
        
        # Setup directories
        dirs = {
            'train_img': self.output_path / 'images' / 'train',
            'val_img': self.output_path / 'images' / 'val',
            'train_lbl': self.output_path / 'labels' / 'train',
            'val_lbl': self.output_path / 'labels' / 'val'
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        images = self.get_image_paths()
        if not images:
            return

        # Split
        random.shuffle(images)
        split_idx = int(len(images) * train_split)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        self._process_batch(train_imgs, dirs['train_img'], dirs['train_lbl'], "Train")
        self._process_batch(val_imgs, dirs['val_img'], dirs['val_lbl'], "Val")

    def _process_batch(self, image_list, img_out, lbl_out, stage_name):
        print(f"\nProcessando {stage_name} set ({len(image_list)} immagini)...")
        
        for idx, img_path in enumerate(image_list):
            if idx % 50 == 0: print(f"   {idx}/{len(image_list)}...", end='\r')
            
            # 1. Carica Immagine
            img = cv2.imread(str(img_path))
            if img is None: continue
            h, w = img.shape[:2]

            # 2. Trova Annotazione JSON
            # STRUTTURA: training/imm/foto.jpg -> training/ann/foto.jpg.json
            json_path = self.dataset_path / 'ann' / f"{img_path.name}.json"
            
            if not json_path.exists():
                # Fallback: prova senza .jpg nel nome json se il primo fallisce
                json_path = self.dataset_path / 'ann' / f"{img_path.stem}.json"
            
            if not json_path.exists():
                if idx < 5: print(f"JSON non trovato: {json_path}")
                continue

            annotations = []
            if json_path.exists():
                annotations = self.parse_dataset_ninja_json(json_path, w, h)
                if not annotations and idx < 5:
                    print(f"JSON vuoto o classi non riconosciute per: {img_path.name}")
            
            # 3. Salva solo se ci sono annotazioni o se vogliamo anche negativi (qui filtro)
            if annotations:
                # Salva Immagine
                cv2.imwrite(str(img_out / img_path.name), img)
                
                # Salva Label
                with open(lbl_out / f"{img_path.stem}.txt", 'w') as f:
                    f.write('\n'.join(annotations))

class YOLOBodySegmentationTrainer:
    def __init__(self, dataset_yaml, model_name='yolov8n-seg.pt'):
        self.dataset_yaml = dataset_yaml
        self.model = YOLO(model_name)
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        print(f"Device: {self.device}")

    def train(self, epochs=50, batch=16, imgsz=640, project_dir='runs/segment'):
        self.model.train(
            data=self.dataset_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=self.device,
            project=project_dir,
            name='body_parts_v1',
            exist_ok=False,
            val=True,
            workers=4
        )

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    
    # --- CONFIGURAZIONE TRAINING ---
    SUBSET_RATIO = 0.05  # Percentuale dataset da usare (0.05 = 5%)
    EPOCHS = 1           # Numero di epoche
    BATCH_SIZE = 32      # Dimensione batch
    IMG_SIZE = 320       # Dimensione immagini

    # 1. DEFINIZIONE PATH CORRETTA
    # __file__ è dentro digital-image-processing/train.py
    CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent 
    
    # PROJECT_ROOT è la cartella che contiene sia 'digital-image-processing' che 'assets'
    PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
    
    # Costruiamo il path come descritto da te
    RAW_DATA_PATH = PROJECT_ROOT / "assets" / "cihp-DatasetNinja" / "training"
    PROCESSED_PATH = PROJECT_ROOT / "assets" / "cihp-DatasetNinja" / "processed"

    print("="*50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Input Data:   {RAW_DATA_PATH}")
    print(f"Output Data:  {PROCESSED_PATH}")
    print("="*50)

    # Verifica esistenza path prima di partire
    if not RAW_DATA_PATH.exists():
        print("ERRORE: Il percorso Input Data non esiste. Controlla i nomi delle cartelle.")
        exit()

    # 2. PREPROCESSING
    preprocessor = BodyPartDatasetPreprocessor(RAW_DATA_PATH, PROCESSED_PATH, subset_ratio=SUBSET_RATIO)
    preprocessor.process_dataset()

    # 3. CONFIGURAZIONE YAML PER YOLO
    yaml_content = {
        'path': str(PROCESSED_PATH.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'testa',
            1: 'braccia',
            2: 'torso',
            3: 'gambe',
            4: 'piedi'
        }
    }
    
    yaml_path = PROCESSED_PATH / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"\nDataset YAML creato: {yaml_path}")

    # 4. TRAINING
    # Verifica che ci siano dati processati
    train_files = list((PROCESSED_PATH / 'images' / 'train').glob('*'))
    if not train_files:
        print("\nERRORE: Nessuna immagine è stata salvata nella cartella processed.")
        exit()

    # Avvio training
    print(f"\nDataset pronto con {len(train_files)} immagini di training.")
    print("\nAvvio Training YOLO...")
    trainer = YOLOBodySegmentationTrainer(str(yaml_path), 'yolov8n-seg.pt')
    
    # Usa path assoluto per evitare duplicazioni (es. runs/segment/runs/segment)
    runs_path = PROJECT_ROOT / 'runs' / 'segment'
    trainer.train(epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE, project_dir=str(runs_path))

    print(f"\nTraining completato! Controlla la cartella '{runs_path}' per i risultati e il modello (weights/best.pt).")