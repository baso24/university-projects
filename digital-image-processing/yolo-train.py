"""
Progetto: Pose Estimation and Fall Detection - Training
Autore: Valentino Basili, Giovanni Paolo Maugeri
"""

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
import shutil
import os

class BodyPartDatasetPreprocessor:
    def __init__(self, dataset_path, output_path, subset_ratio):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.subset_ratio = subset_ratio
        self.cihp_map = {
            # CLASSE 0: TESTA
            'face': 0, 
            'hair': 0, 
            'hat': 0, 
            'sunglasses': 0, 
            'scarf': 0,

            # CLASSE 1: BRACCIA
            'left_arm': 1, 
            'right_arm': 1, 
            'glove': 1,

            # CLASSE 2: TORSO
            'torso_skin': 2, 
            'coat': 2, 
            'dress': 2, 
            'upperclothes': 2,

            # CLASSE 3: GAMBE
            'pants': 3, 
            'left_leg': 3, 
            'right_leg': 3, 
            'skirt': 3,

            # CLASSE 4: PIEDI
            'left_shoe': 4, 
            'right_shoe': 4, 
            'socks': 4
        }

    def get_image_paths(self):
        img_dir = self.dataset_path / "img"
        
        if img_dir is None:
            print(f"ERRORE: Nessuna cartella img trovata in {self.dataset_path}!")
            return []

        ext = '*.jpg'
        all_images = list(img_dir.glob(ext))
            
        print(f"Trovate {len(all_images)} immagini in {img_dir.name}")
        
        if self.subset_ratio < 1.0:
            subset_size = int(len(all_images) * self.subset_ratio)
            random.shuffle(all_images)
            all_images = all_images[:subset_size]
            print(f"Utilizzo un subset del {self.subset_ratio*100}%: {len(all_images)} immagini")
            
        return all_images

    def parse_dataset_ninja_json(self, json_path, width, height):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            yolo_annotations = []

            objects = data.get('objects')
            
            for obj in objects:
                # Recupera la label
                label = obj.get('classTitle', obj.get('label', '')).lower()
                polygons = []

                # Gestione bitmap
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
                            # Gestione canali
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
                        print(f"Errore bitmap {label}: {e}")

                # Gestione poligoni
                elif 'points' in obj:
                    if isinstance(obj['points'], dict) and 'exterior' in obj['points']:
                        polygons.append(obj['points']['exterior'])
                    elif isinstance(obj['points'], list):
                        polygons.append(obj['points'])
                
                if not polygons:
                    # Skip silenzioso se non troviamo geometria valida
                    continue

                # Pulisci la label (minuscolo e rimuovi spazi extra per sicurezza)
                clean_label = label.lower().strip()
                # Se la label è nel dizionario, prendi l'ID, altrimenti -1
                class_id = self.cihp_map.get(clean_label, -1)

                if class_id != -1:
                    for points in polygons:
                        # Normalizzazione YOLO
                        flat_points = []
                        for pt in points:
                            x = min(max(pt[0], 0), width)
                            y = min(max(pt[1], 0), height)
                            flat_points.extend([x / width, y / height])
                        
                        if len(flat_points) >= 6:
                            segment_str = f"{class_id} " + " ".join(f"{p:.6f}" for p in flat_points)
                            yolo_annotations.append(segment_str)
            
            return yolo_annotations

        except Exception as e:
            print(f"Errore lettura JSON {json_path.name}: {e}")
            return []

    def process_dataset(self, train_split): 
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

        # Splitting train/val
        random.shuffle(images)
        split_idx = int(len(images) * train_split)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        self.process_batch(train_imgs, dirs['train_img'], dirs['train_lbl'], "Train")
        self.process_batch(val_imgs, dirs['val_img'], dirs['val_lbl'], "Val")

    def process_batch(self, image_list, img_out, lbl_out, stage_name):
        print(f"\nProcessando {stage_name} set ({len(image_list)} immagini)...")
        
        for idx, img_path in enumerate(image_list):
            if idx % 50 == 0: print(f"   {idx}/{len(image_list)}...", end='\r')
            
            # Carica Immagine
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]

            # Trova Annotazione JSON
            # STRUTTURA: training/imm/foto.jpg -> training/ann/foto.jpg.json
            json_path = self.dataset_path / 'ann' / f"{img_path.name}.json"

            annotations = []
            if json_path.exists():
                annotations = self.parse_dataset_ninja_json(json_path, w, h)
                if not annotations:
                    print(f"JSON vuoto o classi non riconosciute per: {img_path.name}")
            
            if annotations:
                # Salva Immagine
                cv2.imwrite(str(img_out / img_path.name), img)
                
                # Salva Label
                with open(lbl_out / f"{img_path.stem}.txt", 'w') as f:
                    f.write('\n'.join(annotations))

class YOLOBodySegmentationTrainer:
    def __init__(self, dataset_yaml, model_name):
        self.dataset_yaml = dataset_yaml
        self.model = YOLO(model_name)
        if torch.cuda.is_available():
            self.device = "cuda"
        # mps non funziona bene con yolo, ci mette più tempo che con la cpu
        #elif torch.backends.mps.is_available():
            #self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Device utilizzato: {self.device}")

    def train(self, epochs, batch, imgsz, project_dir):
        self.model.train(
            data=self.dataset_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=self.device,
            project=project_dir,
            name='body_parts',
            exist_ok=False,
            val=True,
            workers=4,
            conf=0.25,
            max_det=100,
            plots=True
        )

# ========================================== main ==========================================
if __name__ == "__main__":
    
    # Configurazione training
    SUBSET_RATIO = 0.1    # Percentuale dataset da usare (0.05 = 5%)
    EPOCHS = 300          # Numero di epoche
    BATCH_SIZE = 32       # Dimensione batch
    IMG_SIZE = 416        # Dimensione immagini

    # Parent directory dello script corrente
    CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent 
    
    # Project root, cioè cartella che contiene il parent dello script e assets
    PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
    
    # Costruisco i path completi per input e output delle immagini
    RAW_DATA_PATH = PROJECT_ROOT / "assets" / "cihp-DatasetNinja" / "training"
    PROCESSED_PATH = PROJECT_ROOT / "assets" / "cihp-DatasetNinja" / "processed"

    print("="*50)
    print(f"Current Script Dir: {CURRENT_SCRIPT_DIR}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Input Data:   {RAW_DATA_PATH}")
    print(f"Output Data:  {PROCESSED_PATH}")
    print("="*50)

    # Preprocessing dataset, se la cartella già esiste la rimuovo per evitare di mescolare vecchi dati con nuovi
    if PROCESSED_PATH.exists():
        print(f"Rimozione vecchia cartella processed per generare nuovo subset...")
        shutil.rmtree(PROCESSED_PATH)

    # Creo il preprocessor che si occupa di processare le immagini e le annotazioni
    # Devono essere processate per essere compatibili con il formato YOLO
    preprocessor = BodyPartDatasetPreprocessor(RAW_DATA_PATH, PROCESSED_PATH, subset_ratio=SUBSET_RATIO)
    train_split = 0.8  # 80% train, 20% val
    preprocessor.process_dataset(train_split=train_split)

    # Yaml per yolo
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

    # Verifico che ci siano immagini processate pronte per il training
    train_files = list((PROCESSED_PATH / 'images' / 'train').glob('*'))
    if not train_files:
        print("\nERRORE: Nessuna immagine è stata salvata nella cartella processed.")
        exit()
    print(f"\nDataset pronto con {len(train_files)} immagini di training.")
    
    # Path al modello da cui partire
    runs_path = PROJECT_ROOT / 'runs' / 'segment'
    
    # Configurazione modello
    path_to_best_model = 'yolo26n-seg.pt' 

    # Cerca l'ultimo modello best.pt disponibile nelle run precedenti
    # Disabilitato per il momento
    """
    if runs_path.exists():
        # Ottieni sottocartelle ordinate per data di modifica (più recenti prima)
        subdirs = sorted([d for d in runs_path.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
        for d in subdirs:
            candidate = d / 'weights' / 'best.pt'
            if candidate.exists():
                path_to_best_model = str(candidate)
                print(f"Trovato modello precedente da cui ripartire: {path_to_best_model}")
                break
    """

    # Avvio training
    print("\nAvvio Training YOLO...")
    trainer = YOLOBodySegmentationTrainer(str(yaml_path), path_to_best_model)
    trainer.train(epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE, project_dir=str(runs_path))

    print(f"\nTraining completato! Controlla la cartella '{runs_path}' per i risultati e il modello.")