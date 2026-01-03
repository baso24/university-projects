import os
import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from torchvision.utils import save_image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# Denormalizza le immagini, ogni pixel passa dal range [-1, 1] a [0, 1]
# Serve a Matplotlib per visualizzare correttamente le immagini
# Nel training normalizziamo le immagini in [-1, 1] per far funzionare meglio la rete dato che l'output del generatore è una tanh
def denorm(img_tensors):
  return img_tensors * 0.5 + 0.5

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    
# Mostra un batch di immagini generate dal generatore in una griglia 4x4
def show_images(images, nmax=16):
  fig, ax = plt.subplots(figsize=(8,8))
  ax.set_xticks([]); ax.set_yticks([])
  ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=4).permute(1, 2, 0))

# Mostra un batch di immagini dal dataloader
def show_batch(dl, nmax=16):
  for images, _ in dl:
    show_images(images, nmax)
    break

# Dataset personalizzato per leggere immagini e attributi dal CSV
class CelebADataset(Dataset):
    # root_dir: directory delle immagini
    # csv_file: file CSV con nomi immagini e attributi
    # transform: trasformazioni da applicare alle immagini
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.attr_names = ['Male', 'Young', 'Blond_Hair', 'Smiling']
        self.data = []
        
        # apro il file csv degli attributi
        print(f"Caricamento attributi da {csv_file}...")
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # trovo gli indici delle colonne degli attributi desiderati nel file csv
            try:
                attr_indices = [header.index(name) for name in self.attr_names]
            except ValueError:
                print(f"Errore: Colonne {self.attr_names} non trovate nell'header.")
                raise

            for row in reader: # ogni row è fatta così: [img_name, attr1, attr2, ..., attr40]
                img_name = row[0]
                img_path = os.path.join(root_dir, img_name) # costruisco il path dell'immagine
                if os.path.exists(img_path): # controllo che l'immagine esista nel dataset
                    attributes = [float(row[i]) for i in attr_indices]
                    self.data.append((img_name, attributes))
                    
        print(f"Immagini valide trovate: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)

    # metodo che definisce cosa succede quando si richiede un elemento del dataset
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name, attributes = self.data[idx] # recupero nome immagine e attributi, da self.data che è: ['000001.jpg', [1.0, -1.0, 1.0, -1.0]]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image) # applico le trasformazioni definite, l'immagine passa da PIL a Tensor
            
        return image, torch.tensor(attributes, dtype=torch.float32)

class DCGANDiscriminator(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.n_classes = n_classes
        
        # L'input ha 3 (RGB) + n_classes canali (n_classes è il numero di attributi)
        self.network = nn.Sequential(
            # in: (3 + n_classes) x 64 x 64
            nn.Conv2d(3 + n_classes, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 64 x 32 x 32

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 16 x 16

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 256 x 8 x 8

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 512 x 4 x 4

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # out: 1 x 1 x 1

            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # x: [batch_size, 3, 64, 64]
        # labels: [batch_size, n_classes]
        
        # Le etichette devono diventare un cubo di dimensione 4x1x1 da concatenare all'immagine
        # Questo cubo viene poi espanso a essere 4x64x64
        # Ogni pixel dell'immagine avrà queste label associate oltre ai suoi 3 canali RGB
        labels = labels.view(labels.size(0), self.n_classes, 1, 1)
        labels = labels.expand(-1, -1, x.size(2), x.size(3))
        
        # Concateniamo i 3 canali RGB con i canali delle label
        x = torch.cat([x, labels], dim=1)
        # Output della rete, appiattito a un vettore di dimensione [batch_size, 1]
        return self.network(x).view(-1, 1)

    def train_step(self, real_images, labels, generator, optimizer):
        # Elimino i gradienti calcolati nel passo precedente per evitare accomulazioni
        optimizer.zero_grad()
        
        device = real_images.device
        batch_size = real_images.size(0)

        # Passo le immagini e le etichette reali al disciriminatore (chiamo forward)
        real_preds = self(real_images, labels)
        # Vettore di 1 (il discriminatore deve riconoscere che queste sono immagini vere)
        real_targets = torch.ones(real_images.size(0), 1, device=device)
        # Calcolo la loss per le immagini reali
        real_loss = F.binary_cross_entropy(real_preds, real_targets)
        # Metrica di monitoraggio, più lo score si avvicina a 1 più significa che è bravo a riconoscere immagini reali
        real_score = torch.mean(real_preds).item()

        # Genero immagini false con il generatore a partire da rumore casuale e le etichette uguali a quelle passate prima per le immagini reali
        latent = torch.randn(batch_size, generator.latent_size, 1, 1, device=device)
        fake_images = generator(latent, labels)

        # Passo le immagini false e le etichette al discriminatore
        fake_preds = self(fake_images, labels)
        # Vettore di 0 (il discriminatore deve riconoscere che queste sono immagini false)
        fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
        # Calcolo la loss per le immagini false
        fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
        # Metrica di monitoraggio, più lo score si avvicina a 0 più significa che è bravo a riconoscere immagini false
        fake_score = torch.mean(fake_preds).item()

        # La perdita totale è la somma delle due perdite
        loss = real_loss + fake_loss
        loss.backward()
        optimizer.step()
        return loss.item(), real_score, fake_score

class DCGANGenerator(nn.Module):
    def __init__(self, latent_size, n_classes=3):
        super().__init__()
        self.latent_size = latent_size
        self.n_classes = n_classes
        
        self.network = nn.Sequential(
            # input: (latent_size + n_classes) x 1 x 1
            nn.ConvTranspose2d(latent_size + n_classes, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # out: 512 x 4 x 4

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # out: 256 x 8 x 8

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # out: 128 x 16 x 16

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # out: 64 x 32 x 32

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # out: 3 x 64 x 64
        )

    def forward(self, x, labels):
        # x: [batch_size, latent_size, 1, 1]
        # labels: [batch_size, n_classes]
        
        # Estendo le label a 4x1x1 per poterle concatenare al vettore di rumore
        labels = labels.view(labels.size(0), self.n_classes, 1, 1)
        # Concateno il rumore con le label
        x = torch.cat([x, labels], dim=1)
        return self.network(x)

    def train_step(self, discriminator, labels, optimizer, batch_size, device):
        # Elimino i gradienti calcolati nel passo precedente per evitare accomulazioni
        optimizer.zero_grad()

        # Genero rumore casuale
        latent = torch.randn(batch_size, self.latent_size, 1, 1, device=device)
        # Il generatore prova a creare un'immagine che soddisfi le label fornite
        fake_images = self(latent, labels)

        # Il discriminatore valuta se l'immagine generata sembra reale e coerente con le label
        preds = discriminator(fake_images, labels)
        # Vettore di 1 (il generatore vuole che il discriminatore pensi che queste immagini siano reali)
        targets = torch.ones(batch_size, 1, device=device)
        # Calcolo la loss
        loss = F.binary_cross_entropy(preds, targets)

        loss.backward()
        optimizer.step()

        return loss.item()

class GANDiscriminator(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.n_classes = n_classes
        self.img_flat_size = 3 * 64 * 64
        
        """Primo modello più semplice
        self.network = nn.Sequential(
            nn.Linear(self.img_flat_size + n_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        """

        #Nuovo modello più complesso
        self.network = nn.Sequential(
            nn.Linear(self.img_flat_size + n_classes, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        

    def forward(self, x, labels):
        # Appiattiamo l'immagine: [batch, 3, 64, 64] -> [batch, 12288]
        x = x.view(x.size(0), -1)
        # Concateniamo immagine appiattita e label
        x = torch.cat([x, labels], dim=1)
        return self.network(x)

    def train_step(self, real_images, labels, generator, optimizer):
        # Elimino i gradienti calcolati nel passo precedente per evitare accomulazioni
        optimizer.zero_grad()
        
        device = real_images.device
        batch_size = real_images.size(0)

        # Passo le immagini e le etichette reali al disciriminatore (chiamo forward)
        real_preds = self(real_images, labels)
        # Vettore di 1 (il discriminatore deve riconoscere che queste sono immagini vere)
        real_targets = torch.ones(real_images.size(0), 1, device=device)
        # Calcolo la loss per le immagini reali
        real_loss = F.binary_cross_entropy(real_preds, real_targets)
        # Metrica di monitoraggio, più lo score si avvicina a 1 più significa che è bravo a riconoscere immagini reali
        real_score = torch.mean(real_preds).item()

        # Genero immagini false con il generatore a partire da rumore casuale e le etichette uguali a quelle passate prima per le immagini reali
        latent = torch.randn(batch_size, generator.latent_size, device=device)
        fake_images = generator(latent, labels)

        # Passo le immagini false e le etichette al discriminatore
        fake_preds = self(fake_images, labels)
        # Vettore di 0 (il discriminatore deve riconoscere che queste sono immagini false)
        fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
        # Calcolo la loss per le immagini false
        fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
        # Metrica di monitoraggio, più lo score si avvicina a 0 più significa che è bravo a riconoscere immagini false
        fake_score = torch.mean(fake_preds).item()

        # La perdita totale è la somma delle due perdite
        loss = real_loss + fake_loss
        loss.backward()
        optimizer.step()
        return loss.item(), real_score, fake_score

class GANGenerator(nn.Module):
    def __init__(self, latent_size, n_classes=3):
        super().__init__()
        self.latent_size = latent_size
        self.n_classes = n_classes
        self.img_shape = (3, 64, 64)
        self.img_flat_size = 3 * 64 * 64

        """Primo modello (più semplice)
        self.network = nn.Sequential(
            nn.Linear(latent_size + n_classes, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.img_flat_size),
            nn.Tanh()
        )
        """

        #Nuovo modello più complesso
        self.network = nn.Sequential(
            nn.Linear(latent_size + n_classes, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Linear(4096, self.img_flat_size),
            nn.Tanh()
        )
        

    def forward(self, x, labels):
        # x: [batch_size, latent_size]
        # labels: [batch_size, n_classes]

        # Concateno il rumore con le label
        x = torch.cat([x, labels], dim=1)
        # Output della rete
        out = self.network(x)
        # Reshape da [batch_size, img_flat_size] a [batch_size, 3, 64, 64]
        # (Batch_Size, Canali, Altezza, Larghezza), pronto per essere visualizzato dalle librerie come MatplotLib o passato al discriminatore.
        return out.view(out.size(0), *self.img_shape)

    def train_step(self, discriminator, labels, optimizer, batch_size, device):
        # Elimino i gradienti calcolati nel passo precedente per evitare accomulazioni
        optimizer.zero_grad()

        # Genero rumore casuale
        latent = torch.randn(batch_size, self.latent_size, device=device)
        # Il generatore prova a creare un'immagine che soddisfi le label fornite
        fake_images = self(latent, labels)

        # Il discriminatore valuta se l'immagine generata sembra reale e coerente con le label
        preds = discriminator(fake_images, labels)
        # Vettore di 1 (il generatore vuole che il discriminatore pensi che queste immagini siano reali)
        targets = torch.ones(batch_size, 1, device=device)
        # Calcolo la loss
        loss = F.binary_cross_entropy(preds, targets)

        loss.backward()
        optimizer.step()

        return loss.item()

def save_images(index, latent_tensor, labels, generator, sample_dir, attr_names, show=True):
    # Faccio generare le immagini al generatore
    fake_images = generator(latent_tensor, labels)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    
    images = denorm(fake_images).cpu().detach()
    
    n_images = images.size(0)
    nrow = int(math.sqrt(n_images))
    ncol = math.ceil(n_images / nrow)
    
    # Crea una figura con subplots per mostrare titoli individuali
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*3, nrow*3))
    # trasformo la matrice di assi in un array 1D
    axes = axes.flatten() 
    
    for i in range(n_images):
        ax = axes[i]
        # PyTorch memorizza le immagini come [Canali, Altezza, Larghezza]
        # Matplotlib le vuole come [Altezza, Larghezza, Canali]
        ax.imshow(images[i].permute(1, 2, 0))
        # Rimuovo gli assi dalla visualizzazione
        ax.set_xticks([]); ax.set_yticks([])
        
        # Crea il titolo basato sulle label
        title_parts = []
        for j, name in enumerate(attr_names):
            is_true = labels[i][j] > 0
            if name == 'Male':
                title_parts.append("Male" if is_true else "Female")
            elif name == 'Young':
                title_parts.append("Young" if is_true else "Old")
            elif name == 'Blond':
                title_parts.append("Blond" if is_true else "Not Blond")
            elif name == 'Smiling':
                title_parts.append("Smiling" if is_true else "Not Smiling")
            elif is_true:
                title_parts.append(name)
        
        title = "\n".join(title_parts)
        ax.set_title(title, fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, fake_fname))
    print('Saving', fake_fname)
    
    if show:
        plt.show()
    else:
        plt.close()

def train(EPOCHS, LEARNING_RATE, discriminator, generator, train_dl, device, input_noise, fixed_labels, sample_dir, attr_names, start_idx=1):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    optimizer_d = torch.optim.Adam(discriminator.parameters(), LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_g = torch.optim.Adam(generator.parameters(), LEARNING_RATE, betas=(0.5, 0.999))
    
    print("----- Inizio del training -----")
    
    for epoch in range(EPOCHS):
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        epoch_real_score = 0.0
        epoch_fake_score = 0.0
        
        # Uso tqdm per vedere il progresso batch per batch
        # Mi serve per capire il tempo che ci metterà per ogni epoca
        pbar = tqdm(train_dl, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        # Per ogni batch di 128 immagini reali
        for i, (real_images, labels) in enumerate(pbar):
            real_images = real_images.to(device)
            labels = labels.to(device)
            
            # Train generator
            loss_g = generator.train_step(discriminator, labels, optimizer_g, real_images.size(0), device)
            
            # Train discriminator
            loss_d, real_score, fake_score = discriminator.train_step(real_images, labels, generator, optimizer_d)

            # Accumulo i valori per fare la media a fine epoca
            epoch_loss_g += loss_g
            epoch_loss_d += loss_d
            epoch_real_score += real_score
            epoch_fake_score += fake_score

            # Aggiorno la barra di tqdm con le loss rispettive
            pbar.set_postfix({'loss_d': f'{loss_d:.4f}', 'loss_g': f'{loss_g:.4f}'})

        # Calcolo la media delle loss dell'epoca dividendo per il numero di immagini nel batch
        avg_loss_g = epoch_loss_g / len(train_dl)
        avg_loss_d = epoch_loss_d / len(train_dl)
        avg_real_score = epoch_real_score / len(train_dl)
        avg_fake_score = epoch_fake_score / len(train_dl)

        losses_g.append(avg_loss_g)
        losses_d.append(avg_loss_d)
        real_scores.append(avg_real_score)
        fake_scores.append(avg_fake_score)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(epoch+1, EPOCHS, avg_loss_g, avg_loss_d, avg_real_score, avg_fake_score))
        
        #Salvo le immagini generate in questa epoca
        save_images(epoch+start_idx, input_noise, fixed_labels, generator, sample_dir, attr_names, show=False)
    
    return losses_g, losses_d, real_scores, fake_scores

if __name__ == '__main__':
    
    print("Seleziona il modello da utilizzare:")
    print("1. DCGAN")
    print("2. GAN")
    choice = input("Inserisci il numero (1 o 2): ")
    
    model_prefix = "dcgan"
    if choice == '2':
        model_prefix = "gan"
        print("Hai selezionato: GAN")
    else:
        print("Hai selezionato: DCGAN (default)")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    IMG_DIR = os.path.join(current_dir, '../assets/archive/img_align_celeba/img_align_celeba/')
    ATTR_CSV = os.path.join(current_dir, '../assets/archive/list_attr_celeba.csv')
    
    DEVICE = get_device()
    # Hyperparametri di training cambiabili a piacimento
    SUBSET_DIM = 100000 
    EPOCHS = 100
    LEARNING_RATE = 0.0002
    
    # Valori "fissi" scelti per il training
    latent_size = 128
    image_size = 64
    batch_size = 128
    n_classes = 4
    attr_names = ['Male', 'Young', 'Blond', 'Smiling']
    
    # Generariamo un'immagine per ogni combinazione possibile di attributi (2^4 = 16)
    generated_samples_count = 16

    # Definizione delle trasformazioni da applicare alle immagini e caricamento del dataset
    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    full_dataset = CelebADataset(root_dir=IMG_DIR, csv_file=ATTR_CSV, transform=transform)

    # Test purposes
    if len(full_dataset) == 0:
        print("ERRORE: Nessuna immagine trovata. Verifica che il percorso IMG_DIR sia corretto.")
        exit()

    # Creazione subset del dataset
    subset_size = min(SUBSET_DIM, len(full_dataset))
    random_indices = torch.randperm(len(full_dataset))[:subset_size]
    train_dataset = torch.utils.data.Subset(full_dataset, random_indices)

    # Vediamo la distribuzione delle classi nel subset creato
    print("Analisi distribuzione classi nel subset...")
    stats = torch.zeros(n_classes)
    total = len(train_dataset)
    subset_indices = train_dataset.indices
    count_pos = [0] * n_classes
    for idx in subset_indices:
        _, attrs = train_dataset.dataset.data[idx] 
        for i in range(n_classes):
            if attrs[i] > 0:
                count_pos[i] += 1
    print(f"Totale immagini subset: {total}")
    for i, name in enumerate(attr_names):
        print(f"{name}: {count_pos[i]} positivi ({count_pos[i]/total*100:.2f}%)")

    # Dataloader che divide il dataset in batch e mescola i dati ad ogni epoca.
    # 'num_workers' specifica il numero di subprocessi da usare per caricare le immagini dal disco alla RAM.
    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=(DEVICE.type == 'cuda'))
    
    # Test purposes
    print(f"Immagini caricate: {len(train_dataset)}")
    
    if model_prefix == "gan":
        discriminator = GANDiscriminator(n_classes=n_classes).to(DEVICE)
        generator = GANGenerator(latent_size, n_classes=n_classes).to(DEVICE)
        input_noise = torch.randn(generated_samples_count, latent_size, device=DEVICE)
    else:
        discriminator = DCGANDiscriminator(n_classes=n_classes).to(DEVICE)
        generator = DCGANGenerator(latent_size, n_classes=n_classes).to(DEVICE)
        input_noise = torch.randn(generated_samples_count, latent_size, 1, 1, device=DEVICE)

    # Directory per salvare le immagini generate durante il training
    generated_images_dir = f'{model_prefix}.generated'
    os.makedirs(generated_images_dir, exist_ok=True)
    
    # Directory permanente basata sui parametri di training
    permanent_dir = f'{model_prefix}.{SUBSET_DIM}subset.{EPOCHS}epochs.{LEARNING_RATE}lr'
    os.makedirs(permanent_dir, exist_ok=True)

    # Generiamo tutte le combinazioni possibili di attributi (2^4 = 16 combinazioni)
    labels_list = []
    for i in range(16):
        # Logica bitwise per creare combinazioni di -1 e 1
        l = [1 if (i >> (3-bit)) & 1 else -1 for bit in range(4)]
        labels_list.append(l)
    labels = torch.tensor(labels_list, device=DEVICE).float()

    # TRAINING
    # Mi restituisce loss del generatore e del discriminatore per ogni epoca, insieme agli scores dei reali e falsi del discriminatore
    losses_g, losses_d, real_scores, fake_scores = train(EPOCHS, LEARNING_RATE, discriminator, generator, train_dl, DEVICE, input_noise, labels, generated_images_dir, attr_names)
    
    models_dir = os.path.join(current_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(models_dir, f'{model_prefix}.generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(models_dir, f'{model_prefix}.discriminator.pth'))
    print(f"Modelli salvati in '{models_dir}'")
    
    # Plot losses
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(losses_g, label="Generator")
    plt.plot(losses_d, label="Discriminator")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(generated_images_dir, 'losses.png'))
    plt.savefig(os.path.join(permanent_dir, 'losses.png'))
    
    # Plot real and fake scores
    plt.figure(figsize=(10,5))
    plt.title("Real and Fake Scores of Discriminator During Training")
    plt.plot(real_scores, label="Real")
    plt.plot(fake_scores, label="Fake")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(os.path.join(generated_images_dir, 'scores.png'))
    plt.savefig(os.path.join(permanent_dir, 'scores.png'))
    
    # Salvataggio delle immagini generate finali
    save_images(EPOCHS+1, input_noise, labels, generator, generated_images_dir, attr_names, show=False)
    save_images(EPOCHS+1, input_noise, labels, generator, permanent_dir, attr_names, show=True)