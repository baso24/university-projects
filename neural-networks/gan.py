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

def denorm(img_tensors):
  return img_tensors * 0.5 + 0.5

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def show_images(images, nmax=16):
  fig, ax = plt.subplots(figsize=(8,8))
  ax.set_xticks([]); ax.set_yticks([])
  ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=4).permute(1, 2, 0))

def show_batch(dl, nmax=16):
  for images, _ in dl:
    show_images(images, nmax)
    break

# Dataset personalizzato per leggere immagini e attributi dal CSV
class CelebADataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.attr_names = ['Male', 'Young', 'Blond_Hair', 'Smiling']
        self.data = []
        
        print(f"Caricamento dataset da {csv_file}...")
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # Trova gli indici delle colonne
            try:
                attr_indices = [header.index(name) for name in self.attr_names]
            except ValueError:
                print(f"Errore: Colonne {self.attr_names} non trovate nell'header.")
                raise

            first_row = True
            for row in reader:
                img_name = row[0]
                img_path = os.path.join(root_dir, img_name)
                if first_row:
                    print(f"DEBUG: Sto cercando il primo file qui: {os.path.abspath(img_path)}")
                    print(f"Esiste? {os.path.exists(img_path)}")
                    first_row = False
                if os.path.exists(img_path):
                    attributes = [float(row[i]) for i in attr_indices]
                    self.data.append((img_name, attributes))
                    
        print(f"Immagini valide trovate: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name, attributes = self.data[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(attributes, dtype=torch.float32)

class Discriminator(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.n_classes = n_classes
        self.img_flat_size = 3 * 64 * 64
        
        self.network = nn.Sequential(
            nn.Linear(self.img_flat_size + n_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
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
        # Clear discriminator gradients
        optimizer.zero_grad()
        
        device = real_images.device
        batch_size = real_images.size(0)

        # Pass real images through discriminator
        # Passiamo anche le label reali
        real_preds = self(real_images, labels)
        real_targets = torch.ones(real_images.size(0), 1, device=device)
        real_loss = F.binary_cross_entropy(real_preds, real_targets)
        real_score = torch.mean(real_preds).item()

        # Generate fake images
        latent = torch.randn(batch_size, generator.latent_size, 1, 1, device=device)
        fake_images = generator(latent, labels)

        # Pass Fake images through discriminator
        fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
        # Al discriminatore passiamo l'immagine falsa MA con le label "richieste"
        fake_preds = self(fake_images, labels)
        fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
        fake_score = torch.mean(fake_preds).item()

        # Update discriminator weights
        loss = real_loss + fake_loss
        loss.backward()
        optimizer.step()
        return loss.item(), real_score, fake_score

class Generator(nn.Module):
    def __init__(self, latent_size, n_classes=3):
        super().__init__()
        self.latent_size = latent_size
        self.n_classes = n_classes
        self.img_shape = (3, 64, 64)
        self.img_flat_size = 3 * 64 * 64
        
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

    def forward(self, x, labels):
        # x potrebbe essere [batch, latent, 1, 1] o [batch, latent]. Lo appiattiamo.
        x = x.view(x.size(0), -1)
        x = torch.cat([x, labels], dim=1)
        out = self.network(x)
        # Reshape in immagine
        return out.view(out.size(0), *self.img_shape)

    def train_step(self, discriminator, labels, optimizer, batch_size, device):
        # Clear generator gradients
        optimizer.zero_grad()

        # Generate fake images
        latent = torch.randn(batch_size, self.latent_size, 1, 1, device=device)
        # Il generatore prova a creare un'immagine che soddisfi le label fornite
        fake_images = self(latent, labels)

        # Try to fool the discriminator
        # Il discriminatore valuta se l'immagine generata sembra reale E coerente con le label
        preds = discriminator(fake_images, labels)
        targets = torch.ones(batch_size, 1, device=device)
        loss = F.binary_cross_entropy(preds, targets)

        # Update generator 
        loss.backward()
        optimizer.step()

        return loss.item()

def save_samples(index, latent_tensors, fixed_labels, generator, sample_dir, attr_names, show=True):
    fake_images = generator(latent_tensors, fixed_labels)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    
    images = denorm(fake_images).cpu().detach()
    n_images = images.size(0)
    nrow = int(math.sqrt(n_images))
    ncol = math.ceil(n_images / nrow)
    
    # Crea una figura con subplots per mostrare titoli individuali
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*2.5, nrow*3))
    axes = axes.flatten()
    
    for i in range(n_images):
        ax = axes[i]
        ax.imshow(images[i].permute(1, 2, 0))
        ax.set_xticks([]); ax.set_yticks([])
        
        # Crea il titolo basato sulle label
        title_parts = []
        for j, name in enumerate(attr_names):
            is_true = fixed_labels[i][j] > 0
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

def train_gan(EPOCHS, LEARNING_RATE, discriminator, generator, train_dl, device, input_noise, fixed_labels, sample_dir, attr_names, start_idx=1):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), LEARNING_RATE, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), LEARNING_RATE, betas=(0.5, 0.999))
    
    print("Inizio del training...")
    
    for epoch in range(EPOCHS):
        # Uso tqdm sul dataloader per vedere il progresso batch per batch
        pbar = tqdm(train_dl, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for i, (real_images, labels) in enumerate(pbar):
            real_images = real_images.to(device)
            labels = labels.to(device)
            
            # Train discriminator
            loss_d, real_score, fake_score = discriminator.train_step(real_images, labels, generator, opt_d)
            
            # Train generator
            loss_g = generator.train_step(discriminator, labels, opt_g, real_images.size(0), device)
            
            # Aggiorno la barra di tqdm con le loss correnti
            pbar.set_postfix({'loss_d': f'{loss_d:.4f}', 'loss_g': f'{loss_g:.4f}'})

        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(epoch+1, EPOCHS, loss_g, loss_d, real_score, fake_score))
        # Save generated images
        save_samples(epoch+start_idx, input_noise, fixed_labels, generator, sample_dir, attr_names, show=False)
    
    return losses_g, losses_d, real_scores, fake_scores

if __name__ == '__main__':
    
    DEVICE = get_device()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    IMG_DIR = os.path.join(current_dir, '../assets/archive/img_align_celeba/img_align_celeba/')
    ATTR_CSV = os.path.join(current_dir, '../assets/archive/list_attr_celeba.csv')
    
    SUBSET_DIM = 50000 
    LEARNING_RATE = 0.0002
    EPOCHS = 50
    
    latent_size = 128
    image_size = 64
    batch_size = 128
    n_classes = 4 # Male, Young, Blond, Smiling
    attr_names = ['Male', 'Young', 'Blond', 'Smiling'] # Nomi per la visualizzazione
    
    # 16 immagini per coprire tutte le combinazioni (2^4)
    generated_samples_count = 16

    train_dataset = CelebADataset(root_dir=IMG_DIR, csv_file=ATTR_CSV, transform=T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    if len(train_dataset) == 0:
        print("ERRORE: Nessuna immagine trovata. Verifica che il percorso IMG_DIR sia corretto.")
        exit()

    # Subset per velocizzare il training
    subset_size = min(SUBSET_DIM, len(train_dataset))
    train_dataset = torch.utils.data.Subset(train_dataset, torch.arange(subset_size))

    # Dataloader che divide il dataset in batch e mescola i dati ad ogni epoca.
    # 'num_workers' specifica il numero di subprocessi da usare per caricare le immagini dal disco alla RAM.
    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=(DEVICE.type == 'cuda'))
    
    #just for testing purposes...
    print(f"Immagini caricate: {len(train_dataset)}")
    #show_batch(train_dl)
    #plt.show()
    
    discriminator = Discriminator(n_classes=n_classes).to(DEVICE)
    generator = Generator(latent_size, n_classes=n_classes).to(DEVICE)
    
    input_noise = torch.randn(generated_samples_count, latent_size, 1, 1, device=DEVICE)

    # Directory per salvare le immagini generate durante il training
    generated_images_dir = 'gan.generated'
    os.makedirs(generated_images_dir, exist_ok=True)

    # Generiamo tutte le combinazioni possibili di attributi (2^4 = 16 combinazioni)
    labels_list = []
    for i in range(16):
        # Converte i in binario per ottenere le combinazioni
        l = [1 if (i >> (3-bit)) & 1 else -1 for bit in range(4)]
        labels_list.append(l)
    fixed_labels = torch.tensor(labels_list, device=DEVICE).float()

    # TRAINING:
    # Mi restituisce loss del generatore e del discriminatore per ogni epoca, insieme agli scores dei reali e falsi
    losses_g, losses_d, real_scores, fake_scores = train_gan(EPOCHS, LEARNING_RATE, discriminator, generator, train_dl, DEVICE, input_noise, fixed_labels, generated_images_dir, attr_names)
    
    models_dir = os.path.join(current_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(models_dir, 'gan.generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(models_dir, 'gan.discriminator.pth'))
    print(f"Modelli salvati in '{models_dir}'")
    
    # Plot losses
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(losses_g, label="G")
    plt.plot(losses_d, label="D")
    plt.xlabel("EPOCHS")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(generated_images_dir, 'losses.png'))
    plt.show()
    
    # Plot real and fake scores
    plt.figure(figsize=(10,5))
    plt.title("Real and Fake Scores During Training")
    plt.plot(real_scores, label="Real")
    plt.plot(fake_scores, label="Fake")
    plt.xlabel("EPOCHS")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(os.path.join(generated_images_dir, 'scores.png'))
    plt.show()
    
    # Generate final samples
    save_samples(EPOCHS+1, input_noise, fixed_labels, generator, generated_images_dir, attr_names)