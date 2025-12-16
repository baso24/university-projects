import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm

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

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # in: 3 x 64 x 64
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
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

            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

    def train_step(self, real_images, generator, optimizer):
        # Clear discriminator gradients
        optimizer.zero_grad()
        
        device = real_images.device
        batch_size = real_images.size(0)

        # Pass real images through discriminator
        real_preds = self(real_images)
        real_targets = torch.ones(real_images.size(0), 1, device=device)
        real_loss = F.binary_cross_entropy(real_preds, real_targets)
        real_score = torch.mean(real_preds).item()

        # Generate fake images
        latent = torch.randn(batch_size, generator.latent_size, 1, 1, device=device)
        fake_images = generator(latent)

        # Pass Fake images through discriminator
        fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
        fake_preds = self(fake_images)
        fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
        fake_score = torch.mean(fake_preds).item()

        # Update discriminator weights
        loss = real_loss + fake_loss
        loss.backward()
        optimizer.step()
        return loss.item(), real_score, fake_score

class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.network = nn.Sequential(
            # in: latent_size x 1 x 1
            nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
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

    def forward(self, x):
        return self.network(x)

    def train_step(self, discriminator, optimizer, batch_size, device):
        # Clear generator gradients
        optimizer.zero_grad()

        # Generate fake images
        latent = torch.randn(batch_size, self.latent_size, 1, 1, device=device)
        fake_images = self(latent)

        # Try to fool the discriminator
        preds = discriminator(fake_images)
        targets = torch.ones(batch_size, 1, device=device)
        loss = F.binary_cross_entropy(preds, targets)

        # Update generator 
        loss.backward()
        optimizer.step()

        return loss.item()

def save_samples(index, latent_tensors, generator, sample_dir, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    
    # Calcola nrow automaticamente per fare una griglia quadrata (es. 16 img -> 4x4)
    grid_nrow = int(math.sqrt(latent_tensors.size(0)))
    
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=grid_nrow)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=grid_nrow).permute(1, 2, 0))
        plt.show()

def train_gan(epochs, lr, discriminator, generator, train_dl, device, input_noise, sample_dir, start_idx=1):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    print("Inizio del training...")
    
    for epoch in range(epochs):
        # Uso tqdm sul dataloader per vedere il progresso batch per batch
        pbar = tqdm(train_dl, desc=f"Epoch [{epoch+1}/{epochs}]")
        for i, (real_images, _) in enumerate(pbar):
            real_images = real_images.to(device)
            
            # Train discriminator
            loss_d, real_score, fake_score = discriminator.train_step(real_images, generator, opt_d)
            
            # Train generator
            loss_g = generator.train_step(discriminator, opt_g, real_images.size(0), device)
            
            # Aggiorno la barra di tqdm con le loss correnti
            pbar.set_postfix({'loss_d': f'{loss_d:.4f}', 'loss_g': f'{loss_g:.4f}'})

        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
        # Save generated images
        save_samples(epoch+start_idx, input_noise, generator, sample_dir, show=False)
    
    return losses_g, losses_d, real_scores, fake_scores

if __name__ == '__main__':
    
    DEVICE = get_device()
    print(f'Using device: {DEVICE}')
    
    # ImageFolder richiede il percorso della cartella genitore (root) che contiene le sottocartelle delle classi.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATASET_ROOT = os.path.join(current_dir, '../assets/archive/')

    latent_size = 128
    image_size = 64
    batch_size = 512

    train_dataset = ImageFolder(root=DATASET_ROOT, transform=T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    # Uso un sottoinsieme del dataset per velocizzare il training, prendo le prime 10k
    if len(train_dataset) > 10000:
        indices = torch.arange(10000)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    # Dataloader che divide il dataset in batch e mescola i dati ad ogni epoca.
    # 'num_workers' specifica il numero di subprocessi da usare per caricare le immagini dal disco alla RAM.
    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=(DEVICE.type == 'cuda'))
    
    #just for testing purposes...
    print(f"Immagini caricate: {len(train_dataset)}")
    #show_batch(train_dl)
    #plt.show()
    
    discriminator = Discriminator().to(DEVICE)
    generator = Generator(latent_size).to(DEVICE)

    sample_dir = 'generated'
    os.makedirs(sample_dir, exist_ok=True)
    
    # Scegliere quante immagini generare alla fine di ogni epoca (1, 4, 16, 64...)
    generated_samples_count = 1
    
    input_noise = torch.randn(generated_samples_count, latent_size, 1, 1, device=DEVICE)
    
    lr = 0.0003
    epochs = 100

    history = train_gan(epochs, lr, discriminator, generator, train_dl, DEVICE, input_noise, sample_dir)