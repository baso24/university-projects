import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

# Funzione per denormalizzare le immagini (da -1,1 a 0,1)
def denorm(img_tensors):
    return img_tensors * 0.5 + 0.5

# Funzione per ottenere il device
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# Definizione del Discriminatore (identica a gan-test.py)
class Discriminator(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.n_classes = n_classes
        
        self.network = nn.Sequential(
            nn.Conv2d(3 + n_classes, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        labels = labels.view(labels.size(0), self.n_classes, 1, 1)
        labels = labels.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, labels], dim=1)
        return self.network(x)

# Definizione del Generatore (identica a gan-test.py)
class Generator(nn.Module):
    def __init__(self, latent_size, n_classes=3):
        super().__init__()
        self.latent_size = latent_size
        self.n_classes = n_classes
        
        self.network = nn.Sequential(
            nn.ConvTranspose2d(latent_size + n_classes, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x, labels):
        labels = labels.view(labels.size(0), self.n_classes, 1, 1)
        x = torch.cat([x, labels], dim=1)
        return self.network(x)

if __name__ == '__main__':
    DEVICE = get_device()
    latent_size = 128
    n_classes = 3
    
    # --- CONFIGURAZIONE ---
    GENERATE_MALE = False  # True per generare un uomo, False per una donna
    GENERATE_YOUNG = True # True per giovane, False per vecchio
    GENERATE_BLOND = True # True per biondo, False per non biondo
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, 'models')
    
    # Inizializzazione modelli
    generator = Generator(latent_size, n_classes).to(DEVICE)
    discriminator = Discriminator(n_classes).to(DEVICE)
    
    # Caricamento pesi
    print(f"Caricamento modelli da {models_dir}...")
    generator.load_state_dict(torch.load(os.path.join(models_dir, 'generator.pth'), map_location=DEVICE))
    discriminator.load_state_dict(torch.load(os.path.join(models_dir, 'discriminator.pth'), map_location=DEVICE))
    
    generator.eval()
    discriminator.eval()
    
    # --- GUI TKINTER ---
    root = tk.Tk()
    root.title("GAN Generator")

    # Variabili
    var_num_images = tk.IntVar(value=16)
    var_gender = tk.StringVar(value="Uomo" if GENERATE_MALE else "Donna")
    var_age = tk.StringVar(value="Giovane" if GENERATE_YOUNG else "Vecchio")
    var_hair = tk.StringVar(value="Biondi" if GENERATE_BLOND else "Non Biondi")

    # Layout principale: sinistra (controlli) e destra (immagine)
    frame_left = tk.Frame(root)
    frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

    frame_right = tk.Frame(root)
    frame_right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

    # Frame per i controlli
    frame_controls = tk.Frame(frame_left)
    frame_controls.pack(pady=5)

    # Selezione numero immagini
    frame_num = tk.LabelFrame(frame_controls, text="Numero Immagini")
    frame_num.pack(fill="x", pady=5)
    tk.Radiobutton(frame_num, text="1", variable=var_num_images, value=1).pack(side=tk.LEFT, padx=10)
    tk.Radiobutton(frame_num, text="4", variable=var_num_images, value=4).pack(side=tk.LEFT, padx=10)
    tk.Radiobutton(frame_num, text="16", variable=var_num_images, value=16).pack(side=tk.LEFT, padx=10)

    # Selezione attributi (Select/OptionMenu)
    frame_attrs = tk.LabelFrame(frame_controls, text="Attributi")
    frame_attrs.pack(fill="x", pady=5)
    
    tk.OptionMenu(frame_attrs, var_gender, "Uomo", "Donna").pack(side=tk.LEFT, padx=5)
    tk.OptionMenu(frame_attrs, var_age, "Giovane", "Vecchio").pack(side=tk.LEFT, padx=5)
    tk.OptionMenu(frame_attrs, var_hair, "Biondi", "Non Biondi").pack(side=tk.LEFT, padx=5)

    def generate_and_show():
        num_img = var_num_images.get()
        
        # Costruzione labels in base ai menu a tendina
        val_male = 1.0 if var_gender.get() == "Uomo" else -1.0
        val_young = 1.0 if var_age.get() == "Giovane" else -1.0
        val_blond = 1.0 if var_hair.get() == "Biondi" else -1.0
        labels = torch.tensor([val_male, val_young, val_blond], device=DEVICE).float().unsqueeze(0).repeat(num_img, 1)
        
        # Aggiorna titolo
        title_parts = []
        title_parts.append("Male" if val_male > 0 else "Female")
        title_parts.append("Young" if val_young > 0 else "Old")
        title_parts.append("Blond" if val_blond > 0 else "Not Blond")
        root.title("GAN: " + ", ".join(title_parts))

        noise = torch.randn(num_img, latent_size, 1, 1, device=DEVICE)
        with torch.no_grad():
            fake_images = generator(noise, labels)
        
        grid = make_grid(denorm(fake_images.cpu()), nrow=int(num_img**0.5), padding=2)
        ndarr = grid.permute(1, 2, 0).numpy()
        im_arr = (ndarr * 255).astype(np.uint8)
        img = Image.fromarray(im_arr)
        img = img.resize((600, 600), Image.NEAREST)
        
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_img.imgtk = imgtk
        lbl_img.configure(image=imgtk)

    btn_regen = tk.Button(frame_left, text="Rigenera", command=generate_and_show, font=("Arial", 14))
    btn_regen.pack(pady=10)

    lbl_img = tk.Label(frame_right)
    lbl_img.pack(padx=10, pady=10)

    generate_and_show()
    root.mainloop()
