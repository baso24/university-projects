import os
import torch
from torchvision.utils import make_grid
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

from gan import get_device, denorm, Generator as GANGenerator
from dcgan import Generator as DCGANGenerator

def generate_and_show():
    # Prendo il modello selezionato
    selected_model = var_model_type.get()
    
    if selected_model not in loaded_models:
        print(f"Error: Model {selected_model} not loaded.")
        return
        
    # Prendo il generatore
    generator = loaded_models[selected_model]

    # Prendo il numero di immagini da generare
    num_img = var_num_images.get()
    
    # Costruzione labels in base ai menu a tendina
    val_male = 1.0 if var_gender.get() == "Male" else -1.0
    val_young = 1.0 if var_age.get() == "Young" else -1.0
    val_blond = 1.0 if var_hair.get() == "Blond" else -1.0
    val_smiling = 1.0 if var_smiling.get() == "Smiling" else -1.0
    labels = torch.tensor([val_male, val_young, val_blond, val_smiling], device=DEVICE).float().unsqueeze(0).repeat(num_img, 1)
    
    # Aggiorna titolo
    title_parts = []
    title_parts.append("Male" if val_male > 0 else "Female")
    title_parts.append("Young" if val_young > 0 else "Old")
    title_parts.append("Blond" if val_blond > 0 else "Not Blond")
    title_parts.append("Smiling" if val_smiling > 0 else "Not Smiling")
    if selected_model == "DCGAN":
        title_parts.append("DCGAN" + ", ".join(title_parts))
    elif selected_model == "GAN":
        title_parts.append("GAN"+ ", ".join(title_parts))

    # Generazione immagini, genero il noise e poi lo passo al generatore che genera le fake images
    # La size corrisponde al numero di immagini che voglio generare
    noise = torch.randn(num_img, latent_size, 1, 1, device=DEVICE)
    with torch.no_grad():
        fake_images = generator(noise, labels)
    
    # Sistemazione griglia di immagini generate
    grid = make_grid(denorm(fake_images.cpu()), nrow=int(num_img**0.5), padding=2)
    ndarr = grid.permute(1, 2, 0).numpy()
    im_arr = (ndarr * 255).astype(np.uint8)
    img = Image.fromarray(im_arr)
    img = img.resize((600, 600), Image.NEAREST)
    imgtk = ImageTk.PhotoImage(image=img)
    lbl_img.imgtk = imgtk
    lbl_img.configure(image=imgtk)

if __name__ == '__main__':
    DEVICE = get_device()
    latent_size = 128
    n_classes = 4
    
    # Caratteristiche da poter scegliere per generare i volti
    GENERATE_MALE = True  # True per generare un uomo, False per una donna
    GENERATE_YOUNG = True # True per giovane, False per anziano
    GENERATE_BLOND = True # True per biondo, False per non biondo
    GENERATE_SMILING = True # True per sorridente, False per non sorridente
    
    # Prendo la directory dove sono caricati i modelli
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, 'models')
    loaded_models = {}
    
    # Carico i modelli
    print("Loading DCGAN model...")
    dcgan_path = os.path.join(models_dir, 'dcgan.generator.pth')
    if os.path.exists(dcgan_path):
        dcgan_gen = DCGANGenerator(latent_size, n_classes).to(DEVICE)
        dcgan_gen.load_state_dict(torch.load(dcgan_path, map_location=DEVICE))
        dcgan_gen.eval()
        loaded_models['DCGAN'] = dcgan_gen
    else:
        print(f"Warning: {dcgan_path} not found")
        
    print("Loading GAN model...")
    gan_path = os.path.join(models_dir, 'gan.generator.pth')
    if os.path.exists(gan_path):
        gan_gen = GANGenerator(latent_size, n_classes).to(DEVICE)
        gan_gen.load_state_dict(torch.load(gan_path, map_location=DEVICE))
        gan_gen.eval()
        loaded_models['GAN'] = gan_gen
    else:
        print(f"Warning: {gan_path} not found")
    
    # Inizio GUI Tkinter
    root = tk.Tk()
    root.title("GAN vs DCGAN: Human faces generator")

    # Variabili
    var_num_images = tk.IntVar(value=16)
    var_gender = tk.StringVar(value="Male" if GENERATE_MALE else "Female")
    var_age = tk.StringVar(value="Young" if GENERATE_YOUNG else "Old")
    var_hair = tk.StringVar(value="Blond" if GENERATE_BLOND else "Not Blond")
    var_smiling = tk.StringVar(value="Smiling" if GENERATE_SMILING else "Not Smiling")
    var_model_type = tk.StringVar(value="DCGAN")

    # A sinistra i controlli 
    frame_left = tk.Frame(root)
    frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

    # A destra l'immagine generata
    frame_right = tk.Frame(root)
    frame_right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

    # Frame per i controlli
    frame_controls = tk.Frame(frame_left)
    frame_controls.pack(pady=5)

    # Selezione Modello
    frame_model = tk.LabelFrame(frame_controls, text="Model")
    frame_model.pack(fill="x", pady=5)
    tk.Radiobutton(frame_model, text="DCGAN", variable=var_model_type, value="DCGAN").pack(side=tk.LEFT, padx=10)
    tk.Radiobutton(frame_model, text="GAN", variable=var_model_type, value="GAN").pack(side=tk.LEFT, padx=10)

    # Selezione numero immagini
    frame_num = tk.LabelFrame(frame_controls, text="Number of Images")
    frame_num.pack(fill="x", pady=5)
    tk.Radiobutton(frame_num, text="1", variable=var_num_images, value=1).pack(side=tk.LEFT, padx=10)
    tk.Radiobutton(frame_num, text="4", variable=var_num_images, value=4).pack(side=tk.LEFT, padx=10)
    tk.Radiobutton(frame_num, text="16", variable=var_num_images, value=16).pack(side=tk.LEFT, padx=10)

    # Selezione attributi (Select/OptionMenu)
    frame_attrs = tk.LabelFrame(frame_controls, text="Attributes")
    frame_attrs.pack(fill="x", pady=5)
    tk.OptionMenu(frame_attrs, var_gender, "Male", "Female").pack(side=tk.LEFT, padx=5)
    tk.OptionMenu(frame_attrs, var_age, "Young", "Old").pack(side=tk.LEFT, padx=5)
    tk.OptionMenu(frame_attrs, var_hair, "Blond", "Not Blond").pack(side=tk.LEFT, padx=5)
    tk.OptionMenu(frame_attrs, var_smiling, "Smiling", "Not Smiling").pack(side=tk.LEFT, padx=5)

    # Bottone per generare
    btn_regen = tk.Button(frame_left, text="Regenerate", command=generate_and_show, font=("Arial", 14))
    btn_regen.pack(pady=10)

    lbl_img = tk.Label(frame_right)
    lbl_img.pack(padx=10, pady=10)

    # Generazione iniziale al partire della GUI
    generate_and_show()
    root.mainloop()
