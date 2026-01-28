import os
import torch
from torchvision.utils import make_grid
from tkinter import ttk
import tkinter as tk
from ttkthemes import ThemedTk
from PIL import Image, ImageTk
import numpy as np

from train import DCGANGenerator, GANGenerator, get_device, denorm

def generate_and_show():
    # Prendo il modello selezionato leggendo il pulsante selezionato
    selected_model = var_model_type.get()
    
    # Controllo errori
    if selected_model not in loaded_models:
        print(f"Error: Model {selected_model} not loaded.")
        return
        
    # Prendo il generatore
    generator = loaded_models[selected_model]

    # Prendo il numero di immagini da generare leggendo il pulsante selezionato
    num_img = var_num_images.get()
    
    # Costruzione labels in base ai menu a tendina
    # Ogni attributo deve essere rappresentato come 1.0 o -1.0 per la rete
    val_male = 1.0 if var_gender.get() == "Male" else -1.0
    val_young = 1.0 if var_age.get() == "Young" else -1.0
    val_blond = 1.0 if var_hair.get() == "Blond" else -1.0
    val_smiling = 1.0 if var_smiling.get() == "Smiling" else -1.0
    # Creo tensore delle lables con le caratteristiche selezionate, aggiungo una dimensione con unsqueeze e lo ripeto per il numero di immagini
    # Se voglio generare 16 immagini diventerà una matrice 16x4
    labels = torch.tensor([val_male, val_young, val_blond, val_smiling], device=DEVICE).float().unsqueeze(0).repeat(num_img, 1)
    
    # Aggiornamento del titolo
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
    # noise dovrà essere delle dimensioni [num_img, latent_size, 1, 1]
    if selected_model == "DCGAN":
        noise = torch.randn(num_img, latent_size, 1, 1, device=DEVICE)
    elif selected_model == "GAN":
         noise = torch.randn(num_img, latent_size, device=DEVICE)
    with torch.no_grad():
        fake_images = generator(noise, labels)
    
    # Sistemazione in griglia delle immagini generate
    # nrow è il numero di immagini per riga, lo calcolo come radice quadrata del numero totale di immagini
    grid = make_grid(denorm(fake_images.cpu()), nrow=int(num_img**0.5), padding=2)
    # Conversione in immagine visualizzabile da Tkinter, dalla rete infatti ottengo un tensore torch organizzato come  (Canali, Altezza, Larghezza)
    # Tkinter vuole (Altezza, Larghezza, Canali) e i valori dei pixel devono essere in [0, 255] come uint8
    ndarr = grid.permute(1, 2, 0).numpy()
    image_arr = (ndarr * 255).astype(np.uint8)
    image = Image.fromarray(image_arr)
    image = image.resize((600, 600), Image.NEAREST)
    img_for_tkinter = ImageTk.PhotoImage(image=image)
    lbl_img.img_for_tkinter = img_for_tkinter
    lbl_img.configure(image=img_for_tkinter)

if __name__ == '__main__':
    DEVICE = get_device()

    # Parametri fissi
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
    root = ThemedTk(theme="arc")
    root.title("GAN and DCGAN: Human faces generator")

    # Configurazione stile per il bottone 
    style = ttk.Style()
    style.configure("Big.TButton", font=("Arial", 14))

    # Variabili
    var_num_images = tk.IntVar(value=16)
    var_gender = tk.StringVar(value="Male" if GENERATE_MALE else "Female")
    var_age = tk.StringVar(value="Young" if GENERATE_YOUNG else "Old")
    var_hair = tk.StringVar(value="Blond" if GENERATE_BLOND else "Not Blond")
    var_smiling = tk.StringVar(value="Smiling" if GENERATE_SMILING else "Not Smiling")
    var_model_type = tk.StringVar(value="DCGAN")

    # A sinistra i controlli 
    frame_left = ttk.Frame(root)
    frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

    # A destra l'immagine generata
    frame_right = ttk.Frame(root)
    frame_right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
    # Frame per i controlli
    frame_controls = ttk.Frame(frame_left)
    frame_controls.pack(pady=5)

    # Selezione Modello
    frame_model = ttk.LabelFrame(frame_controls, text="Model")
    frame_model.pack(fill="x", pady=5)
    ttk.Radiobutton(frame_model, text="DCGAN", variable=var_model_type, value="DCGAN").pack(side=tk.LEFT, padx=10)
    ttk.Radiobutton(frame_model, text="GAN", variable=var_model_type, value="GAN").pack(side=tk.LEFT, padx=10)

    # Selezione numero immagini
    frame_num = ttk.LabelFrame(frame_controls, text="Number of Images")
    frame_num.pack(fill="x", pady=5)
    ttk.Radiobutton(frame_num, text="1", variable=var_num_images, value=1).pack(side=tk.LEFT, padx=10)
    ttk.Radiobutton(frame_num, text="4", variable=var_num_images, value=4).pack(side=tk.LEFT, padx=10)
    ttk.Radiobutton(frame_num, text="16", variable=var_num_images, value=16).pack(side=tk.LEFT, padx=10)

    # Selezione attributi (Select/OptionMenu)
    frame_attrs = ttk.LabelFrame(frame_controls, text="Attributes")
    frame_attrs.pack(fill="x", pady=5)
    ttk.OptionMenu(frame_attrs, var_gender, var_gender.get(), "Male", "Female").pack(side=tk.LEFT, padx=5)
    ttk.OptionMenu(frame_attrs, var_age, var_age.get(), "Young", "Old").pack(side=tk.LEFT, padx=5)
    ttk.OptionMenu(frame_attrs, var_hair, var_hair.get(), "Blond", "Not Blond").pack(side=tk.LEFT, padx=5)
    ttk.OptionMenu(frame_attrs, var_smiling, var_smiling.get(), "Smiling", "Not Smiling").pack(side=tk.LEFT, padx=5)

    # Bottone per generare
    btn_regen = ttk.Button(frame_left, text="Regenerate", command=generate_and_show, style="Big.TButton")
    btn_regen.pack(pady=10)

    lbl_img = ttk.Label(frame_right)
    lbl_img.pack(padx=10, pady=10)

    # Generazione iniziale al partire della GUI
    generate_and_show()

    root.mainloop()
