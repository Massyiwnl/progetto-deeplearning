import os
import shutil
from PIL import Image

try:
    from torchvision import transforms
except ImportError:
    raise ImportError("torchvision non è installato.")

IMG_SIZE = 224

# Definisce le trasformazioni per il preprocessing delle immagini
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Preprocessamento dell'immagine
def preprocess_image(img_path, save_path):
    image = Image.open(img_path).convert("L") # converte in scala di grigi
    image = transform(image) # applica le trasformazioni
    os.makedirs(os.path.dirname(save_path), exist_ok=True) # crea la directory se non esiste
    image = transforms.ToPILImage()(image) # converte di nuovo in immagine PIL per il salvataggio
    image.save(save_path) # salva l'immagine preprocessata

# Crea un dataset binario a partire dal dataset originale
def create_binary_dataset(raw_dir, output_dir):
    alzheimer_classes = [
        "MildDemented",
        "ModerateDemented",
        "VeryMildDemented"
    ]

    for class_name in os.listdir(raw_dir): # itera sulle classi nel dataset originale
        class_path = os.path.join(raw_dir, class_name) # percorso della classe
        if not os.path.isdir(class_path): # se non è una directory
            continue

        # determina l'etichetta binaria
        label = "Alzheimer" if class_name in alzheimer_classes else "NonAlzheimer"
        # problema: le due classi sono sbilanciate.
        # -> tecniche di bilanciamento successivamente
        # tipo oversampling, data augmentation, ecc.

        # crea la directory di output per l'etichetta
        for img in os.listdir(class_path):
            src = os.path.join(class_path, img) # percorso dell'immagine originale
            dst = os.path.join(output_dir, label, img) # percorso di destinazione
            preprocess_image(src, dst) # preprocessa e salva l'immagine
