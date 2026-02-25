import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    # setta il seed per la riproducibilità
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    # restituisce il dispositivo disponibile (GPU se disponibile, altrimenti CPU)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, path):
    # salva il modello in un file specificato
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    # carica il modello da un file specificato
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
