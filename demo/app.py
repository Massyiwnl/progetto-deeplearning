import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Aggiungiamo la root directory al path per poter importare da src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.architectures import get_mobilenet # Assicurati di aver implementato questo file

# --- CONFIGURAZIONE ---
CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
st.set_page_config(page_title="NeuroVision DL", page_icon="🧠", layout="centered")

# --- TRASFORMAZIONI (Stesse del Test Set) ---
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- FUNZIONE PER CARICARE IL MODELLO ---
@st.cache_resource # Evita di ricaricare il modello a ogni click
def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "MobileNetV2":
        model = get_mobilenet(num_classes=4)
        # Sostituisci il path con quello dove hai salvato i pesi
        model.load_state_dict(torch.load("/content/unict-dlcmm-Maccarrone-Brancaforte-Cassia/results/mobilenet_alzheimer.pth", map_location=device))
        
    # Qui potrai aggiungere gli "elif" per ResNet o MLP quando li avrai addestrati
    
    model = model.to(device)
    model.eval()
    return model, device

# --- INTERFACCIA UTENTE ---
st.title("🧠 NeuroVision DL: Alzheimer MRI Classification")
st.write("Progetto DLCMM - Carica un'immagine MRI per analizzare lo stadio di deterioramento cognitivo.")

# Selettore del Modello
model_choice = st.selectbox("Seleziona il modello di Deep Learning:", ["MobileNetV2", "ResNet-18", "MLP"])

# Uploader dell'immagine
uploaded_file = st.file_uploader("Carica una scansione MRI (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Mostriamo l'immagine caricata
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Immagine MRI Caricata', use_container_width=True)
    
    if st.button("Analizza Immagine"):
        with st.spinner('Analisi in corso con ' + model_choice + '...'):
            try:
                # 1. Carichiamo il modello e processiamo l'immagine
                model, device = load_model(model_choice)
                img_tensor = test_transforms(image).unsqueeze(0).to(device)
                
                # 2. Inferenza
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = F.softmax(outputs, dim=1)[0] * 100 # Percentuali
                    _, preds = torch.max(outputs, 1)
                
                predicted_class = CLASSES[preds.item()]
                
                # 3. Mostriamo il risultato predetto
                st.success(f"Diagnosi Predetta: **{predicted_class}**")
                
                # 4. Grafico a barre delle probabilità
                st.write("### Confidenza del Modello")
                fig, ax = plt.subplots(figsize=(8, 4))
                y_pos = np.arange(len(CLASSES))
                ax.barh(y_pos, probs.cpu().numpy(), align='center', color='skyblue')
                ax.set_yticks(y_pos, labels=CLASSES)
                ax.invert_yaxis()  # La classe principale in alto
                ax.set_xlabel('Probabilità (%)')
                st.pyplot(fig)
                
            except FileNotFoundError:
                st.error("Errore: Impossibile trovare i pesi del modello. Assicurati di aver addestrato il modello e salvato il file in `results/mobilenet_alzheimer.pth`")