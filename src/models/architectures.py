import torch.nn as nn
from torchvision import models

# 1. Baseline: Multi-Layer Perceptron (MLP)
# Serve per dimostrare i limiti di una rete densa classica sulle immagini
class MLPBaseline(nn.Module):
    def __init__(self, num_classes=4):
        super(MLPBaseline, self).__init__()
        self.flatten = nn.Flatten()
        # Le immagini sono 224x224 pixel con 3 canali (RGB).
        # Pertanto, l'input "appiattito" è 224 * 224 * 3 = 150528
        self.fc_layers = nn.Sequential(
            nn.Linear(150528, 512),
            nn.ReLU(),
            nn.Dropout(0.3), # Previene l'overfitting spegnendo neuroni casuali
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.fc_layers(x)

# 2. Modello Leggero: MobileNetV2
# CNN ottimizzata, ideale se ci sono limiti computazionali
def get_mobilenet(num_classes=4):
    # Carichiamo i pesi pre-addestrati (Transfer Learning)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Sostituiamo solo l'ultimo layer di classificazione per le nostre 4 classi
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

# 3. Modello Performante: ResNet-18
# CNN classica basata su connessioni residue per estrarre feature avanzate
def get_resnet18(num_classes=4):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Sostituiamo il fully connected layer (fc) finale
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Testiamo l'istanziazione per assicurarci che non ci siano errori
mlp_model = MLPBaseline(num_classes=4)
mobilenet_model = get_mobilenet(num_classes=4)
resnet_model = get_resnet18(num_classes=4)

print("✅ Architetture (MLP, MobileNet, ResNet) caricate e configurate per 4 classi!")