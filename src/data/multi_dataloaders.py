import torch
from torch.utils.data import DataLoader, random_split
# Importiamo gli strumenti che abbiamo appena creato nel file preprocessing
from src.data.multi_preprocessing import AlzheimerDataset, train_transforms, test_transforms, CLASSES # type: ignore

def get_dataloaders(train_dir, original_dir, batch_size=32):
    """
    Inizializza i dataset, li divide e restituisce i DataLoader.
    """
    # 1. Inizializzazione dei Dataset
    train_dataset = AlzheimerDataset(root_dir=train_dir, transform=train_transforms)
    original_dataset = AlzheimerDataset(root_dir=original_dir, transform=test_transforms)

    # 2. Suddivisione dell'OriginalDataset in Val (50%) e Test (50%)
    val_size = int(0.5 * len(original_dataset))
    test_size = len(original_dataset) - val_size

    val_dataset, test_dataset = random_split(
        original_dataset, [val_size, test_size],
        generator=torch.Generator().manual_seed(42) 
    )

    # 3. Creazione dei DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Stampiamo un resoconto
    print(f"✅ Dataloaders creati con successo!")
    print(f" - Batch per Addestramento: {len(train_loader)}")
    print(f" - Batch per Validazione: {len(val_loader)}")
    print(f" - Batch per Test: {len(test_loader)}")

    return train_loader, val_loader, test_loader
